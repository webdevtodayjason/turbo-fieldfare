import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-03 wave-2: Q/KV LoRA epilogue for one decode token — wq_a down-proj
/// -> RMSNorm -> latent qr -> wq_b up-proj -> weight-free per-head renorm ->
/// trailing RoPE for queries; wkv + kv_norm + RoPE written straight into the
/// window ring slot; fp32 wkv/wgate compressor projections; indexer q from
/// the shared qr. CPU reference mirrors each stage including the kernel's
/// fp16 round-trip points.
@Suite struct V4QKVEpilogueTests {

    private static let dim = 4096
    private static let qLora = 1024
    private static let heads = 64
    private static let headDim = 512
    private static let indexDim = 128
    private static let position = 137

    // MARK: - CPU reference pieces

    private static func yarnFreq(_ i: Int,
                                 theta: Float = 160_000,
                                 factor: Float = 16,
                                 origSeqLen: Float = 65_536,
                                 betaFast: Float = 32,
                                 betaSlow: Float = 1) -> Float {
        let d: Float = 64
        let freq = pow(theta, -2 * Float(i) / d)
        let logBase = log(theta)
        let lowRot = d * log(origSeqLen / (betaFast * 2 * .pi)) / (2 * logBase)
        let highRot = d * log(origSeqLen / (betaSlow * 2 * .pi)) / (2 * logBase)
        let low = max(floor(lowRot), 0)
        var high = min(ceil(highRot), d - 1)
        if high == low { high = low + 0.001 }
        let ramp = min(max((Float(i) - low) / (high - low), 0), 1)
        let smooth = 1 - ramp
        return (freq / factor) * (1 - smooth) + freq * smooth
    }

    private static func fp16Round(_ v: [Float]) -> [Float] {
        v.map { Float(Float16($0)) }
    }

    private static func rmsnorm(_ v: [Float], gamma: [Float]) -> [Float] {
        let ms = v.reduce(0) { $0 + $1 * $1 } / Float(v.count)
        let rs = 1 / (ms + 1e-6).squareRoot()
        return (0..<v.count).map { v[$0] * rs * gamma[$0] }
    }

    private static func renormPerHead(_ v: [Float], heads: Int, headDim: Int) -> [Float] {
        var out = v
        for h in 0..<heads {
            let base = h * headDim
            var ms: Float = 0
            for i in 0..<headDim { ms += out[base + i] * out[base + i] }
            let rs = 1 / (ms / Float(headDim) + 1e-6).squareRoot()
            for i in 0..<headDim { out[base + i] *= rs }
        }
        return out
    }

    private static func ropeTrailing(_ v: [Float], rows: Int, width: Int,
                                     position: Float) -> [Float] {
        var x = v
        for r in 0..<rows {
            let base = r * width + (width - 64)
            for i in 0..<32 {
                let angle = position * yarnFreq(i)
                let cs = cos(angle), sn = sin(angle)
                let x0 = x[base + 2 * i], x1 = x[base + 2 * i + 1]
                x[base + 2 * i] = x0 * cs - x1 * sn
                x[base + 2 * i + 1] = x0 * sn + x1 * cs
            }
        }
        return x
    }

    private static func hadamardFP4QAT(_ v: [Float], rows: Int, width: Int) -> [Float] {
        precondition(width == 128)
        let fp4: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        let invSqrt2 = 1 / Float(2).squareRoot()
        var out = v
        for row in 0..<rows {
            let base = row * width
            var x = Array(out[base..<(base + width)])
            var stride = 1
            while stride < width {
                var block = 0
                while block < width {
                    for k in 0..<stride {
                        let a = x[block + k]
                        let b = x[block + k + stride]
                        x[block + k] = (a + b) * invSqrt2
                        x[block + k + stride] = (a - b) * invSqrt2
                    }
                    block += stride * 2
                }
                stride *= 2
            }
            for group in 0..<(width / 32) {
                let lo = group * 32
                let hi = lo + 32
                let amax = x[lo..<hi].reduce(Float.zero) { max($0, abs($1)) }
                let scale = pow(2, ceil(log2(max(amax, 6 * pow(2, -126)) / 6)))
                for i in lo..<hi {
                    let q = min(abs(x[i] / scale), 6)
                    var best = 0
                    var bestDiff = Float.infinity
                    for code in 0..<fp4.count {
                        let diff = abs(fp4[code] - q)
                        if diff < bestDiff || (diff == bestDiff && code.isMultiple(of: 2)) {
                            best = code
                            bestDiff = diff
                        }
                    }
                    let signed = x[i] < 0 ? -fp4[best] : fp4[best]
                    x[i] = Float(Float16(signed * scale))
                }
            }
            out.replaceSubrange(base..<(base + width), with: x)
        }
        return out
    }

    private static func randomFP8(m: Int, n: Int, rng: inout SplitMix64)
        -> V4Quantization.FP8BlockMatrix {
        let codes = (0..<(m * n)).map { _ -> UInt8 in
            var b = UInt8(clamping: Int(rng.uniform(0, 256))) & 0x7F
            if b == 0x7F { b = 0x7E }
            if rng.uniform(0, 1) < 0.5 { b |= 0x80 }
            return b
        }
        let scales = (0..<((m / 128) * (n / 128))).map { _ -> UInt8 in
            UInt8(clamping: 127 + Int(rng.uniform(-6, -2)))
        }
        return V4Quantization.FP8BlockMatrix(m: m, n: n, codes: codes, scales: scales)
    }

    // MARK: - Full decode path

    @Test func decodePath_matchesCPUReferenceChain() throws {
        var rng = SeedTree(0xE01).key("qkv-epilogue")
        let x16 = (0..<Self.dim).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let x = x16.map { Float($0) }
        let wqA = Self.randomFP8(m: Self.qLora, n: Self.dim, rng: &rng)
        let wqB = Self.randomFP8(m: Self.heads * Self.headDim, n: Self.qLora, rng: &rng)
        let wkv = Self.randomFP8(m: Self.headDim, n: Self.dim, rng: &rng)
        let widx = Self.randomFP8(m: Self.heads * Self.indexDim, n: Self.qLora, rng: &rng)
        let qGamma = (0..<Self.qLora).map { _ in rng.uniform(0.5, 1.5) }
        let kvGamma = (0..<Self.headDim).map { _ in rng.uniform(0.5, 1.5) }
        let s = 1 / Float(Self.dim).squareRoot()
        let wkvC = (0..<(1024 * Self.dim)).map { _ in rng.uniform(-s, s) }
        let wgateC = (0..<(1024 * Self.dim)).map { _ in rng.uniform(-s, s) }

        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: [4])
        let kvStore = try CompressedKVCacheManager(device: ctx.device,
                                                   config: config, maxContext: 4096)
        kvStore.advance(by: Self.position + 1)
        let epi = try V4QKVEpilogue(device: ctx.device)

        func bytes<T>(_ values: [T]) -> MTLBuffer? {
            ctx.device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride,
                                  options: .storageModeShared)
        }
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x16),
              let wqAC = bytes(wqA.codes), let wqAS = bytes(wqA.scales),
              let wqBC = bytes(wqB.codes), let wqBS = bytes(wqB.scales),
              let wkvC8 = bytes(wkv.codes), let wkvS8 = bytes(wkv.scales),
              let widxC = bytes(widx.codes), let widxS = bytes(widx.scales),
              let qGB = bytes(qGamma), let kvGB = bytes(kvGamma),
              let wkvCB = bytes(wkvC), let wgateCB = bytes(wgateC),
              let qOut = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim),
              let idxOut = Fp16Buffer.make(ctx.device, count: Self.heads * Self.indexDim),
              let compKVOut = ctx.device.makeBuffer(length: 1024 * 4, options: .storageModeShared),
              let compGateOut = ctx.device.makeBuffer(length: 1024 * 4, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        var w = V4QKVEpilogue.Weights(wqA: (wqAC, wqAS), wqB: (wqBC, wqBS),
                                      qNormGamma: qGB,
                                      windowWKV: (wkvC8, wkvS8), kvNormGamma: kvGB)
        w.compressorWKV = wkvCB
        w.compressorWGate = wgateCB
        w.compressorOutDim = 1024
        w.indexerWqB = (widxC, widxS)

        let slot = kvStore.windowSlot(layer: 0, position: Self.position)
        let cb = ctx.queue.makeCommandBuffer()!
        epi.encodeDecode(commandBuffer: cb,
                         x: xBuf, position: Self.position,
                         weights: w, rope: .compressedLayer,
                         qOut: qOut,
                         windowSlot: .init(buffer: slot.buffer, offset: slot.offset),
                         compressorWKVOut: compKVOut,
                         compressorWGateOut: compGateOut,
                         indexQOut: idxOut)
        cb.commit(); cb.waitUntilCompleted()

        // Q reference: gemv -> fp16 -> rmsnorm -> fp16 -> gemv -> fp16 ->
        // renorm -> fp16 -> rope.
        let qrRef = Self.fp16Round(Self.rmsnorm(
            Self.fp16Round(V4Quantization.gemvFP8(matrix: wqA, x: x)), gamma: qGamma))
        let qRef = Self.ropeTrailing(
            Self.fp16Round(Self.renormPerHead(
                Self.fp16Round(V4Quantization.gemvFP8(matrix: wqB, x: qrRef)),
                heads: Self.heads, headDim: Self.headDim)),
            rows: Self.heads, width: Self.headDim, position: Float(Self.position))
        let qGot = Fp16Buffer.read(qOut, count: Self.heads * Self.headDim)
        let relQ = RelError.compute(actual: qGot, reference: qRef)
        #expect(relQ < Tolerance.fp16ChainedReduction, "q rel=\(relQ)")

        // Latent qr must also match (it feeds the indexer).
        let qrGot = Fp16Buffer.read(epi.qrBuffer, count: Self.qLora)
        let relQR = RelError.compute(actual: qrGot, reference: qrRef)
        #expect(relQR < Tolerance.fp16Reduction, "qr rel=\(relQR)")

        // Window KV reference, at the ring slot for position 137. The
        // checkpoint was trained with in-place FP8 simulation on the 448
        // non-RoPE channels after norm/RoPE; the final 64 channels remain
        // FP16 for positional precision.
        var kvRef = Self.ropeTrailing(
            Self.fp16Round(Self.rmsnorm(
                Self.fp16Round(V4Quantization.gemvFP8(matrix: wkv, x: x)), gamma: kvGamma)),
            rows: 1, width: Self.headDim, position: Float(Self.position))
        for block in 0..<(448 / 64) {
            let base = block * 64
            let amax = (0..<64).reduce(Float.zero) {
                max($0, abs(kvRef[base + $1]))
            }
            let scale = V4FP8.blockScale(amax: amax)
            for i in 0..<64 {
                let quantized = V4FP8.e4m3Encode(kvRef[base + i] / scale)
                kvRef[base + i] = Float(Float16(V4FP8.e4m3Decode(quantized) * scale))
            }
        }
        let ring = slot.buffer.contents().advanced(by: slot.offset)
        let kvGot = (0..<Self.headDim).map {
            Float(ring.load(fromByteOffset: $0 * 2, as: Float16.self))
        }
        let relKV = RelError.compute(actual: kvGot, reference: kvRef)
        #expect(relKV < Tolerance.fp16ChainedReduction, "window kv rel=\(relKV)")
        #expect(kvStore.windowPhysicalSlot(position: Self.position) == Self.position % 128)

        // Compressor projections (fp32 GEMVs, fp16 input read exactly).
        for (buf, wgt, name) in [(compKVOut, wkvC, "wkv"), (compGateOut, wgateC, "wgate")] as [(MTLBuffer, [Float], String)] {
            let ref = (0..<1024).map { r -> Float in
                var acc: Float = 0
                for i in 0..<Self.dim { acc += wgt[r * Self.dim + i] * x[i] }
                return acc
            }
            let got = (0..<1024).map { buf.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
            let rel = RelError.compute(actual: got, reference: ref)
            #expect(rel < Tolerance.fp16Reduction, "compressor \(name) rel=\(rel)")
        }

        // Indexer q from the shared qr, trailing RoPE, normalized Hadamard,
        // then block-32 e2m1/e8m0 quantize-dequantize per the QAT reference.
        let idxRef = Self.hadamardFP4QAT(Self.fp16Round(Self.ropeTrailing(
            Self.fp16Round(V4Quantization.gemvFP8(matrix: widx, x: qrRef)),
            rows: Self.heads, width: Self.indexDim, position: Float(Self.position))),
            rows: Self.heads, width: Self.indexDim)
        let idxGot = Fp16Buffer.read(idxOut, count: Self.heads * Self.indexDim)
        let relIdx = RelError.compute(actual: idxGot, reference: idxRef)
        let diffs = zip(idxGot, idxRef).enumerated().filter { abs($0.element.0 - $0.element.1) > 0 }
        // WHT reassociation can move an exact FP4 midpoint to either adjacent
        // ties-to-even bucket. The transform must otherwise be bit-exact and
        // midpoint differences must remain one legal FP4 step.
        #expect(diffs.count <= 4, "indexer q mismatches=\(diffs.count)")
        #expect(diffs.allSatisfy { abs($0.element.0 - $0.element.1) <= 128 })
        #expect(relIdx <= 0.0625, "indexer q rel=\(relIdx)")
    }

    /// The weight-free per-head renorm must actually run: a q whose heads
    /// have wildly different norms collapses to unit RMS per head.
    @Test func perHeadRenorm_producesUnitRMSHeads() throws {
        let ctx = try MetalContext()
        let epi = try V4QKVEpilogue(device: ctx.device)
        // Head h filled with the constant (h+1) * 0.1: renorm maps every
        // element to c * rsqrt(c^2 + eps) ~= 1.
        var q16 = [Float16](repeating: 0, count: Self.heads * Self.headDim)
        for h in 0..<Self.heads {
            for i in 0..<Self.headDim { q16[h * Self.headDim + i] = Float16(h + 1) * 0.1 }
        }
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q16) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        epi.encodePerHeadRenorm(commandBuffer: cb, buf: qBuf,
                                heads: Self.heads, headDim: Self.headDim, eps: 1e-6)
        cb.commit(); cb.waitUntilCompleted()
        let got = Fp16Buffer.read(qBuf, count: Self.heads * Self.headDim)
        for (i, v) in got.enumerated() {
            #expect(abs(v - 1) < 1e-3, "element \(i) = \(v), expected ~1")
        }
    }
}
