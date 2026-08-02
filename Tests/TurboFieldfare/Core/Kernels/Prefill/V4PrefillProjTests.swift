import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-06c A2: batched prefill projection/attention kernels. Every check is
/// against an fp32 CPU reference on synthetic payloads:
///   - FP8 block GEMM at the four projection shapes (plus the o-proj up
///     shape), fp32 and fp16 output modes.
///   - BF16 GEMM at the compressor/router shapes.
///   - Window ring write slot addressing (including wrap).
///   - Causal window MQA: exact attended set (ring prefix + rows[0...i]),
///     future-row leak check, huge-sink -> ~0 behavior.
///   - Grouped o-projection: group-slice isolation (NOT a flat GEMM) and the
///     full down->up chain.
@Suite struct V4PrefillProjTests {

    // MARK: - Helpers

    static func bf16Encode(_ v: Float) -> UInt16 {
        let b = v.bitPattern
        let rounded = (b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    static func bf16Decode(_ e: UInt16) -> Float {
        Float(bitPattern: UInt32(e) << 16)
    }

    static func readF32(_ buf: MTLBuffer, count: Int) -> [Float] {
        let base = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    static func makeF32(_ device: MTLDevice, values: [Float]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: values.count * MemoryLayout<Float>.size,
                              options: .storageModeShared)
        }
    }

    static func makeF32(_ device: MTLDevice, count: Int) -> MTLBuffer? {
        device.makeBuffer(length: count * MemoryLayout<Float>.size,
                          options: .storageModeShared)
    }

    /// Random FP8 block matrix with tame power-of-two scales (NaN code 0x7F
    /// masked out, matching checkpoint reality).
    static func randomFP8Matrix(m: Int, n: Int, rng: inout SplitMix64,
                                scaleLo: Int = -4, scaleHi: Int = 1)
        -> V4Quantization.FP8BlockMatrix {
        let codes = (0..<(m * n)).map { _ -> UInt8 in
            var b = UInt8(rng.uniform(0, 256)) & 0x7F
            if b == 0x7F { b = 0x7E }
            if rng.uniform(0, 1) < 0.5 { b |= 0x80 }
            return b
        }
        let scales = (0..<((m / 128) * (n / 128))).map { _ -> UInt8 in
            UInt8(clamping: 127 + Int(rng.uniform(Float(scaleLo), Float(scaleHi))))
        }
        return V4Quantization.FP8BlockMatrix(m: m, n: n, codes: codes, scales: scales)
    }

    /// CPU fp32 reference: y[r] = dequant(W) x x[r] for each activation row.
    static func refGemmFP8(matrix: V4Quantization.FP8BlockMatrix,
                           x: [[Float]]) -> [[Float]] {
        let w = V4Quantization.dequantizeFP8BlockMatrix(matrix)
        return x.map { row in
            var out = [Float](repeating: 0, count: matrix.m)
            for r in 0..<matrix.m {
                var acc: Float = 0
                let base = r * matrix.n
                for i in 0..<matrix.n { acc += w[base + i] * row[i] }
                out[r] = acc
            }
            return out
        }
    }

    static func refGemmDense(w: [Float], m: Int, n: Int, x: [[Float]]) -> [[Float]] {
        x.map { row in
            var out = [Float](repeating: 0, count: m)
            for r in 0..<m {
                var acc: Float = 0
                let base = r * n
                for i in 0..<n { acc += w[base + i] * row[i] }
                out[r] = acc
            }
            return out
        }
    }

    /// Run one FP8 GEMM shape end to end and compare against the CPU
    /// reference. Returns the relative error.
    static func runFP8Shape(proj: V4PrefillProj, device: MTLDevice, queue: MTLCommandQueue,
                            m: Int, n: Int, rows: Int, seed: UInt64, label: String,
                            outFP16: Bool) throws -> Float {
        var rng = SeedTree(seed).key(label)
        let mat = randomFP8Matrix(m: m, n: n, rng: &rng)
        let xHalves: [[Float16]] = (0..<rows).map { _ in
            (0..<n).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        }
        guard let wBuf = device.makeBuffer(bytes: mat.codes,
                                           length: mat.codes.count,
                                           options: .storageModeShared),
              let sBuf = device.makeBuffer(bytes: mat.scales,
                                           length: mat.scales.count,
                                           options: .storageModeShared),
              let xBuf = Fp16Buffer.make(device, halves: xHalves.flatMap { $0 }),
              let outBuf = outFP16
                ? Fp16Buffer.make(device, count: rows * m)
                : makeF32(device, count: rows * m) else {
            Issue.record("alloc failed"); return .infinity
        }
        let cb = queue.makeCommandBuffer()!
        proj.encodeFP8GEMM(commandBuffer: cb,
                           weights: wBuf, scales: sBuf, x: xBuf, out: outBuf,
                           rows: rows, m: m, n: n, outFP16: outFP16)
        cb.commit(); cb.waitUntilCompleted()

        let xF = xHalves.map { $0.map { Float($0) } }
        let ref = refGemmFP8(matrix: mat, x: xF).flatMap { $0 }
        let actual = outFP16 ? Fp16Buffer.read(outBuf, count: rows * m)
                             : readF32(outBuf, count: rows * m)
        return RelError.compute(actual: actual, reference: ref)
    }

    // MARK: - 1. FP8 block GEMM shapes

    @Test func fp8GEMM_wqA_matchesCPU() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rel = try Self.runFP8Shape(proj: proj, device: ctx.device, queue: ctx.queue,
                                       m: 1024, n: 4096, rows: 4,
                                       seed: 0xA01, label: "wq_a", outFP16: false)
        #expect(rel < 1e-3, "wq_a fp32 rel=\(rel)")
    }

    @Test func fp8GEMM_wqB_matchesCPU() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rel = try Self.runFP8Shape(proj: proj, device: ctx.device, queue: ctx.queue,
                                       m: 32768, n: 1024, rows: 2,
                                       seed: 0xA02, label: "wq_b", outFP16: false)
        #expect(rel < 1e-3, "wq_b fp32 rel=\(rel)")
    }

    @Test func fp8GEMM_wkv_matchesCPU() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rel = try Self.runFP8Shape(proj: proj, device: ctx.device, queue: ctx.queue,
                                       m: 512, n: 4096, rows: 4,
                                       seed: 0xA03, label: "wkv", outFP16: false)
        #expect(rel < 1e-3, "wkv fp32 rel=\(rel)")
    }

    @Test func fp8GEMM_indexerWqB_matchesCPU() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rel = try Self.runFP8Shape(proj: proj, device: ctx.device, queue: ctx.queue,
                                       m: 8192, n: 1024, rows: 3,
                                       seed: 0xA04, label: "idx_wq_b", outFP16: false)
        #expect(rel < 1e-3, "indexer wq_b fp32 rel=\(rel)")
    }

    @Test func fp8GEMM_fp16Out_matchesCPU() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rel = try Self.runFP8Shape(proj: proj, device: ctx.device, queue: ctx.queue,
                                       m: 1024, n: 4096, rows: 4,
                                       seed: 0xA05, label: "wq_a_fp16", outFP16: true)
        #expect(rel < Tolerance.fp16ChainedReduction, "wq_a fp16 rel=\(rel)")
    }

    @Test func fp8GEMM_fp16Out_matchesSerialKernelBitExactlyAcrossRows() throws {
        var rng = SeedTree(0xA06).key("fp8-batched-vs-serial")
        let rows = 3, m = 512, n = 4096
        let matrix = Self.randomFP8Matrix(m: m, n: n, rng: &rng)
        let hidden = (0..<(rows * n)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let serial = try DequantFP8BlockGEMV(context: ctx)
        guard let weights = ctx.device.makeBuffer(bytes: matrix.codes, length: matrix.codes.count,
                                                  options: .storageModeShared),
              let scales = ctx.device.makeBuffer(bytes: matrix.scales, length: matrix.scales.count,
                                                 options: .storageModeShared),
              let x = Fp16Buffer.make(ctx.device, halves: hidden),
              let batched = Fp16Buffer.make(ctx.device, count: rows * m),
              let reference = Fp16Buffer.make(ctx.device, count: rows * m) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeFP8GEMM(commandBuffer: cb, weights: weights, scales: scales,
                           x: x, out: batched, rows: rows, m: m, n: n,
                           outFP16: true)
        for row in 0..<rows {
            serial.encode(commandBuffer: cb,
                          weights: weights,
                          scales: scales,
                          x: x, xOffset: row * n * MemoryLayout<Float16>.stride,
                          y: reference, yOffset: row * m * MemoryLayout<Float16>.stride,
                          m: UInt32(m), n: UInt32(n))
        }
        cb.commit(); cb.waitUntilCompleted()
        #expect(Fp16Buffer.read(batched, count: rows * m) ==
                Fp16Buffer.read(reference, count: rows * m),
                "batched FP8 fp16 output must be bit-exact to serial GEMV")
    }

    // MARK: - 2. BF16 GEMM shapes

    private func runBF16Shape(m: Int, seed: UInt64, label: String) throws -> Float {
        var rng = SeedTree(seed).key(label)
        let n = 4096, rows = 4
        let w16 = (0..<(m * n)).map { _ in Self.bf16Encode(rng.uniform(-0.05, 0.05)) }
        let xHalves: [[Float16]] = (0..<rows).map { _ in
            (0..<n).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        }
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        guard let wBuf = ctx.device.makeBuffer(bytes: w16, length: w16.count * 2,
                                               options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xHalves.flatMap { $0 }),
              let outBuf = Self.makeF32(ctx.device, count: rows * m) else {
            Issue.record("alloc failed"); return .infinity
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeBF16GEMM(commandBuffer: cb,
                            weights: wBuf, x: xBuf, out: outBuf,
                            rows: rows, m: m, n: n, outFP16: false)
        cb.commit(); cb.waitUntilCompleted()

        let wF = w16.map { Self.bf16Decode($0) }
        let ref = Self.refGemmDense(w: wF, m: m, n: n,
                                    x: xHalves.map { $0.map { Float($0) } }).flatMap { $0 }
        let actual = Self.readF32(outBuf, count: rows * m)
        return RelError.compute(actual: actual, reference: ref)
    }

    @Test func bf16GEMM_compressorCSA_matchesCPU() throws {
        let rel = try runBF16Shape(m: 1024, seed: 0xB01, label: "csa")
        #expect(rel < 1e-3, "CSA projection rel=\(rel)")
    }

    @Test func bf16GEMM_compressorHCA_matchesCPU() throws {
        let rel = try runBF16Shape(m: 512, seed: 0xB02, label: "hca")
        #expect(rel < 1e-3, "HCA projection rel=\(rel)")
    }

    @Test func bf16GEMM_routerGate_matchesCPU() throws {
        let rel = try runBF16Shape(m: 256, seed: 0xB03, label: "router")
        #expect(rel < 1e-3, "router gate rel=\(rel)")
    }

    // MARK: - 3. Window ring write

    @Test func windowRingWrite_landsAtWindowSlotAddressing() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let window = 128, headDim = 512, rows = 6, start = 125
        var rng = SeedTree(0xC01).key("ring")
        let kv = (0..<(rows * headDim)).map { _ in Float16(rng.uniform(-1, 1)) }
        guard let kvBuf = Fp16Buffer.make(ctx.device, halves: kv),
              let ringBuf = Fp16Buffer.make(ctx.device, count: window * headDim) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeWindowRingWrite(commandBuffer: cb,
                                   kv: kvBuf, ring: ringBuf,
                                   startPosition: start, rows: rows)
        cb.commit(); cb.waitUntilCompleted()

        let ring = Fp16Buffer.read(ringBuf, count: window * headDim)
        for i in 0..<rows {
            let slot = (start + i) % window
            let got = Array(ring[(slot * headDim)..<((slot + 1) * headDim)])
            let want = kv[i * headDim..<(i + 1) * headDim].map { Float($0) }
            #expect(got == want, "row \(i) must land at slot \(slot)")
        }
    }

    // MARK: - 3b. Causal window MQA prefill attention

    private static let heads = 64
    private static let hd = 512
    private static let scale: Float = 1.0 / Float(512).squareRoot()

    /// fp32 CPU reference: row i attends exactly the last 128 logical tokens
    /// from the wrapped prefix ring + chunk[0...i]; sink in the denominator
    /// only (not the max).
    static func refAttention(q: [Float], ring: [Float], chunk: [Float],
                             sinks: [Float], rows: Int, prefix: Int,
                             ringStart: Int = 0) -> [Float] {
        var out = [Float](repeating: 0, count: rows * heads * hd)
        let window = 128
        let prefixVisible = min(prefix, window)
        let prefixOldest = prefix - prefixVisible
        for r in 0..<rows {
            let logicalTotal = prefix + r + 1
            let firstLogical = max(0, logicalTotal - window)
            let total = min(logicalTotal, window)
            for h in 0..<heads {
                let qBase = (r * heads + h) * hd
                var scores = [Float](repeating: 0, count: total)
                var m = -Float.infinity
                for e in 0..<total {
                    let logical = firstLogical + e
                    var kSrc = ring
                    var kBase = 0
                    if logical < prefix {
                        let slot = (ringStart + (logical - prefixOldest)) % window
                        kBase = slot * hd
                    } else {
                        kSrc = chunk
                        kBase = (logical - prefix) * hd
                    }
                    var dot: Float = 0
                    for i in 0..<hd { dot += q[qBase + i] * kSrc[kBase + i] }
                    scores[e] = dot * scale
                    m = max(m, scores[e])
                }
                var denom = exp(sinks[h] - m)
                for e in 0..<total { denom += exp(scores[e] - m) }
                let oBase = (r * heads + h) * hd
                for e in 0..<total {
                    let p = exp(scores[e] - m)
                    let logical = firstLogical + e
                    var kSrc = ring
                    var kBase = 0
                    if logical < prefix {
                        let slot = (ringStart + (logical - prefixOldest)) % window
                        kBase = slot * hd
                    } else {
                        kSrc = chunk
                        kBase = (logical - prefix) * hd
                    }
                    for i in 0..<hd { out[oBase + i] += p * kSrc[kBase + i] }
                }
                for i in 0..<hd { out[oBase + i] /= denom }
            }
        }
        return out
    }

    private struct AttnFixture {
        var q, ring, chunk, sinks: [Float]
        let rows, prefix: Int
        var ringStart: Int = 0
    }

    private func runAttention(_ fx: AttnFixture, proj: V4PrefillProj,
                              ctx: MetalContext) throws -> [Float] {
        guard let qBuf = Fp16Buffer.make(ctx.device, values: fx.q),
              let ringBuf = Fp16Buffer.make(ctx.device, values: fx.ring),
              let chunkBuf = Fp16Buffer.make(ctx.device, values: fx.chunk),
              let sinkBuf = Self.makeF32(ctx.device, values: fx.sinks),
              let outBuf = Fp16Buffer.make(ctx.device, count: fx.rows * Self.heads * Self.hd) else {
            Issue.record("alloc failed"); return []
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeWindowMQAPrefill(commandBuffer: cb,
                                    q: qBuf, windowK: ringBuf,
                                    prefixCount: fx.prefix,
                                    ringStart: fx.ringStart,
                                    chunkKV: chunkBuf, rows: fx.rows,
                                    sinks: sinkBuf, out: outBuf)
        cb.commit(); cb.waitUntilCompleted()
        return Fp16Buffer.read(outBuf, count: fx.rows * Self.heads * Self.hd)
    }

    private func makeFixture(rows: Int, prefix: Int, seed: UInt64,
                             label: String) -> AttnFixture {
        var rng = SeedTree(seed).key(label)
        let q = (0..<(rows * Self.heads * Self.hd)).map { _ in rng.uniform(-1, 1) }
        let ring = (0..<(Self.hd * 128)).map { _ in rng.uniform(-1, 1) }
        let chunk = (0..<(rows * Self.hd)).map { _ in rng.uniform(-1, 1) }
        let sinks = (0..<Self.heads).map { _ in rng.uniform(-2, 2) }
        return AttnFixture(q: q, ring: ring, chunk: chunk, sinks: sinks,
                           rows: rows, prefix: prefix)
    }

    private func populateWrappedPrefixRing(prefixTokens: [Float], prefix: Int,
                                           ringStart: Int) -> [Float] {
        var ring = [Float](repeating: -123, count: Self.hd * 128)
        let visible = min(prefix, 128)
        let oldest = prefix - visible
        for logical in oldest..<prefix {
            let slot = (ringStart + (logical - oldest)) % 128
            ring.replaceSubrange((slot * Self.hd)..<((slot + 1) * Self.hd),
                                 with: prefixTokens[(logical * Self.hd)..<((logical + 1) * Self.hd)])
        }
        return ring
    }

    @Test func windowMQA_causalSet_matchesCPUReference() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        // fp16 round-trip the inputs so the reference sees what the kernel sees.
        var fx = makeFixture(rows: 8, prefix: 5, seed: 0xD01, label: "causal")
        fx.q = fx.q.map { Float(Float16($0)) }
        fx.ring = fx.ring.map { Float(Float16($0)) }
        fx.chunk = fx.chunk.map { Float(Float16($0)) }
        let actual = try runAttention(fx, proj: proj, ctx: ctx)
        let ref = Self.refAttention(q: fx.q, ring: fx.ring, chunk: fx.chunk,
                                    sinks: fx.sinks, rows: fx.rows, prefix: fx.prefix)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "causal attention rel=\(rel)")
    }

    @Test func windowMQA_firstChunkPrefixZero_matchesCPUReference() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        var fx = makeFixture(rows: 6, prefix: 0, seed: 0xD02, label: "first-chunk")
        fx.q = fx.q.map { Float(Float16($0)) }
        fx.chunk = fx.chunk.map { Float(Float16($0)) }
        let actual = try runAttention(fx, proj: proj, ctx: ctx)
        let ref = Self.refAttention(q: fx.q, ring: fx.ring, chunk: fx.chunk,
                                    sinks: fx.sinks, rows: fx.rows, prefix: fx.prefix)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "prefix-0 attention rel=\(rel)")
    }

    @Test func windowMQA_futureRowsDoNotLeak() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rows = 6, leakFrom = 4
        var fx = makeFixture(rows: rows, prefix: 3, seed: 0xD03, label: "leak")
        fx.q = fx.q.map { Float(Float16($0)) }
        fx.ring = fx.ring.map { Float(Float16($0)) }
        fx.chunk = fx.chunk.map { Float(Float16($0)) }
        let baseline = try runAttention(fx, proj: proj, ctx: ctx)

        // Poison future rows with huge KV; rows < leakFrom must be untouched.
        var poisoned = fx
        for i in 0..<Self.hd {
            poisoned.chunk[leakFrom * Self.hd + i] = 300
            poisoned.chunk[(leakFrom + 1) * Self.hd + i] = -300
        }
        let after = try runAttention(poisoned, proj: proj, ctx: ctx)

        let count = leakFrom * Self.heads * Self.hd
        let diff = RelError.maxAbsDiff(Array(baseline[0..<count]), Array(after[0..<count]))
        #expect(diff == 0, "rows < \(leakFrom) changed by future KV: maxAbsDiff=\(diff)")
        // And the poisoned rows themselves must differ (the poison took effect).
        let tail = RelError.maxAbsDiff(Array(baseline[count...]), Array(after[count...]))
        #expect(tail > 1, "poisoned rows should diverge, maxAbsDiff=\(tail)")
    }

    @Test func windowMQA_prefixPlusRowOver128_usesExactLast128_chunk64() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rows = 64, prefix = 96
        var fx = makeFixture(rows: rows, prefix: prefix, seed: 0xD05, label: "chunk64-over128")
        fx.q = fx.q.map { Float(Float16($0)) }
        fx.ring = fx.ring.map { Float(Float16($0)) }
        fx.chunk = fx.chunk.map { Float(Float16($0)) }
        let actual = try runAttention(fx, proj: proj, ctx: ctx)
        let ref = Self.refAttention(q: fx.q, ring: fx.ring, chunk: fx.chunk,
                                    sinks: fx.sinks, rows: fx.rows, prefix: fx.prefix,
                                    ringStart: fx.ringStart)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "chunk64 last-128 rel=\(rel)")
    }

    @Test func windowMQA_wrappedRing_usesRingStart_chunk128() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rows = 128, prefix = 141, ringStart = 73
        var fx = makeFixture(rows: rows, prefix: prefix, seed: 0xD06, label: "chunk128-wrap")
        var rng = SeedTree(0xD06).key("prefix-tokens")
        let prefixTokens = (0..<(prefix * Self.hd)).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        fx.q = fx.q.map { Float(Float16($0)) }
        fx.chunk = fx.chunk.map { Float(Float16($0)) }
        fx.ringStart = ringStart
        fx.ring = populateWrappedPrefixRing(prefixTokens: prefixTokens, prefix: prefix,
                                            ringStart: ringStart)

        let actual = try runAttention(fx, proj: proj, ctx: ctx)
        let ref = Self.refAttention(q: fx.q, ring: fx.ring, chunk: fx.chunk,
                                    sinks: fx.sinks, rows: fx.rows, prefix: fx.prefix,
                                    ringStart: fx.ringStart)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "wrapped chunk128 rel=\(rel)")

        // Row 0 excludes logical prefix token 12 even though it is retained in
        // the ring; poisoning that physical slot must not affect row 0.
        let excludedLogical = prefix - 128
        let excludedSlot = (ringStart + (excludedLogical - (prefix - 128))) % 128
        var poisoned = fx
        poisoned.ring.replaceSubrange((excludedSlot * Self.hd)..<((excludedSlot + 1) * Self.hd),
                                      with: repeatElement(Float(250), count: Self.hd))
        let after = try runAttention(poisoned, proj: proj, ctx: ctx)
        let row0Count = Self.heads * Self.hd
        let diff = RelError.maxAbsDiff(Array(actual[0..<row0Count]), Array(after[0..<row0Count]))
        #expect(diff == 0, "row 0 changed after poisoning excluded oldest ring slot: \(diff)")
    }

    @Test func windowMQA_poisonFutureInvariant_chunk128() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        let rows = 128, leakFrom = 64
        var fx = makeFixture(rows: rows, prefix: 17, seed: 0xD07, label: "chunk128-future")
        fx.q = fx.q.map { Float(Float16($0)) }
        fx.ring = fx.ring.map { Float(Float16($0)) }
        fx.chunk = fx.chunk.map { Float(Float16($0)) }
        let baseline = try runAttention(fx, proj: proj, ctx: ctx)

        var poisoned = fx
        for row in leakFrom..<rows {
            for i in 0..<Self.hd { poisoned.chunk[row * Self.hd + i] = (row % 2 == 0) ? 300 : -300 }
        }
        let after = try runAttention(poisoned, proj: proj, ctx: ctx)

        let count = leakFrom * Self.heads * Self.hd
        let diff = RelError.maxAbsDiff(Array(baseline[0..<count]), Array(after[0..<count]))
        #expect(diff == 0, "rows before poisoned future changed: maxAbsDiff=\(diff)")
    }

    @Test func windowMQA_hugeSink_drivesOutputToZero() throws {
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        var fx = makeFixture(rows: 4, prefix: 4, seed: 0xD04, label: "sink")
        fx.sinks = [Float](repeating: 30, count: Self.heads)   // exp(30) ~ 1e13
        let actual = try runAttention(fx, proj: proj, ctx: ctx)
        let maxAbs = actual.reduce(0) { max($0, abs($1)) }
        #expect(maxAbs < 1e-3, "huge sink should zero the output, maxAbs=\(maxAbs)")
    }

    // MARK: - 4. Grouped o-projection

    private static let groups = 8
    private static let groupDim = 4096
    private static let lora = 1024
    private static let hidden = 4096

    /// CPU reference for the grouped down stage (group slices, not flat).
    static func refOProjDown(attn: [[Float]], woA: [Float]) -> [[Float]] {
        attn.map { row in
            var out = [Float](repeating: 0, count: groups * lora)
            for g in 0..<groups {
                for i in 0..<lora {
                    var acc: Float = 0
                    let wBase = (g * lora + i) * groupDim
                    let xBase = g * groupDim
                    for d in 0..<groupDim { acc += woA[wBase + d] * row[xBase + d] }
                    out[g * lora + i] = acc
                }
            }
            return out
        }
    }

    @Test func oProjDown_inputGroupSliceIsolation() throws {
        // Only group 3's input slice is nonzero: every other group's low-rank
        // rows must be exactly zero (a flat GEMM would fail this).
        var rng = SeedTree(0xE01).key("oproj-slice")
        let rows = 2
        let mat = Self.randomFP8Matrix(m: Self.groups * Self.lora, n: Self.groupDim,
                                       rng: &rng)
        var attn = [[Float]](repeating: [Float](repeating: 0, count: Self.groups * Self.groupDim),
                             count: rows)
        for r in 0..<rows {
            for d in 0..<Self.groupDim {
                attn[r][3 * Self.groupDim + d] = Float(Float16(rng.uniform(-1, 1)))
            }
        }
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        guard let wBuf = ctx.device.makeBuffer(bytes: mat.codes, length: mat.codes.count,
                                               options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: mat.scales, length: mat.scales.count,
                                               options: .storageModeShared),
              let aBuf = Fp16Buffer.make(ctx.device, values: attn.flatMap { $0 }),
              let lowBuf = Fp16Buffer.make(ctx.device, count: rows * Self.groups * Self.lora) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeGroupedOProjDown(commandBuffer: cb,
                                    attn: aBuf, woAWeights: wBuf, woAScales: sBuf,
                                    rows: rows, lowRank: lowBuf)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(lowBuf, count: rows * Self.groups * Self.lora)
        for r in 0..<rows {
            for g in 0..<Self.groups where g != 3 {
                let base = r * Self.groups * Self.lora + g * Self.lora
                let sliceMax = actual[base..<(base + Self.lora)].reduce(0) { max($0, abs($1)) }
                #expect(sliceMax == 0, "row \(r) group \(g) must be zero, max=\(sliceMax)")
            }
        }
        // Group 3 must match the CPU reference for its slice.
        let w = V4Quantization.dequantizeFP8BlockMatrix(mat)
        let ref = Self.refOProjDown(attn: attn, woA: w).flatMap { $0 }
        var g3Actual: [Float] = [], g3Ref: [Float] = []
        for r in 0..<rows {
            let base = r * Self.groups * Self.lora + 3 * Self.lora
            g3Actual.append(contentsOf: actual[base..<(base + Self.lora)])
            g3Ref.append(contentsOf: ref[base..<(base + Self.lora)])
        }
        let rel = RelError.compute(actual: g3Actual, reference: g3Ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "group 3 rel=\(rel)")
    }

    @Test func oProjDown_weightGroupSliceIsolation() throws {
        // Only group 5's weight rows are nonzero: only group 5's low-rank
        // rows may be nonzero, for every query row.
        var rng = SeedTree(0xE02).key("oproj-wslice")
        let rows = 2, n = Self.groupDim
        var codes = [UInt8](repeating: 0, count: Self.groups * Self.lora * n)
        let scales = [UInt8](repeating: 127, count: (Self.groups * Self.lora / 128) * (n / 128))
        for i in 0..<(Self.lora * n) {
            var b = UInt8(rng.uniform(0, 256)) & 0x7F
            if b == 0x7F { b = 0x7E }
            if rng.uniform(0, 1) < 0.5 { b |= 0x80 }
            codes[5 * Self.lora * n + i] = b
        }
        let mat = V4Quantization.FP8BlockMatrix(m: Self.groups * Self.lora, n: n,
                                                codes: codes, scales: scales)
        let attn: [[Float]] = (0..<rows).map { _ in
            (0..<(Self.groups * Self.groupDim)).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        }
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        guard let wBuf = ctx.device.makeBuffer(bytes: mat.codes, length: mat.codes.count,
                                               options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: mat.scales, length: mat.scales.count,
                                               options: .storageModeShared),
              let aBuf = Fp16Buffer.make(ctx.device, values: attn.flatMap { $0 }),
              let lowBuf = Fp16Buffer.make(ctx.device, count: rows * Self.groups * Self.lora) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeGroupedOProjDown(commandBuffer: cb,
                                    attn: aBuf, woAWeights: wBuf, woAScales: sBuf,
                                    rows: rows, lowRank: lowBuf)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(lowBuf, count: rows * Self.groups * Self.lora)
        for r in 0..<rows {
            for g in 0..<Self.groups where g != 5 {
                let base = r * Self.groups * Self.lora + g * Self.lora
                let sliceMax = actual[base..<(base + Self.lora)].reduce(0) { max($0, abs($1)) }
                #expect(sliceMax == 0, "row \(r) group \(g) must be zero, max=\(sliceMax)")
            }
            let base5 = r * Self.groups * Self.lora + 5 * Self.lora
            let g5Max = actual[base5..<(base5 + Self.lora)].reduce(0) { max($0, abs($1)) }
            #expect(g5Max > 0, "row \(r) group 5 should be nonzero")
        }
    }

    @Test func oProj_fullChain_matchesCPUReference() throws {
        var rng = SeedTree(0xE03).key("oproj-full")
        let rows = 2
        let matA = Self.randomFP8Matrix(m: Self.groups * Self.lora, n: Self.groupDim,
                                        rng: &rng, scaleLo: -10, scaleHi: -6)
        let matB = Self.randomFP8Matrix(m: Self.hidden, n: Self.groups * Self.lora,
                                        rng: &rng, scaleLo: -10, scaleHi: -6)
        let attn: [[Float]] = (0..<rows).map { _ in
            (0..<(Self.groups * Self.groupDim)).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        }
        let ctx = try MetalContext()
        let proj = try V4PrefillProj(device: ctx.device)
        guard let wABuf = ctx.device.makeBuffer(bytes: matA.codes, length: matA.codes.count,
                                                options: .storageModeShared),
              let sABuf = ctx.device.makeBuffer(bytes: matA.scales, length: matA.scales.count,
                                                options: .storageModeShared),
              let wBBuf = ctx.device.makeBuffer(bytes: matB.codes, length: matB.codes.count,
                                                options: .storageModeShared),
              let sBBuf = ctx.device.makeBuffer(bytes: matB.scales, length: matB.scales.count,
                                                options: .storageModeShared),
              let aBuf = Fp16Buffer.make(ctx.device, values: attn.flatMap { $0 }),
              let lowBuf = Fp16Buffer.make(ctx.device, count: rows * Self.groups * Self.lora),
              let outBuf = Fp16Buffer.make(ctx.device, count: rows * Self.hidden) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeGroupedOProjDown(commandBuffer: cb,
                                    attn: aBuf, woAWeights: wABuf, woAScales: sABuf,
                                    rows: rows, lowRank: lowBuf)
        proj.encodeOProjUp(commandBuffer: cb,
                           lowRank: lowBuf, woBWeights: wBBuf, woBScales: sBBuf,
                           rows: rows, out: outBuf)
        cb.commit(); cb.waitUntilCompleted()

        // Reference: grouped down -> fp16 round-trip (the GPU low-rank
        // scratch is fp16) -> FP8 up GEMM.
        let wA = V4Quantization.dequantizeFP8BlockMatrix(matA)
        let low = Self.refOProjDown(attn: attn, woA: wA)
            .map { $0.map { Float(Float16($0)) } }
        let ref = Self.refGemmFP8(matrix: matB, x: low).flatMap { $0 }
        let actual = Fp16Buffer.read(outBuf, count: rows * Self.hidden)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "o-proj chain rel=\(rel)")
    }
}
