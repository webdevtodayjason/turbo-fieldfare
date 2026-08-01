import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-03 wave-2: HCA compressor — 128:1 non-overlapped softmax pooling
/// with the ape positional bias, RMSNorm, group-start trailing RoPE, and the
/// split FP8/FP16 quantize. Mirrors the committed CSA compressor test.
@Suite struct V4HCACompressorTests {

    private static let headDim = 512
    private static let rows = 128

    // MARK: - CPU reference

    private static func yarnFreq(_ i: Int,
                                 theta: Float = 160_000,
                                 factor: Float = 16,
                                 origSeqLen: Float = 65_536,
                                 betaFast: Float = 32,
                                 betaSlow: Float = 1) -> Float {
        let dim: Float = 64
        let freq = pow(theta, -2 * Float(i) / dim)
        let logBase = log(theta)
        let lowRot = dim * log(origSeqLen / (betaFast * 2 * .pi)) / (2 * logBase)
        let highRot = dim * log(origSeqLen / (betaSlow * 2 * .pi)) / (2 * logBase)
        let low = max(floor(lowRot), 0)
        var high = min(ceil(highRot), dim - 1)
        if high == low { high = low + 0.001 }
        let ramp = min(max((Float(i) - low) / (high - low), 0), 1)
        let smooth = 1 - ramp
        return (freq / factor) * (1 - smooth) + freq * smooth
    }

    /// HCA pooling reference (recon §3): non-overlapped softmax pooling over
    /// the 128-token group + RMSNorm + group-start partial RoPE. Returns the
    /// 512-dim entry BEFORE FP8 quantization.
    private static func refCompressGroup(kv: [Float], gate: [Float],
                                         ape: [Float], gamma: [Float],
                                         ropePosition: Int) -> [Float] {
        var x = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim {
            var mx = -Float.infinity
            for j in 0..<rows { mx = max(mx, gate[j * headDim + d] + ape[j * headDim + d]) }
            var acc: Float = 0, sum: Float = 0
            for j in 0..<rows {
                let w = exp(gate[j * headDim + d] + ape[j * headDim + d] - mx)
                sum += w
                acc += w * kv[j * headDim + d]
            }
            x[d] = acc / sum
        }
        let ms = x.reduce(0) { $0 + $1 * $1 } / Float(headDim)
        let inv = 1 / (ms + 1e-6).squareRoot()
        for d in 0..<headDim { x[d] = x[d] * inv * gamma[d] }
        for i in 0..<32 {
            let angle = Float(ropePosition) * yarnFreq(i)
            let cs = cos(angle), sn = sin(angle)
            let x0 = x[448 + i], x1 = x[448 + 32 + i]
            x[448 + i] = x0 * cs - x1 * sn
            x[448 + 32 + i] = x0 * sn + x1 * cs
        }
        return x
    }

    @Test func hcaCompressGroup_matchesCPUReference() throws {
        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: [128])
        let kvStore = try CompressedKVCacheManager(device: ctx.device,
                                                   config: config, maxContext: 512)
        let compressor = try V4HCACompressor(device: ctx.device)

        var rng = SeedTree(0xD01).key("hca-compressor")
        let kv = (0..<(Self.rows * Self.headDim)).map { _ in rng.uniform(-1, 1) }
        let gate = (0..<(Self.rows * Self.headDim)).map { _ in rng.uniform(-1, 1) }
        let ape = (0..<(Self.rows * Self.headDim)).map { _ in rng.uniform(-0.25, 0.25) }
        let gamma = (0..<Self.headDim).map { _ in rng.uniform(0.5, 1.5) }

        func f32buf(_ values: [Float]) -> MTLBuffer? {
            ctx.device.makeBuffer(bytes: values, length: values.count * 4,
                                  options: .storageModeShared)
        }
        guard let kb = f32buf(kv), let gb = f32buf(gate),
              let ab = f32buf(ape), let gmb = f32buf(gamma) else {
            Issue.record("alloc failed"); return
        }

        let group = 2
        let slot = kvStore.compressedSlot(layer: 0, group: group)
        let ropePos = kvStore.ropePosition(layer: 0, group: group)
        #expect(ropePos == 256)   // group-start position: 128 * group
        let cb = ctx.queue.makeCommandBuffer()!
        compressor.encodeGroup(commandBuffer: cb,
                               kv: kb, gate: gb, ape: ab, gamma: gmb,
                               outValues: slot.values.buffer,
                               valuesOffset: slot.values.offset,
                               outScales: slot.scales.buffer,
                               scalesOffset: slot.scales.offset,
                               outRope: slot.rope.buffer,
                               ropeOffset: slot.rope.offset,
                               ropePosition: UInt32(ropePos))
        cb.commit(); cb.waitUntilCompleted()

        let actual = kvStore.readEntry(layer: 0, group: group)
        let ref = Self.refCompressGroup(kv: kv, gate: gate, ape: ape,
                                        gamma: gamma, ropePosition: ropePos)
        // Rope dims: FP16 storage of the same fp32 math — near-exact.
        for d in 448..<512 {
            let err = abs(actual[d] - ref[d])
            #expect(err < 1e-2, "rope dim \(d) err \(err)")
        }
        // Non-rope dims: CPU reference through the same quantizer; slack for
        // rounding-boundary flips (same gate as the CSA compressor test).
        let q = CompressedKVCacheManager.quantizeEntry(ref, config: config)
        let qref = CompressedKVCacheManager.dequantizeEntry(
            values: q.values, scales: q.scales, rope: q.rope, config: config)
        let rel = RelError.compute(actual: Array(actual[0..<448]),
                                   reference: Array(qref[0..<448]))
        #expect(rel < 8e-2, "compressed entry rel=\(rel)")
    }

    /// Positional bias matters: zeroing ape must change the output, proving
    /// the kernel adds it (and softmax weights genuinely pool).
    @Test func hcaCompressGroup_apeIsApplied() throws {
        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: [128])
        let kvStore = try CompressedKVCacheManager(device: ctx.device,
                                                   config: config, maxContext: 512)
        let compressor = try V4HCACompressor(device: ctx.device)

        var rng = SeedTree(0xD02).key("hca-ape")
        let kv = (0..<(Self.rows * Self.headDim)).map { _ in rng.uniform(-1, 1) }
        let gate = (0..<(Self.rows * Self.headDim)).map { _ in rng.uniform(-1, 1) }
        let ape = (0..<(Self.rows * Self.headDim)).map { _ in rng.uniform(-0.5, 0.5) }
        let apeZero = [Float](repeating: 0, count: Self.rows * Self.headDim)
        let gamma = [Float](repeating: 1, count: Self.headDim)

        func f32buf(_ values: [Float]) -> MTLBuffer? {
            ctx.device.makeBuffer(bytes: values, length: values.count * 4,
                                  options: .storageModeShared)
        }
        guard let kb = f32buf(kv), let gb = f32buf(gate),
              let ab = f32buf(ape), let az = f32buf(apeZero),
              let gmb = f32buf(gamma) else {
            Issue.record("alloc failed"); return
        }
        let s0 = kvStore.compressedSlot(layer: 0, group: 0)
        let s1 = kvStore.compressedSlot(layer: 0, group: 1)
        let cb = ctx.queue.makeCommandBuffer()!
        compressor.encodeGroup(commandBuffer: cb, kv: kb, gate: gb, ape: ab, gamma: gmb,
                               outValues: s0.values.buffer, valuesOffset: s0.values.offset,
                               outScales: s0.scales.buffer, scalesOffset: s0.scales.offset,
                               outRope: s0.rope.buffer, ropeOffset: s0.rope.offset,
                               ropePosition: 0)
        compressor.encodeGroup(commandBuffer: cb, kv: kb, gate: gb, ape: az, gamma: gmb,
                               outValues: s1.values.buffer, valuesOffset: s1.values.offset,
                               outScales: s1.scales.buffer, scalesOffset: s1.scales.offset,
                               outRope: s1.rope.buffer, ropeOffset: s1.rope.offset,
                               ropePosition: 0)
        cb.commit(); cb.waitUntilCompleted()
        let withApe = kvStore.readEntry(layer: 0, group: 0)
        let without = kvStore.readEntry(layer: 0, group: 1)
        let rel = RelError.compute(actual: withApe, reference: without)
        #expect(rel > 1e-3, "ape has no effect: rel=\(rel)")
    }
}
