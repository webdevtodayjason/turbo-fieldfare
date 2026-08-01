import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-03 wave-2: trailing-slice partial RoPE (q / KV / compressed-entry
/// group-start) and the negative-position output de-rotation, against a CPU
/// reference. mHC and RoPE are gated near-exact: fp32 math on both sides,
/// only the fp16 storage round-trip separates them.
@Suite struct V4RoPETests {

    private static let rows = 64
    private static let width = 512
    private static let ropeDim = 64

    // MARK: - CPU reference

    /// DeepSeek YaRN frequency for rope pair `i` (rope dim 64), mirroring
    /// `v4b_yarn_freq` / the reference precompute_freqs_cis.
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

    private static func freq(_ i: Int, config: V4RoPE.Config) -> Float {
        config.useYarn
            ? yarnFreq(i, theta: config.theta, factor: config.yarnFactor,
                       origSeqLen: config.originalSeqLen,
                       betaFast: config.betaFast, betaSlow: config.betaSlow)
            : pow(config.theta, -2 * Float(i) / 64)
    }

    /// Trailing-slice rotation over [rows, width], forward or conjugate.
    private static func refRotate(_ input: [Float],
                                  rows: Int, width: Int, ropeDim: Int,
                                  position: Float, inverse: Bool,
                                  config: V4RoPE.Config) -> [Float] {
        var x = input
        let half = ropeDim / 2
        for r in 0..<rows {
            let base = r * width + (width - ropeDim)
            for i in 0..<half {
                let angle = position * freq(i, config: config)
                let cs = cos(angle)
                let sn = inverse ? -sin(angle) : sin(angle)
                let x0 = x[base + i], x1 = x[base + half + i]
                x[base + i] = x0 * cs - x1 * sn
                x[base + half + i] = x0 * sn + x1 * cs
            }
        }
        return x
    }

    private func runKernel(x: [Float], rows: Int, width: Int,
                           position: Float, inverse: Bool,
                           config: V4RoPE.Config) throws -> [Float] {
        let ctx = try MetalContext()
        let rope = try V4RoPE(device: ctx.device)
        guard let buf = Fp16Buffer.make(ctx.device, values: x) else {
            Issue.record("alloc failed"); return []
        }
        let cb = ctx.queue.makeCommandBuffer()!
        rope.encode(commandBuffer: cb, x: buf,
                    rows: rows, width: width,
                    position: position, inverse: inverse, config: config)
        cb.commit(); cb.waitUntilCompleted()
        return Fp16Buffer.read(buf, count: x.count)
    }

    // MARK: - Forward rotation

    @Test func forward_compressedConfig_matchesReference() throws {
        var rng = SeedTree(0xA01).key("rope-fwd")
        let x = (0..<(Self.rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        for pos: Float in [0, 1, 137, 40_000, 65_536] {
            let actual = try runKernel(x: x, rows: Self.rows, width: Self.width,
                                       position: pos, inverse: false,
                                       config: .compressedLayer)
            let ref = Self.refRotate(x, rows: Self.rows, width: Self.width,
                                     ropeDim: Self.ropeDim, position: pos,
                                     inverse: false, config: .compressedLayer)
            let rel = RelError.compute(actual: actual, reference: ref)
            #expect(rel < Tolerance.fp16Reduction, "pos \(pos) rel=\(rel)")
        }
    }

    @Test func forward_passthroughConfig_matchesReference() throws {
        var rng = SeedTree(0xA02).key("rope-fwd-pt")
        let x = (0..<(Self.rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        let actual = try runKernel(x: x, rows: Self.rows, width: Self.width,
                                   position: 999, inverse: false,
                                   config: .passthroughLayer)
        let ref = Self.refRotate(x, rows: Self.rows, width: Self.width,
                                 ropeDim: Self.ropeDim, position: 999,
                                 inverse: false, config: .passthroughLayer)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")
        // The two configs must genuinely differ (theta 10000 vs 160000+YaRN).
        let wrongConfig = Self.refRotate(x, rows: Self.rows, width: Self.width,
                                         ropeDim: Self.ropeDim, position: 999,
                                         inverse: false, config: .compressedLayer)
        let relWrong = RelError.compute(actual: actual, reference: wrongConfig)
        #expect(relWrong > 1e-2, "configs indistinguishable: \(relWrong)")
    }

    /// Non-rope dims must pass through untouched (bit-exact fp16 copy).
    @Test func forward_leavesNonRopeDimsUntouched() throws {
        var rng = SeedTree(0xA03).key("rope-passthru")
        let x = (0..<(Self.rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        let actual = try runKernel(x: x, rows: Self.rows, width: Self.width,
                                   position: 512, inverse: false,
                                   config: .compressedLayer)
        let stored = x.map { Float(Float16($0)) }   // fp16 storage round-trip
        for r in 0..<Self.rows {
            for d in 0..<(Self.width - Self.ropeDim) {
                #expect(actual[r * Self.width + d] == stored[r * Self.width + d])
            }
        }
    }

    // MARK: - De-rotation (output inverse)

    @Test func inverse_matchesConjugateReference() throws {
        var rng = SeedTree(0xA04).key("rope-inv")
        let x = (0..<(Self.rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        // Negative and fractional positions are legal on this path.
        for pos: Float in [137, -0.5, -42.25] {
            let actual = try runKernel(x: x, rows: Self.rows, width: Self.width,
                                       position: pos, inverse: true,
                                       config: .compressedLayer)
            let ref = Self.refRotate(x, rows: Self.rows, width: Self.width,
                                     ropeDim: Self.ropeDim, position: pos,
                                     inverse: true, config: .compressedLayer)
            let rel = RelError.compute(actual: actual, reference: ref)
            #expect(rel < Tolerance.fp16Reduction, "pos \(pos) rel=\(rel)")
        }
    }

    /// Rotate then de-rotate at the same position must round-trip.
    @Test func rotateThenDerotate_isIdentity() throws {
        var rng = SeedTree(0xA05).key("rope-roundtrip")
        let x = (0..<(Self.rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        let ctx = try MetalContext()
        let rope = try V4RoPE(device: ctx.device)
        guard let buf = Fp16Buffer.make(ctx.device, values: x) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        rope.encode(commandBuffer: cb, x: buf, rows: Self.rows, width: Self.width,
                    position: 137, inverse: false, config: .compressedLayer)
        rope.encode(commandBuffer: cb, x: buf, rows: Self.rows, width: Self.width,
                    position: 137, inverse: true, config: .compressedLayer)
        cb.commit(); cb.waitUntilCompleted()
        let actual = Fp16Buffer.read(buf, count: x.count)
        let rel = RelError.compute(actual: actual, reference: x)
        #expect(rel < Tolerance.fp16Reduction, "round-trip rel=\(rel)")
    }

    /// Two-token delta test (work-order risk table): tokens at positions p
    /// and p+1 de-rotated at p must differ by exactly one relative rotation
    /// step. Guards sign/order errors in the negative-position path.
    @Test func twoTokenDelta_relativeRotationIsExact() throws {
        var rng = SeedTree(0xA06).key("rope-delta")
        let x = (0..<(Self.rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        let ctx = try MetalContext()
        let rope = try V4RoPE(device: ctx.device)
        guard let bufA = Fp16Buffer.make(ctx.device, values: x),
              let bufB = Fp16Buffer.make(ctx.device, values: x) else {
            Issue.record("alloc failed"); return
        }
        let p: Float = 1000
        let cb = ctx.queue.makeCommandBuffer()!
        // Token at p and token at p+1, both de-rotated at the query position p.
        rope.encode(commandBuffer: cb, x: bufA, rows: Self.rows, width: Self.width,
                    position: p, inverse: false, config: .compressedLayer)
        rope.encode(commandBuffer: cb, x: bufA, rows: Self.rows, width: Self.width,
                    position: p, inverse: true, config: .compressedLayer)
        rope.encode(commandBuffer: cb, x: bufB, rows: Self.rows, width: Self.width,
                    position: p + 1, inverse: false, config: .compressedLayer)
        rope.encode(commandBuffer: cb, x: bufB, rows: Self.rows, width: Self.width,
                    position: p, inverse: true, config: .compressedLayer)
        cb.commit(); cb.waitUntilCompleted()
        let a = Fp16Buffer.read(bufA, count: x.count)
        let b = Fp16Buffer.read(bufB, count: x.count)
        // a ~= x; b ~= R(+1) x.
        let relA = RelError.compute(actual: a, reference: x)
        #expect(relA < Tolerance.fp16Reduction, "identity leg rel=\(relA)")
        let refB = Self.refRotate(x, rows: Self.rows, width: Self.width,
                                  ropeDim: Self.ropeDim, position: 1,
                                  inverse: false, config: .compressedLayer)
        let relB = RelError.compute(actual: b, reference: refB)
        #expect(relB < Tolerance.fp16Reduction, "delta leg rel=\(relB)")
    }

    /// Indexer-width coverage: width 128 with the trailing 64 rope dims
    /// (the indexer q path shape).
    @Test func forward_indexerWidth_matchesReference() throws {
        var rng = SeedTree(0xA07).key("rope-idx")
        let rows = 64, width = 128
        let x = (0..<(rows * width)).map { _ in rng.uniform(-1, 1) }
        let actual = try runKernel(x: x, rows: rows, width: width,
                                   position: 77, inverse: false,
                                   config: .compressedLayer)
        let ref = Self.refRotate(x, rows: rows, width: width,
                                 ropeDim: Self.ropeDim, position: 77,
                                 inverse: false, config: .compressedLayer)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16Reduction, "rel=\(rel)")
    }
}
