import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-06c A1: batched prefill boundary kernels — mHC params (24-mix from
/// the RMS-normalized [rows, 4*dim] state, exact decode ordering: sigmoid
/// pre + eps, 2*sigmoid post, eps-biased Sinkhorn with 1 softmax + 20 column
/// + 19 row norms), the pre gather and column-gather post merge, the
/// fp32-in/fp16-out RMSNorm, and the per-row-position trailing-64 RoPE
/// (INTERLEAVED adjacent pairs). CPU references are straight fp32
/// transcriptions of the committed decode tests; the fp32 paths gate
/// near-exact, the fp16-storage paths gate at fp16 tolerance.
@Suite struct V4PrefillBoundaryTests {

    private static let hc = 4
    private static let dim = 4096
    private static let width = 512
    private static let ropeDim = 64
    private static let normEps: Float = 1e-6
    private static let hcEps: Float = 1e-6

    // MARK: - CPU reference: mHC (exact recon §4 ordering)

    private static func refMixes(x: [Float], fn: [Float], mixCount: Int) -> [Float] {
        let n = x.count
        let ms = x.reduce(0) { $0 + $1 * $1 } / Float(n)
        let rs = 1 / (ms + normEps).squareRoot()
        return (0..<mixCount).map { j in
            var dot: Float = 0
            for i in 0..<n { dot += fn[j * n + i] * x[i] }
            return dot * rs
        }
    }

    private static func sigmoid(_ z: Float) -> Float { 1 / (1 + exp(-z)) }

    /// Returns pre[4] + post[4] + comb[16] row-major (the kernel layout).
    private static func refParams(x: [Float], fn: [Float], base: [Float],
                                  scale: [Float]) -> [Float] {
        let mixes = refMixes(x: x, fn: fn, mixCount: 24)
        let pre = (0..<hc).map { sigmoid(mixes[$0] * scale[0] + base[$0]) + hcEps }
        let post = (0..<hc).map { 2 * sigmoid(mixes[4 + $0] * scale[1] + base[4 + $0]) }
        var comb = [Float](repeating: 0, count: 16)
        for j in 0..<hc {
            for k in 0..<hc {
                comb[j * 4 + k] = mixes[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k]
            }
        }
        // (1) row softmax (max-subtracted, normalized) + hc_eps.
        for j in 0..<hc {
            let mx = (0..<hc).map { comb[j * 4 + $0] }.max()!
            var sum: Float = 0
            for k in 0..<hc { comb[j * 4 + k] = exp(comb[j * 4 + k] - mx); sum += comb[j * 4 + k] }
            for k in 0..<hc { comb[j * 4 + k] = comb[j * 4 + k] / sum + hcEps }
        }
        // (2) first column normalize.
        for k in 0..<hc {
            let sum = (0..<hc).reduce(0) { $0 + comb[$1 * 4 + k] }
            for j in 0..<hc { comb[j * 4 + k] /= (sum + hcEps) }
        }
        // (3) 19x { row normalize, column normalize }.
        for _ in 0..<19 {
            for j in 0..<hc {
                let sum = (0..<hc).reduce(0) { $0 + comb[j * 4 + $1] }
                for k in 0..<hc { comb[j * 4 + k] /= (sum + hcEps) }
            }
            for k in 0..<hc {
                let sum = (0..<hc).reduce(0) { $0 + comb[$1 * 4 + k] }
                for j in 0..<hc { comb[j * 4 + k] /= (sum + hcEps) }
            }
        }
        return pre + post + comb
    }

    // MARK: - CPU reference: RoPE (interleaved adjacent pairs)

    private static func yarnFreq(_ i: Int,
                                 theta: Float,
                                 factor: Float,
                                 origSeqLen: Float,
                                 betaFast: Float,
                                 betaSlow: Float) -> Float {
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

    /// Per-row-position trailing-slice rotation over [rows, width], forward
    /// or conjugate. INTERLEAVED (adjacent-pair) convention.
    private static func refRotate(_ input: [Float],
                                  rows: Int, width: Int, ropeDim: Int,
                                  positions: [Float], inverse: Bool,
                                  config: V4RoPE.Config) -> [Float] {
        var x = input
        let half = ropeDim / 2
        for r in 0..<rows {
            let base = r * width + (width - ropeDim)
            for i in 0..<half {
                let angle = positions[r] * freq(i, config: config)
                let cs = cos(angle)
                let sn = inverse ? -sin(angle) : sin(angle)
                let x0 = x[base + 2 * i], x1 = x[base + 2 * i + 1]
                x[base + 2 * i] = x0 * cs - x1 * sn
                x[base + 2 * i + 1] = x0 * sn + x1 * cs
            }
        }
        return x
    }

    // MARK: - Fixtures + drivers

    private struct Fixture {
        var x: [Float]        // [rows, 4*dim]
        var fn: [Float]       // [24, 4*dim]
        var base: [Float]     // [24]
        var scale: [Float]    // [3]
    }

    private static func makeFixture(seed: UInt64, rows: Int) -> Fixture {
        var rng = SeedTree(seed).key("mhc-pf")
        let n = hc * dim
        let x = (0..<(rows * n)).map { _ in rng.uniform(-1, 1) }
        let s = 1 / Float(n).squareRoot()
        let fn = (0..<(24 * n)).map { _ in rng.uniform(-s, s) }
        let base = (0..<24).map { _ in rng.uniform(-0.5, 0.5) }
        let scale = (0..<3).map { _ in rng.uniform(0.5, 1.5) }
        return Fixture(x: x, fn: fn, base: base, scale: scale)
    }

    private static func f32buf(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        device.makeBuffer(bytes: values, length: values.count * 4,
                          options: .storageModeShared)
    }

    private static func readF32(_ buf: MTLBuffer, count: Int) -> [Float] {
        (0..<count).map { buf.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
    }

    private struct HCSetup {
        var ctx: MetalContext
        var boundary: V4PrefillBoundary
        var xb: MTLBuffer
        var fb: MTLBuffer
        var bb: MTLBuffer
        var sb: MTLBuffer
    }

    private static func makeHCSetup(_ f: Fixture, rows: Int) throws -> HCSetup {
        let ctx = try MetalContext()
        let boundary = try V4PrefillBoundary(device: ctx.device)
        guard let xb = f32buf(ctx.device, f.x),
              let fb = f32buf(ctx.device, f.fn),
              let bb = f32buf(ctx.device, f.base),
              let sb = f32buf(ctx.device, f.scale) else {
            throw MetalError.noDevice
        }
        return HCSetup(ctx: ctx, boundary: boundary, xb: xb, fb: fb, bb: bb, sb: sb)
    }

    // MARK: - Batched mHC params

    @Test func params_matchCPUReference() throws {
        let rows = 8
        let f = Self.makeFixture(seed: 0xC01, rows: rows)
        let s = try Self.makeHCSetup(f, rows: rows)
        let cb = s.ctx.queue.makeCommandBuffer()!
        s.boundary.encodeParams(commandBuffer: cb, x: s.xb, hcFn: s.fb,
                                hcBase: s.bb, hcScale: s.sb,
                                rows: rows, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()

        let got = Self.readF32(s.boundary.paramsBuffer,
                               count: rows * V4PrefillBoundary.paramsFloatsPerRow)
        let n = Self.hc * Self.dim
        var expected: [Float] = []
        for r in 0..<rows {
            let xr = Array(f.x[(r * n)..<((r + 1) * n)])
            expected += Self.refParams(x: xr, fn: f.fn, base: f.base, scale: f.scale)
        }
        let rel = RelError.compute(actual: got, reference: expected)
        #expect(rel < 1e-5, "batched params rel=\(rel)")
    }

    /// Full-chunk row count (128 rows) and per-row Sinkhorn structure.
    @Test func params_fullChunk_structuralInvariants() throws {
        let rows = 128
        let f = Self.makeFixture(seed: 0xC02, rows: rows)
        let s = try Self.makeHCSetup(f, rows: rows)
        let cb = s.ctx.queue.makeCommandBuffer()!
        s.boundary.encodeParams(commandBuffer: cb, x: s.xb, hcFn: s.fb,
                                hcBase: s.bb, hcScale: s.sb,
                                rows: rows, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()
        let p = Self.readF32(s.boundary.paramsBuffer,
                             count: rows * V4PrefillBoundary.paramsFloatsPerRow)
        for r in 0..<rows {
            let pr = Array(p[(r * 24)..<((r + 1) * 24)])
            for j in 0..<4 {
                #expect(pr[j] > Self.hcEps && pr[j] < 1 + Self.hcEps,
                        "row \(r) pre[\(j)]=\(pr[j])")
                #expect(pr[4 + j] > 0 && pr[4 + j] < 2,
                        "row \(r) post[\(j)]=\(pr[4 + j])")
            }
            for k in 0..<4 {
                let colSum = (0..<4).reduce(Float(0)) { $0 + pr[8 + $1 * 4 + k] }
                #expect(colSum > 0.999 && colSum <= 1.0, "row \(r) col \(k) sum=\(colSum)")
                let rowSum = (0..<4).reduce(Float(0)) { $0 + pr[8 + k * 4 + $1] }
                #expect(rowSum > 0.99 && rowSum < 1.01, "row \(r) row \(k) sum=\(rowSum)")
            }
        }
    }

    /// Batched params must match the decode single-row kernel row for row
    /// (both fp32 with identical reduction structure).
    @Test func params_matchDecodeKernelPerRow() throws {
        let rows = 8
        let f = Self.makeFixture(seed: 0xC03, rows: rows)
        let s = try Self.makeHCSetup(f, rows: rows)
        let decode = try V4HyperConnections(device: s.ctx.device)
        let cb = s.ctx.queue.makeCommandBuffer()!
        s.boundary.encodeParams(commandBuffer: cb, x: s.xb, hcFn: s.fb,
                                hcBase: s.bb, hcScale: s.sb,
                                rows: rows, dim: Self.dim)
        let n = Self.hc * Self.dim
        for r in 0..<rows {
            decode.encodeParams(commandBuffer: cb,
                                x: s.xb, xOffset: r * n * 4,
                                hcFn: s.fb, hcBase: s.bb, hcScale: s.sb,
                                dim: Self.dim)
            if r == 0 {
                // Decode wrapper holds one params slot; snapshot per row.
            }
        }
        // The decode wrapper reuses one params buffer, so encode + read per
        // row in separate buffers instead: rerun serially.
        cb.commit(); cb.waitUntilCompleted()
        var decodeAll: [Float] = []
        for r in 0..<rows {
            let cb2 = s.ctx.queue.makeCommandBuffer()!
            decode.encodeParams(commandBuffer: cb2,
                                x: s.xb, xOffset: r * n * 4,
                                hcFn: s.fb, hcBase: s.bb, hcScale: s.sb,
                                dim: Self.dim)
            cb2.commit(); cb2.waitUntilCompleted()
            decodeAll += Self.readF32(decode.paramsBuffer, count: 24)
        }
        let got = Self.readF32(s.boundary.paramsBuffer, count: rows * 24)
        let rel = RelError.compute(actual: got, reference: decodeAll)
        #expect(rel < 1e-6, "batched vs decode params rel=\(rel)")
    }

    // MARK: - Batched pre gather + post merge

    @Test func preAndPost_matchCPUReference() throws {
        let rows = 8
        let f = Self.makeFixture(seed: 0xC04, rows: rows)
        var rng = SeedTree(0xC04).key("mhc-pf-io")
        let sublayer16 = (0..<(rows * Self.dim)).map { _ in Float16(rng.uniform(-1, 1)) }
        let s = try Self.makeHCSetup(f, rows: rows)
        guard let sub = Fp16Buffer.make(s.ctx.device, halves: sublayer16),
              let yBuf = s.ctx.device.makeBuffer(length: rows * Self.dim * 4, options: .storageModeShared),
              let outBuf = s.ctx.device.makeBuffer(length: rows * 4 * Self.dim * 4, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = s.ctx.queue.makeCommandBuffer()!
        s.boundary.encodeParams(commandBuffer: cb, x: s.xb, hcFn: s.fb,
                                hcBase: s.bb, hcScale: s.sb,
                                rows: rows, dim: Self.dim)
        s.boundary.encodePre(commandBuffer: cb, x: s.xb, out: yBuf,
                             rows: rows, dim: Self.dim)
        s.boundary.encodePost(commandBuffer: cb, residual: s.xb, sublayer: sub,
                              out: outBuf, rows: rows, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()

        let n = Self.hc * Self.dim
        var yRef = [Float](repeating: 0, count: rows * Self.dim)
        var outRef = [Float](repeating: 0, count: rows * n)
        for r in 0..<rows {
            let xr = Array(f.x[(r * n)..<((r + 1) * n)])
            let p = Self.refParams(x: xr, fn: f.fn, base: f.base, scale: f.scale)
            let pre = Array(p[0..<4]), post = Array(p[4..<8]), comb = Array(p[8..<24])
            for d in 0..<Self.dim {
                for j in 0..<4 { yRef[r * Self.dim + d] += pre[j] * f.x[r * n + j * Self.dim + d] }
                let sv = Float(sublayer16[r * Self.dim + d])
                for k in 0..<4 {
                    var acc = post[k] * sv
                    for j in 0..<4 { acc += comb[j * 4 + k] * f.x[r * n + j * Self.dim + d] }
                    outRef[r * n + k * Self.dim + d] = acc
                }
            }
        }
        let yGot = Self.readF32(yBuf, count: rows * Self.dim)
        let relPre = RelError.compute(actual: yGot, reference: yRef)
        #expect(relPre < 1e-5, "pre rel=\(relPre)")
        let outGot = Self.readF32(outBuf, count: rows * n)
        let relPost = RelError.compute(actual: outGot, reference: outRef)
        #expect(relPost < 1e-5, "post rel=\(relPost)")
    }

    /// In-place stream update across the batch: out aliasing the residual
    /// must be safe.
    @Test func post_inPlace_matchesOutOfPlace() throws {
        let rows = 8
        let f = Self.makeFixture(seed: 0xC05, rows: rows)
        var rng = SeedTree(0xC05).key("mhc-pf-inplace")
        let sublayer16 = (0..<(rows * Self.dim)).map { _ in Float16(rng.uniform(-1, 1)) }
        let s = try Self.makeHCSetup(f, rows: rows)
        guard let resInPlace = Self.f32buf(s.ctx.device, f.x),
              let resOut = Self.f32buf(s.ctx.device, f.x),
              let sub = Fp16Buffer.make(s.ctx.device, halves: sublayer16),
              let outBuf = s.ctx.device.makeBuffer(length: rows * 4 * Self.dim * 4, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = s.ctx.queue.makeCommandBuffer()!
        s.boundary.encodeParams(commandBuffer: cb, x: s.xb, hcFn: s.fb,
                                hcBase: s.bb, hcScale: s.sb,
                                rows: rows, dim: Self.dim)
        s.boundary.encodePost(commandBuffer: cb, residual: resInPlace, sublayer: sub,
                              out: resInPlace, rows: rows, dim: Self.dim)
        s.boundary.encodePost(commandBuffer: cb, residual: resOut, sublayer: sub,
                              out: outBuf, rows: rows, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()
        let a = Self.readF32(resInPlace, count: rows * 4 * Self.dim)
        let b = Self.readF32(outBuf, count: rows * 4 * Self.dim)
        #expect(a == b, "in-place and out-of-place batched post merge diverge")
    }

    // MARK: - Batched fp32-in/fp16-out RMSNorm

    @Test func rmsnorm_matchesCPUReference() throws {
        let rows = 8
        let n = Self.dim
        var rng = SeedTree(0xC06).key("rms-pf")
        let x = (0..<(rows * n)).map { _ in rng.uniform(-2, 2) }
        let gamma = (0..<n).map { _ in rng.uniform(0.5, 1.5) }
        let ctx = try MetalContext()
        let boundary = try V4PrefillBoundary(device: ctx.device)
        guard let xb = Self.f32buf(ctx.device, x),
              let gb = Self.f32buf(ctx.device, gamma),
              let yb = Fp16Buffer.make(ctx.device, count: rows * n),
              let ybNoGamma = Fp16Buffer.make(ctx.device, count: rows * n) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        boundary.encodeRMSNorm(commandBuffer: cb, x: xb, gamma: gb, out: yb,
                               rows: rows, n: n, useGamma: true)
        boundary.encodeRMSNorm(commandBuffer: cb, x: xb, gamma: gb, out: ybNoGamma,
                               rows: rows, n: n, useGamma: false)
        cb.commit(); cb.waitUntilCompleted()

        var ref = [Float](repeating: 0, count: rows * n)
        var refNoGamma = [Float](repeating: 0, count: rows * n)
        for r in 0..<rows {
            let row = Array(x[(r * n)..<((r + 1) * n)])
            let ms = row.reduce(0) { $0 + $1 * $1 } / Float(n)
            let rs = 1 / (ms + Self.normEps).squareRoot()
            for i in 0..<n {
                ref[r * n + i] = row[i] * rs * gamma[i]
                refNoGamma[r * n + i] = row[i] * rs
            }
        }
        let got = Fp16Buffer.read(yb, count: rows * n)
        let rel = RelError.compute(actual: got, reference: ref)
        #expect(rel < Tolerance.fp16Reduction, "rmsnorm rel=\(rel)")
        let gotNoGamma = Fp16Buffer.read(ybNoGamma, count: rows * n)
        let relNG = RelError.compute(actual: gotNoGamma, reference: refNoGamma)
        #expect(relNG < Tolerance.fp16Reduction, "rmsnorm no-gamma rel=\(relNG)")
    }

    // MARK: - Batched per-row-position RoPE

    private func runRope(x: [Float], rows: Int, width: Int,
                         positions: [Float], inverse: Bool,
                         config: V4RoPE.Config) throws -> [Float] {
        let ctx = try MetalContext()
        let boundary = try V4PrefillBoundary(device: ctx.device)
        guard let buf = Fp16Buffer.make(ctx.device, values: x),
              let posBuf = Self.f32buf(ctx.device, positions) else {
            Issue.record("alloc failed"); return []
        }
        let cb = ctx.queue.makeCommandBuffer()!
        boundary.encodeRoPE(commandBuffer: cb, x: buf, positions: posBuf,
                            rows: rows, width: width,
                            inverse: inverse, config: config)
        cb.commit(); cb.waitUntilCompleted()
        return Fp16Buffer.read(buf, count: x.count)
    }

    @Test func rope_perRowPositions_matchesInterleavedReference() throws {
        let positions: [Float] = [0, 1, 127, 30_000]
        let rows = positions.count
        var rng = SeedTree(0xC07).key("rope-pf")
        let x = (0..<(rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        for config in [V4RoPE.Config.compressedLayer, .passthroughLayer] {
            let actual = try runRope(x: x, rows: rows, width: Self.width,
                                     positions: positions, inverse: false,
                                     config: config)
            // CPU reference consumes the fp16 storage round-trip.
            let stored = x.map { Float(Float16($0)) }
            let ref = Self.refRotate(stored, rows: rows, width: Self.width,
                                     ropeDim: Self.ropeDim, positions: positions,
                                     inverse: false, config: config)
            let rel = RelError.compute(actual: actual, reference: ref)
            #expect(rel < Tolerance.fp16Reduction,
                    "useYarn=\(config.useYarn) rel=\(rel)")
        }
        // The two configs must genuinely differ (theta 10000 vs 160000+YaRN).
        let stored = x.map { Float(Float16($0)) }
        let refCompressed = Self.refRotate(stored, rows: rows, width: Self.width,
                                           ropeDim: Self.ropeDim, positions: positions,
                                           inverse: false, config: .compressedLayer)
        let refPassthrough = Self.refRotate(stored, rows: rows, width: Self.width,
                                            ropeDim: Self.ropeDim, positions: positions,
                                            inverse: false, config: .passthroughLayer)
        let relConfigs = RelError.compute(actual: refPassthrough, reference: refCompressed)
        #expect(relConfigs > 1e-2, "configs indistinguishable: \(relConfigs)")
    }

    /// Non-rope dims must pass through untouched (bit-exact fp16 copy) at
    /// every row position.
    @Test func rope_leavesNonRopeDimsUntouched() throws {
        let positions: [Float] = [0, 1, 127, 30_000]
        let rows = positions.count
        var rng = SeedTree(0xC08).key("rope-pf-passthru")
        let x = (0..<(rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        let actual = try runRope(x: x, rows: rows, width: Self.width,
                                 positions: positions, inverse: false,
                                 config: .compressedLayer)
        let stored = x.map { Float(Float16($0)) }
        for r in 0..<rows {
            for d in 0..<(Self.width - Self.ropeDim) {
                #expect(actual[r * Self.width + d] == stored[r * Self.width + d])
            }
        }
    }

    /// Per-row positions must actually be applied per row: rotating every
    /// row at one shared position must differ from the per-row result.
    @Test func rope_positionsArePerRow() throws {
        let positions: [Float] = [0, 1, 127, 30_000]
        let rows = positions.count
        var rng = SeedTree(0xC09).key("rope-pf-perrow")
        let x = (0..<(rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        let actual = try runRope(x: x, rows: rows, width: Self.width,
                                 positions: positions, inverse: false,
                                 config: .compressedLayer)
        let stored = x.map { Float(Float16($0)) }
        let wrongRef = Self.refRotate(stored, rows: rows, width: Self.width,
                                      ropeDim: Self.ropeDim,
                                      positions: [Float](repeating: positions[0], count: rows),
                                      inverse: false, config: .compressedLayer)
        let relWrong = RelError.compute(actual: actual, reference: wrongRef)
        #expect(relWrong > 1e-2, "per-row positions not applied: \(relWrong)")
    }

    /// Inverse (conjugate) mode at POSITIVE positions: rotate forward then
    /// de-rotate with the same per-row positions must round-trip.
    @Test func rope_inverseVsForward_recovers() throws {
        let positions: [Float] = [0, 1, 127, 30_000]
        let rows = positions.count
        var rng = SeedTree(0xC0A).key("rope-pf-roundtrip")
        let x = (0..<(rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        let ctx = try MetalContext()
        let boundary = try V4PrefillBoundary(device: ctx.device)
        guard let buf = Fp16Buffer.make(ctx.device, values: x),
              let posBuf = Self.f32buf(ctx.device, positions) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        boundary.encodeRoPE(commandBuffer: cb, x: buf, positions: posBuf,
                            rows: rows, width: Self.width,
                            inverse: false, config: .compressedLayer)
        boundary.encodeRoPE(commandBuffer: cb, x: buf, positions: posBuf,
                            rows: rows, width: Self.width,
                            inverse: true, config: .compressedLayer)
        cb.commit(); cb.waitUntilCompleted()
        let actual = Fp16Buffer.read(buf, count: x.count)
        let rel = RelError.compute(actual: actual, reference: x)
        #expect(rel < Tolerance.fp16Reduction, "round-trip rel=\(rel)")
    }

    /// Inverse mode matches the conjugate CPU reference directly (positive
    /// positions, both layer configs).
    @Test func rope_inverse_matchesConjugateReference() throws {
        let positions: [Float] = [0, 1, 127, 30_000]
        let rows = positions.count
        var rng = SeedTree(0xC0B).key("rope-pf-inv")
        let x = (0..<(rows * Self.width)).map { _ in rng.uniform(-1, 1) }
        for config in [V4RoPE.Config.compressedLayer, .passthroughLayer] {
            let actual = try runRope(x: x, rows: rows, width: Self.width,
                                     positions: positions, inverse: true,
                                     config: config)
            let stored = x.map { Float(Float16($0)) }
            let ref = Self.refRotate(stored, rows: rows, width: Self.width,
                                     ropeDim: Self.ropeDim, positions: positions,
                                     inverse: true, config: config)
            let rel = RelError.compute(actual: actual, reference: ref)
            #expect(rel < Tolerance.fp16Reduction,
                    "inverse useYarn=\(config.useYarn) rel=\(rel)")
        }
    }
}
