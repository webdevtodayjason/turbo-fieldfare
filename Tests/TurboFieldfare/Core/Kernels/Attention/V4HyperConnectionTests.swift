import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-03 wave-2: mHC layer-boundary kernels — dynamic parameter generation
/// (24-wide projection of the RMS-normalized flattened 4x state), sigmoid
/// clamps, the eps-biased Sinkhorn (1 row-softmax + 20 column norms + 19 row
/// norms), the pre gather, and the column-gather post merge. CPU reference is
/// a straight fp32 transcription of recon §4; the kernels are fp32 end to
/// end, so these gates are near-exact.
@Suite struct V4HyperConnectionTests {

    private static let hc = 4
    private static let dim = 4096
    private static let normEps: Float = 1e-6
    private static let hcEps: Float = 1e-6

    // MARK: - CPU reference (exact recon §4 ordering)

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

    /// Returns (pre[4], post[4], comb[16] row-major).
    private static func refParams(x: [Float], fn: [Float], base: [Float],
                                  scale: [Float]) -> (pre: [Float], post: [Float], comb: [Float]) {
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
        return (pre, post, comb)
    }

    private static func refHeadPre(x: [Float], fn: [Float], base: [Float],
                                   scale: Float) -> [Float] {
        let mixes = refMixes(x: x, fn: fn, mixCount: 4)
        return (0..<hc).map { sigmoid(mixes[$0] * scale + base[$0]) + hcEps }
    }

    // MARK: - Fixture + driver

    private struct Fixture {
        var x: [Float]        // [4*dim]
        var fn: [Float]       // [24, 4*dim]
        var base: [Float]     // [24]
        var scale: [Float]    // [3]
    }

    private static func makeFixture(seed: UInt64) -> Fixture {
        var rng = SeedTree(seed).key("mhc")
        let n = hc * dim
        // Unit-ish state so rsqrt is O(1); fn scaled so mixes are O(1).
        let x = (0..<n).map { _ in rng.uniform(-1, 1) }
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

    private static func readParams(_ hcObj: V4HyperConnections, count: Int) -> [Float] {
        let ptr = hcObj.paramsBuffer.contents()
        return (0..<count).map { ptr.load(fromByteOffset: $0 * 4, as: Float.self) }
    }

    // MARK: - Params kernel

    @Test func params_matchCPUReference() throws {
        let f = Self.makeFixture(seed: 0xB01)
        let ctx = try MetalContext()
        let hcObj = try V4HyperConnections(device: ctx.device)
        guard let xb = Self.f32buf(ctx.device, f.x),
              let fb = Self.f32buf(ctx.device, f.fn),
              let bb = Self.f32buf(ctx.device, f.base),
              let sb = Self.f32buf(ctx.device, f.scale) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        hcObj.encodeParams(commandBuffer: cb, x: xb, hcFn: fb, hcBase: bb,
                           hcScale: sb, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()

        let got = Self.readParams(hcObj, count: 24)
        let ref = Self.refParams(x: f.x, fn: f.fn, base: f.base, scale: f.scale)
        let expected = ref.pre + ref.post + ref.comb
        let rel = RelError.compute(actual: got, reference: expected)
        #expect(rel < 1e-4, "params rel=\(rel)")
    }

    /// Sigmoid clamps and Sinkhorn structure: pre in (eps, 1+eps), post in
    /// (0, 2), comb ~doubly stochastic after 20 column + 19 row norms.
    @Test func params_structuralInvariants() throws {
        let f = Self.makeFixture(seed: 0xB02)
        let ctx = try MetalContext()
        let hcObj = try V4HyperConnections(device: ctx.device)
        guard let xb = Self.f32buf(ctx.device, f.x),
              let fb = Self.f32buf(ctx.device, f.fn),
              let bb = Self.f32buf(ctx.device, f.base),
              let sb = Self.f32buf(ctx.device, f.scale) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        hcObj.encodeParams(commandBuffer: cb, x: xb, hcFn: fb, hcBase: bb,
                           hcScale: sb, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()
        let p = Self.readParams(hcObj, count: 24)
        for j in 0..<4 {
            #expect(p[j] > Self.hcEps && p[j] < 1 + Self.hcEps, "pre[\(j)]=\(p[j])")
            #expect(p[4 + j] > 0 && p[4 + j] < 2, "post[\(j)]=\(p[4 + j])")
        }
        for k in 0..<4 {
            let colSum = (0..<4).reduce(Float(0)) { $0 + p[8 + $1 * 4 + k] }
            #expect(colSum > 0.999 && colSum <= 1.0, "col \(k) sum=\(colSum)")
            let rowSum = (0..<4).reduce(Float(0)) { $0 + p[8 + k * 4 + $1] }
            #expect(rowSum > 0.99 && rowSum < 1.01, "row \(k) sum=\(rowSum)")
        }
    }

    /// Iteration-count sensitivity: with a large hc_eps the eps-biased
    /// normalizations move values measurably per iteration, so 18 vs 19
    /// row-norm passes are clearly separable and the kernel must match 19
    /// exactly.
    @Test func params_iterationCountIsExactly19() throws {
        let f = Self.makeFixture(seed: 0xB03)
        // With eps >> comb entries the biased normalization never converges:
        // each row/col pass shrinks values by a large, iteration-dependent
        // factor, so 18 vs 19 passes are separated by orders of magnitude.
        let bigEps: Float = 10
        let ctx = try MetalContext()
        let hcObj = try V4HyperConnections(device: ctx.device)
        guard let xb = Self.f32buf(ctx.device, f.x),
              let fb = Self.f32buf(ctx.device, f.fn),
              let bb = Self.f32buf(ctx.device, f.base),
              let sb = Self.f32buf(ctx.device, f.scale) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        hcObj.encodeParams(commandBuffer: cb, x: xb, hcFn: fb, hcBase: bb,
                           hcScale: sb, dim: Self.dim, hcEps: bigEps)
        cb.commit(); cb.waitUntilCompleted()
        let got = Array(Self.readParams(hcObj, count: 24)[8..<24])

        func combRef(rowNorms: Int) -> [Float] {
            let mixes = Self.refMixes(x: f.x, fn: f.fn, mixCount: 24)
            var comb = [Float](repeating: 0, count: 16)
            for j in 0..<4 { for k in 0..<4 {
                comb[j * 4 + k] = mixes[8 + j * 4 + k] * f.scale[2] + f.base[8 + j * 4 + k]
            } }
            for j in 0..<4 {
                let mx = (0..<4).map { comb[j * 4 + $0] }.max()!
                var sum: Float = 0
                for k in 0..<4 { comb[j * 4 + k] = exp(comb[j * 4 + k] - mx); sum += comb[j * 4 + k] }
                for k in 0..<4 { comb[j * 4 + k] = comb[j * 4 + k] / sum + bigEps }
            }
            func colNorm() {
                for k in 0..<4 {
                    let sum = (0..<4).reduce(Float(0)) { $0 + comb[$1 * 4 + k] }
                    for j in 0..<4 { comb[j * 4 + k] /= (sum + bigEps) }
                }
            }
            colNorm()
            for _ in 0..<rowNorms {
                for j in 0..<4 {
                    let sum = (0..<4).reduce(Float(0)) { $0 + comb[j * 4 + $1] }
                    for k in 0..<4 { comb[j * 4 + k] /= (sum + bigEps) }
                }
                colNorm()
            }
            return comb
        }
        let rel19 = RelError.compute(actual: got, reference: combRef(rowNorms: 19))
        let rel18 = RelError.compute(actual: got, reference: combRef(rowNorms: 18))
        #expect(rel19 < 1e-5, "kernel vs exact 19-iteration ref: rel=\(rel19)")
        #expect(rel19 < rel18 * 1e-2,
                "kernel not at exact count: rel19=\(rel19) rel18=\(rel18)")
    }

    @Test func headParams_preOnly_matchesCPUReference() throws {
        let f = Self.makeFixture(seed: 0xB04)
        let ctx = try MetalContext()
        let hcObj = try V4HyperConnections(device: ctx.device)
        let headFn = Array(f.fn[0..<(4 * Self.hc * Self.dim)])
        let headBase = Array(f.base[0..<4])
        guard let xb = Self.f32buf(ctx.device, f.x),
              let fb = Self.f32buf(ctx.device, headFn),
              let bb = Self.f32buf(ctx.device, headBase),
              let sb = Self.f32buf(ctx.device, [f.scale[0]]) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        hcObj.encodeHeadParams(commandBuffer: cb, x: xb, hcFn: fb, hcBase: bb,
                               hcScale: sb, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()
        let got = Self.readParams(hcObj, count: 4)
        let ref = Self.refHeadPre(x: f.x, fn: headFn, base: headBase, scale: f.scale[0])
        let rel = RelError.compute(actual: got, reference: ref)
        #expect(rel < 1e-4, "head pre rel=\(rel)")
    }

    // MARK: - Pre gather + post merge

    @Test func preAndPost_matchCPUReference() throws {
        let f = Self.makeFixture(seed: 0xB05)
        let ctx = try MetalContext()
        let hcObj = try V4HyperConnections(device: ctx.device)
        var rng = SeedTree(0xB05).key("mhc-io")
        let sublayer16 = (0..<Self.dim).map { _ in Float16(rng.uniform(-1, 1)) }
        guard let xb = Self.f32buf(ctx.device, f.x),
              let fb = Self.f32buf(ctx.device, f.fn),
              let bb = Self.f32buf(ctx.device, f.base),
              let sb = Self.f32buf(ctx.device, f.scale),
              let sub = Fp16Buffer.make(ctx.device, halves: sublayer16),
              let yBuf = ctx.device.makeBuffer(length: Self.dim * 4, options: .storageModeShared),
              let outBuf = ctx.device.makeBuffer(length: 4 * Self.dim * 4, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        hcObj.encodeParams(commandBuffer: cb, x: xb, hcFn: fb, hcBase: bb,
                           hcScale: sb, dim: Self.dim)
        hcObj.encodePre(commandBuffer: cb, x: xb, out: yBuf, dim: Self.dim)
        hcObj.encodePost(commandBuffer: cb, residual: xb, sublayer: sub,
                         out: outBuf, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()

        let ref = Self.refParams(x: f.x, fn: f.fn, base: f.base, scale: f.scale)
        // Pre reference.
        var yRef = [Float](repeating: 0, count: Self.dim)
        for d in 0..<Self.dim {
            for j in 0..<4 { yRef[d] += ref.pre[j] * f.x[j * Self.dim + d] }
        }
        let yGot = (0..<Self.dim).map { yBuf.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
        let relPre = RelError.compute(actual: yGot, reference: yRef)
        #expect(relPre < 1e-5, "pre rel=\(relPre)")
        // Post reference (column gather: out k reads comb[:, k]).
        var outRef = [Float](repeating: 0, count: 4 * Self.dim)
        for d in 0..<Self.dim {
            let s = Float(sublayer16[d])
            for k in 0..<4 {
                var acc = ref.post[k] * s
                for j in 0..<4 { acc += ref.comb[j * 4 + k] * f.x[j * Self.dim + d] }
                outRef[k * Self.dim + d] = acc
            }
        }
        let outGot = (0..<(4 * Self.dim)).map { outBuf.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
        let relPost = RelError.compute(actual: outGot, reference: outRef)
        #expect(relPost < 1e-5, "post rel=\(relPost)")
    }

    /// In-place stream update: out aliasing the residual must be safe.
    @Test func post_inPlace_matchesOutOfPlace() throws {
        let f = Self.makeFixture(seed: 0xB06)
        let ctx = try MetalContext()
        let hcObj = try V4HyperConnections(device: ctx.device)
        var rng = SeedTree(0xB06).key("mhc-inplace")
        let sublayer16 = (0..<Self.dim).map { _ in Float16(rng.uniform(-1, 1)) }
        guard let resInPlace = Self.f32buf(ctx.device, f.x),
              let resOut = Self.f32buf(ctx.device, f.x),
              let xb = Self.f32buf(ctx.device, f.x),
              let fb = Self.f32buf(ctx.device, f.fn),
              let bb = Self.f32buf(ctx.device, f.base),
              let sb = Self.f32buf(ctx.device, f.scale),
              let sub = Fp16Buffer.make(ctx.device, halves: sublayer16),
              let outBuf = ctx.device.makeBuffer(length: 4 * Self.dim * 4, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        hcObj.encodeParams(commandBuffer: cb, x: xb, hcFn: fb, hcBase: bb,
                           hcScale: sb, dim: Self.dim)
        hcObj.encodePost(commandBuffer: cb, residual: resInPlace, sublayer: sub,
                         out: resInPlace, dim: Self.dim)
        hcObj.encodePost(commandBuffer: cb, residual: resOut, sublayer: sub,
                         out: outBuf, dim: Self.dim)
        cb.commit(); cb.waitUntilCompleted()
        let a = (0..<(4 * Self.dim)).map { resInPlace.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
        let b = (0..<(4 * Self.dim)).map { outBuf.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
        #expect(a == b, "in-place and out-of-place post merge diverge")
    }
}
