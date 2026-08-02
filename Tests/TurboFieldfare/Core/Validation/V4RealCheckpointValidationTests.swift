import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Real-checkpoint validation gate (option C, pre-download de-risking).
///
/// Runs the V4 GPU kernels on byte-exact tensors range-fetched from the real
/// DeepSeek V4-Flash shard 7 (layer 5) and compares against independent
/// from-spec numpy goldens produced by `scripts/v4f/make_v4f_golden.py`:
///
///   * FP4 e2m1 GEMV on expert 0 w1/w2/w3 (real I8 containers + ue8m0 scales)
///   * FP8 e4m3 128x128-block GEMV on wq_a / wq_b / shared_experts.w1
///   * EmbedLookupFP8 row gather on wq_a (exact dequant check)
///   * BF16 router GEMV + sqrt-softplus top-6 selection on gate.weight/bias
///   * Fused SwiGLU MoE phase1/phase2 on expert 0 (blob replicated 6 slots)
///
/// Every GEMV golden exists in float64 (primary) and float32 (kernel-parity)
/// accumulation; both errors are reported. The GPU kernels take fp16
/// activations and emit fp16 outputs, so the expected error floor is fp16
/// output rounding (~5e-4 relative); the pass bar is 5e-3, ten times that.
///
/// Self-contained: reads scratch/v4f-recon/{real-tensors,golden} via absolute
/// paths derived from #filePath; prints a clear SKIP line and vacuously
/// passes when the fixtures are absent (fresh checkout without the recon
/// download).
@Suite struct V4RealCheckpointValidationTests {

    // MARK: - Fixture access

    private static let repoRoot: URL = {
        // .../Tests/TurboFieldfare/Core/Validation/V4RealCheckpointValidationTests.swift
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Validation
            .deletingLastPathComponent()  // Core
            .deletingLastPathComponent()  // TurboFieldfare
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }()

    private static var realDir: URL { repoRoot.appendingPathComponent("scratch/v4f-recon/real-tensors") }
    private static var goldenDir: URL { repoRoot.appendingPathComponent("scratch/v4f-recon/golden") }

    private static func fixturesAvailable() -> Bool {
        let required = [
            realDir.appendingPathComponent("layers.5.ffn.experts.0.w1.weight.bin"),
            realDir.appendingPathComponent("layers.5.attn.wq_b.weight.bin"),
            realDir.appendingPathComponent("layers.5.ffn.gate.weight.bin"),
            goldenDir.appendingPathComponent("manifest.json"),
            goldenDir.appendingPathComponent("w1.gemv.f64.bin"),
            goldenDir.appendingPathComponent("moe.out.f64.bin"),
        ]
        return required.allSatisfy { FileManager.default.fileExists(atPath: $0.path) }
    }

    private static func requireFixtures() -> Bool {
        if fixturesAvailable() { return true }
        print("V4F-REAL SKIP: scratch/v4f-recon/{real-tensors,golden} not present; "
              + "stage the recon tensors and run scripts/v4f/make_v4f_golden.py first")
        return false
    }

    private static func loadBytes(_ url: URL) -> [UInt8] {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else {
            Issue.record("failed to read \(url.path)")
            return []
        }
        return [UInt8](data)
    }

    private static func loadF32(_ name: String) -> [Float] {
        let bytes = loadBytes(goldenDir.appendingPathComponent(name))
        return bytes.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Float.self))
        }
    }

    private static func loadF64(_ name: String) -> [Double] {
        let bytes = loadBytes(goldenDir.appendingPathComponent(name))
        return bytes.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: Double.self))
        }
    }

    private static func loadU32(_ name: String) -> [UInt32] {
        let bytes = loadBytes(goldenDir.appendingPathComponent(name))
        return bytes.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: UInt32.self))
        }
    }

    private static func realTensor(_ stem: String) -> [UInt8] {
        loadBytes(realDir.appendingPathComponent(stem + ".bin"))
    }

    // MARK: - Error reporting

    /// maxAbs + block-relative error, printed in a grep-able form for the
    /// validation report. rel = maxAbs / max|reference|.
    @discardableResult
    private static func report(_ label: String,
                               actual: [Float],
                               reference64: [Double],
                               reference32: [Float]? = nil) -> (maxAbs: Float, rel: Float) {
        precondition(actual.count == reference64.count, "\(label): length mismatch")
        var maxAbs: Float = 0
        var refNorm: Float = 0
        for i in 0..<actual.count {
            maxAbs = max(maxAbs, abs(actual[i] - Float(reference64[i])))
            refNorm = max(refNorm, abs(Float(reference64[i])))
        }
        let rel = maxAbs / max(refNorm, 1e-6)
        if let r32 = reference32 {
            var d32: Float = 0
            for i in 0..<actual.count { d32 = max(d32, abs(actual[i] - r32[i])) }
            print("V4F-REAL \(label): maxAbs_vs_f64=\(maxAbs) rel_vs_f64=\(rel) maxAbs_vs_f32=\(d32)")
        } else {
            print("V4F-REAL \(label): maxAbs_vs_f64=\(maxAbs) rel_vs_f64=\(rel)")
        }
        return (maxAbs, rel)
    }

    /// fp16-output GEMV pass bar: 10x the fp16 rounding floor.
    private static let gemvTolerance: Float = 5e-3

    // MARK: - FP4 e2m1 GEMV on real routed-expert weights

    @Test(arguments: [
        (stem: "layers.5.ffn.experts.0.w1", key: "w1", m: 2048, n: 4096),
        (stem: "layers.5.ffn.experts.0.w2", key: "w2", m: 4096, n: 2048),
        (stem: "layers.5.ffn.experts.0.w3", key: "w3", m: 2048, n: 4096),
    ] as [(stem: String, key: String, m: Int, n: Int)])
    func fp4GemvOnRealExpertWeights(spec: (stem: String, key: String, m: Int, n: Int)) throws {
        guard Self.requireFixtures() else { return }
        let weights = Self.realTensor(spec.stem + ".weight")
        let scales = Self.realTensor(spec.stem + ".scale")
        #expect(weights.count == spec.m * spec.n / 2)
        #expect(scales.count == spec.m * spec.n / 32)
        let x = Self.loadF32("\(spec.key).x.f32.bin")
        let gold64 = Self.loadF64("\(spec.key).gemv.f64.bin")
        let gold32 = Self.loadF32("\(spec.key).gemv.f32.bin")
        #expect(x.count == spec.n && gold64.count == spec.m && gold32.count == spec.m)

        let ctx = try MetalContext()
        let kernel = try DequantFP4E2M1GEMV(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(bytes: weights, length: weights.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales, length: scales.count, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, values: x),
              let yBuf = Fp16Buffer.make(ctx.device, count: spec.m),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed"); return
        }
        kernel.encode(commandBuffer: cmd, weights: wBuf, scales: sBuf,
                      x: xBuf, y: yBuf, m: UInt32(spec.m), n: UInt32(spec.n))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let actual = Fp16Buffer.read(yBuf, count: spec.m)
        let result = Self.report("fp4-gemv-\(spec.key)", actual: actual,
                                 reference64: gold64, reference32: gold32)
        #expect(result.rel < Self.gemvTolerance,
                "FP4 GEMV \(spec.key): rel=\(result.rel) maxAbs=\(result.maxAbs)")
    }

    // MARK: - FP8 e4m3 128x128-block GEMV on real dense weights

    @Test(arguments: [
        (stem: "layers.5.attn.wq_a", key: "wq_a", m: 1024, n: 4096),
        (stem: "layers.5.attn.wq_b", key: "wq_b", m: 32768, n: 1024),
        (stem: "layers.5.ffn.shared_experts.w1", key: "shared_w1", m: 2048, n: 4096),
    ] as [(stem: String, key: String, m: Int, n: Int)])
    func fp8BlockGemvOnRealDenseWeights(spec: (stem: String, key: String, m: Int, n: Int)) throws {
        guard Self.requireFixtures() else { return }
        let codes = Self.realTensor(spec.stem + ".weight")
        let scales = Self.realTensor(spec.stem + ".scale")
        #expect(codes.count == spec.m * spec.n)
        #expect(scales.count == (spec.m / 128) * (spec.n / 128))
        let x = Self.loadF32("\(spec.key).x.f32.bin")
        let gold64 = Self.loadF64("\(spec.key).gemv.f64.bin")
        let gold32 = Self.loadF32("\(spec.key).gemv.f32.bin")
        #expect(x.count == spec.n && gold64.count == spec.m && gold32.count == spec.m)

        let ctx = try MetalContext()
        let kernel = try DequantFP8BlockGEMV(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(bytes: codes, length: codes.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales, length: scales.count, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, values: x),
              let yBuf = Fp16Buffer.make(ctx.device, count: spec.m),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed"); return
        }
        kernel.encode(commandBuffer: cmd, weights: wBuf, scales: sBuf,
                      x: xBuf, y: yBuf, m: UInt32(spec.m), n: UInt32(spec.n))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let actual = Fp16Buffer.read(yBuf, count: spec.m)
        let result = Self.report("fp8-gemv-\(spec.key)", actual: actual,
                                 reference64: gold64, reference32: gold32)
        #expect(result.rel < Self.gemvTolerance,
                "FP8 block GEMV \(spec.key): rel=\(result.rel) maxAbs=\(result.maxAbs)")
    }

    // MARK: - EmbedLookupFP8 exact dequant on a real wq_a row

    /// Row gather + e4m3 decode + block scale is exact math; the only error
    /// is the fp16 output store, so this pins the format interpretation
    /// (LUT, block grid, scale decode) element-by-element.
    @Test func embedLookupFP8OnRealWQA() throws {
        guard Self.requireFixtures() else { return }
        let m = 1024, d = 4096, token = 517
        let table = Self.realTensor("layers.5.attn.wq_a.weight")
        let scales = Self.realTensor("layers.5.attn.wq_a.scale")
        let dequant = Self.loadF32("wq_a.dequant.f32.bin")
        #expect(table.count == m * d && scales.count == (m / 128) * (d / 128))
        #expect(dequant.count == m * d)
        let goldRow = Array(dequant[(token * d)..<((token + 1) * d)])

        let ctx = try MetalContext()
        let kernel = try EmbedLookupFP8(context: ctx)
        guard let tBuf = ctx.device.makeBuffer(bytes: table, length: table.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales, length: scales.count, options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: d),
              let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed"); return
        }
        kernel.encode(commandBuffer: cmd, table: tBuf, scales: sBuf, out: outBuf,
                      tokenID: UInt32(token), d: UInt32(d), outScale: 1.0)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let actual = Fp16Buffer.read(outBuf, count: d)
        let rel = RelError.compute(actual: actual, reference: goldRow)
        let maxAbs = RelError.maxAbsDiff(actual, goldRow)
        print("V4F-REAL embed-lookup-wq_a-row\(token): maxAbs=\(maxAbs) rel=\(rel)")
        #expect(rel < 1e-3, "EmbedLookupFP8 wq_a row \(token): rel=\(rel) maxAbs=\(maxAbs)")
    }

    // MARK: - Router: raw BF16 GEMV logits + sqrt-softplus top-6 selection

    @Test func routerMatchesRealGateBytes() throws {
        guard Self.requireFixtures() else { return }
        let numExperts = 256, d = 4096
        let weights = Self.realTensor("layers.5.ffn.gate.weight")   // raw BF16
        let biasBytes = Self.realTensor("layers.5.ffn.gate.bias")   // raw F32
        #expect(weights.count == numExperts * d * 2)
        #expect(biasBytes.count == numExperts * 4)
        let bias = biasBytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let x = Self.loadF32("router.x.f32.bin")
        let goldLogits = Self.loadF64("router.logits.f64.bin")
        let goldIdx = Self.loadU32("router.indices.u32.bin")
        let goldWeights = Self.loadF32("router.weights.f32.bin")
        #expect(goldLogits.count == numExperts && goldIdx.count == 6 && goldWeights.count == 6)

        let ctx = try MetalContext()
        let kernel = try MoEV4(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(bytes: weights, length: weights.count, options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: bias, length: bias.count * 4, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, values: x),
              let logitsBuf = ctx.device.makeBuffer(length: numExperts * 4, options: .storageModeShared),
              let idxBuf = ctx.device.makeBuffer(length: 6 * 4, options: .storageModeShared),
              let wOutBuf = ctx.device.makeBuffer(length: 6 * 4, options: .storageModeShared) else {
            Issue.record("buffer allocation failed"); return
        }

        // (a) Raw BF16 GEMV: encode router_v4_gemv_bf16 straight from the
        // shared library so the logits can be compared against numpy f64.
        let gemvPSO = try ctx.pipeline("router_v4_gemv_bf16")
        guard let cmd = ctx.queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            Issue.record("encoder allocation failed"); return
        }
        enc.setComputePipelineState(gemvPSO)
        enc.setBuffer(wBuf, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        enc.setBuffer(logitsBuf, offset: 0, index: 2)
        var experts = UInt32(numExperts), dim = UInt32(d)
        enc.setBytes(&experts, length: 4, index: 3)
        enc.setBytes(&dim, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (numExperts + 3) / 4, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        let logits = logitsBuf.contents().bindMemory(to: Float.self, capacity: numExperts)
        let actualLogits = (0..<numExperts).map { logits[$0] }
        let logitErr = Self.report("router-gemv-logits", actual: actualLogits,
                                   reference64: goldLogits)
        #expect(logitErr.rel < Self.gemvTolerance,
                "router logits: rel=\(logitErr.rel) maxAbs=\(logitErr.maxAbs)")

        // (b) Full router path: GEMV + sqrt-softplus top-6 selection.
        guard let cmd2 = ctx.queue.makeCommandBuffer() else {
            Issue.record("command buffer allocation failed"); return
        }
        kernel.encodeRouterV4(commandBuffer: cmd2, weights: wBuf, bias: bBuf,
                              hidden: xBuf, outIndices: idxBuf, outWeights: wOutBuf,
                              numExperts: UInt32(numExperts), d: UInt32(d),
                              routeScale: 1.5)
        cmd2.commit()
        cmd2.waitUntilCompleted()
        #expect(cmd2.error == nil)

        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: 6)
        let actualIdx = (0..<6).map { idxPtr[$0] }
        let wPtr = wOutBuf.contents().bindMemory(to: Float.self, capacity: 6)
        let actualWeights = (0..<6).map { wPtr[$0] }
        let maxWeightErr = zip(actualWeights, goldWeights).map { abs($0 - $1) }.max() ?? 0
        print("V4F-REAL router-select: indices=\(actualIdx) golden=\(goldIdx) maxWeightErr=\(maxWeightErr)")
        // Selection indices must match exactly.
        #expect(actualIdx == goldIdx,
                "router top-6 mismatch: got \(actualIdx), golden \(goldIdx)")
        #expect(maxWeightErr < 5e-3, "router weights maxErr=\(maxWeightErr)")
    }

    // MARK: - Fused SwiGLU MoE phase1/phase2 on real expert 0

    /// Builds the six-sub-tensor V4 expert blob from the real w1/w3/w2 bytes
    /// and replicates it into all six streamed slots. Phase 1 acts must match
    /// the numpy clamped-SwiGLU golden; phase 2 reduces with the fixed
    /// routing weights so out = residual + sum(w) * (w2 @ act).
    @Test func fusedMoEPhase1Phase2OnRealExpert0() throws {
        guard Self.requireFixtures() else { return }
        let d = 4096, f = 2048, topK = 6
        let w1 = Self.realTensor("layers.5.ffn.experts.0.w1.weight")
        let s1 = Self.realTensor("layers.5.ffn.experts.0.w1.scale")
        let w2 = Self.realTensor("layers.5.ffn.experts.0.w2.weight")
        let s2 = Self.realTensor("layers.5.ffn.experts.0.w2.scale")
        let w3 = Self.realTensor("layers.5.ffn.experts.0.w3.weight")
        let s3 = Self.realTensor("layers.5.ffn.experts.0.w3.scale")
        #expect(w1.count == f * d / 2 && s1.count == f * d / 32)
        #expect(w2.count == d * f / 2 && s2.count == d * f / 32)

        var blob = [UInt8]()
        func appendPadded(_ values: [UInt8]) -> UInt32 {
            let off = UInt32(blob.count)
            blob.append(contentsOf: values)
            while !blob.count.isMultiple(of: 4) { blob.append(0) }
            return off
        }
        let offsets = V4ExpertOffsets(
            gateWOff: appendPadded(w1), gateSOff: appendPadded(s1),
            upWOff: appendPadded(w3), upSOff: appendPadded(s3),
            downWOff: appendPadded(w2), downSOff: appendPadded(s2))

        let x = Self.loadF32("moe.x.f32.bin")
        let goldActs = Self.loadF32("moe.acts.f32.bin")
        let residual = Self.loadF32("moe.residual.f32.bin")
        let routing = Self.loadF32("moe.routing.f32.bin")
        let goldOut64 = Self.loadF64("moe.out.f64.bin")
        let goldOut32 = Self.loadF32("moe.out.f32.bin")
        #expect(x.count == d && goldActs.count == f && residual.count == d
                && routing.count == topK && goldOut64.count == d)

        let ctx = try MetalContext()
        let kernel = try MoEV4(context: ctx)
        let blobs = try (0..<topK).map { _ -> MTLBuffer in
            guard let buf = ctx.device.makeBuffer(bytes: blob, length: blob.count,
                                                  options: .storageModeShared) else {
                Issue.record("blob allocation failed")
                throw CocoaError(.coderReadCorrupt)
            }
            return buf
        }
        guard let argBuf = kernel.makeRoutedArgumentBuffer(routedBlobs: blobs, topK: UInt32(topK)),
              let xBuf = Fp16Buffer.make(ctx.device, values: x),
              let resBuf = Fp16Buffer.make(ctx.device, values: residual),
              let routingBuf = ctx.device.makeBuffer(bytes: routing, length: routing.count * 4,
                                                     options: .storageModeShared),
              let actsBuf = Fp16Buffer.make(ctx.device, count: topK * f),
              let outBuf = Fp16Buffer.make(ctx.device, count: d) else {
            Issue.record("buffer allocation failed"); return
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("command buffer allocation failed"); return
        }
        kernel.encodeRoutedPhase1SwiGLU(commandBuffer: cmd, routedArgBuffer: argBuf,
                                        routedBlobs: blobs, routedOffsets: offsets,
                                        x: xBuf, acts: actsBuf,
                                        d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
        kernel.encodeRoutedPhase2Reduce(commandBuffer: cmd, routedArgBuffer: argBuf,
                                        routedBlobs: blobs, routedOffsets: offsets,
                                        acts: actsBuf, routingWeights: routingBuf,
                                        residual: resBuf, y: outBuf,
                                        d: UInt32(d), f: UInt32(f), topK: UInt32(topK))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let acts = Fp16Buffer.read(actsBuf, count: topK * f)
        // Slot 0 against the numpy SwiGLU golden; every other slot must be
        // bit-identical (same blob, same x, deterministic kernel).
        let slot0 = Array(acts[0..<f])
        let actsRel = RelError.compute(actual: slot0, reference: goldActs)
        let actsMaxAbs = RelError.maxAbsDiff(slot0, goldActs)
        print("V4F-REAL moe-phase1-acts: maxAbs=\(actsMaxAbs) rel=\(actsRel)")
        #expect(actsRel < Self.gemvTolerance,
                "MoE phase1 acts: rel=\(actsRel) maxAbs=\(actsMaxAbs)")
        for slot in 1..<topK {
            let slotActs = Array(acts[(slot * f)..<((slot + 1) * f)])
            #expect(slotActs == slot0, "MoE phase1 slot \(slot) differs from slot 0")
        }

        let out = Fp16Buffer.read(outBuf, count: d)
        let result = Self.report("moe-phase2-out", actual: out,
                                 reference64: goldOut64, reference32: goldOut32)
        #expect(result.rel < Self.gemvTolerance,
                "MoE phase2 out: rel=\(result.rel) maxAbs=\(result.maxAbs)")
    }
}
