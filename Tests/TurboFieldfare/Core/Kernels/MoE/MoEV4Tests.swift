import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// End-to-end check of the fused V4 routed MoE (FP4 e2m1 + ue8m0 expert
/// dequant, clamped SwiGLU phase 1, weighted top-6 reduce phase 2) and the
/// BF16 sqrt-softplus router against a CPU reference transcribed from the
/// official inference code (V4F-reference-notes §5):
///   up = clamp(up, ±10), gate = min(gate, 10) (no lower clamp),
///   act = silu(gate) * up, y = residual + Σ w_slot * (w2 · act_slot).
@Suite struct MoEV4Tests {
    private static let dimension = 256     // 8 FP4 groups: one full kernel block
    private static let intermediate = 224  // 7 FP4 groups: remainder path only
    private static let topK = 6

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: V4ExpertOffsets
    }

    private struct Fixture {
        let gates: [[V4Quantization.FP4Row]]   // [slot][F]
        let ups: [[V4Quantization.FP4Row]]
        let downs: [[V4Quantization.FP4Row]]   // [slot][D]
        let x: [Float]                          // half-rounded
        let residual: [Float]
        let routingWeights: [Float]
    }

    // MARK: - CPU reference

    /// Stable silu matching the kernel: sigmoid via exp(-|g|).
    private static func silu(_ g: Float) -> Float {
        let e = exp(-abs(g))
        let sig = g >= 0 ? 1 / (1 + e) : e / (1 + e)
        return g * sig
    }

    private static func swigluAct(gate: Float, up: Float) -> Float {
        silu(min(gate, 10)) * min(10, max(-10, up))
    }

    private static func dot(_ a: [Float], _ b: [Float]) -> Float {
        var acc: Float = 0
        for i in 0..<a.count { acc += a[i] * b[i] }
        return acc
    }

    /// Phase 1 in FP32, with the kernel's half store of `acts` reproduced.
    private static func referencePhase1(_ fixture: Fixture) -> [Float] {
        var acts = [Float](repeating: 0, count: Self.topK * Self.intermediate)
        for slot in 0..<Self.topK {
            for f in 0..<Self.intermediate {
                let gRow = V4Quantization.dequantizeFP4Row(fixture.gates[slot][f],
                                                           n: Self.dimension)
                let uRow = V4Quantization.dequantizeFP4Row(fixture.ups[slot][f],
                                                           n: Self.dimension)
                let act = swigluAct(gate: dot(gRow, fixture.x),
                                    up: dot(uRow, fixture.x))
                acts[slot * Self.intermediate + f] = Float(Float16(act))
            }
        }
        return acts
    }

    private static func referencePhase2(_ fixture: Fixture, acts: [Float]) -> [Float] {
        var out = fixture.residual
        for d in 0..<Self.dimension {
            var acc: Float = 0
            for slot in 0..<Self.topK {
                let wRow = V4Quantization.dequantizeFP4Row(fixture.downs[slot][d],
                                                           n: Self.intermediate)
                let actSlot = Array(acts[(slot * Self.intermediate)..<((slot + 1) * Self.intermediate)])
                acc += fixture.routingWeights[slot] * dot(wRow, actSlot)
            }
            out[d] += acc
        }
        return out
    }

    // MARK: - Fixture construction

    private static func makeFixture(seed: UInt64,
                                    clampExpert: Bool) -> Fixture {
        var rng = SeedTree(seed).key("moe-v4-fixture-clamp\(clampExpert)")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in (0..<columns).map { _ in rng.uniform(-0.4, 0.4) } }
        }
        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<Self.topK {
            gates.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            ups.append(matrix(rows: Self.intermediate, columns: Self.dimension))
            downs.append(matrix(rows: Self.dimension, columns: Self.intermediate))
        }
        let x = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        if clampExpert {
            // Drive gate/up pre-activations well past ±10 so the SwiGLU clamp
            // (and the gate's missing lower clamp) is genuinely exercised.
            // Rows proportional to sign(x) make the dots ±0.5·Σ|x| ≈ ±30,
            // deterministic in sign: expert 0 gates clamp at +10, expert 1
            // gates go deeply negative (no lower clamp → silu ≈ 0), up rows
            // alternate both clamp signs.
            let s = x.map { $0 >= 0 ? Float(0.5) : Float(-0.5) }
            for f in 0..<Self.intermediate {
                gates[0][f] = s
                gates[1][f] = s.map { -$0 }
                ups[0][f] = f.isMultiple(of: 2) ? s : s.map { -$0 }
            }
        }
        let residual = (0..<Self.dimension).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        let routingWeights = (0..<Self.topK).map { 0.1 + Float($0) * 0.05 }
        return Fixture(
            gates: gates.map { $0.map { V4Quantization.quantizeFP4Row($0) } },
            ups: ups.map { $0.map { V4Quantization.quantizeFP4Row($0) } },
            downs: downs.map { $0.map { V4Quantization.quantizeFP4Row($0) } },
            x: x, residual: residual, routingWeights: routingWeights)
    }

    private static func makeBlob(gate: [V4Quantization.FP4Row],
                                 up: [V4Quantization.FP4Row],
                                 down: [V4Quantization.FP4Row]) -> RoutedBlob {
        var bytes = [UInt8]()
        func appendPadded(_ values: [UInt8]) -> UInt32 {
            let off = UInt32(bytes.count)
            bytes.append(contentsOf: values)
            while !bytes.count.isMultiple(of: 4) { bytes.append(0) }
            return off
        }
        let gateW = appendPadded(gate.flatMap(\.packed))
        let gateS = appendPadded(gate.flatMap(\.scales))
        let upW = appendPadded(up.flatMap(\.packed))
        let upS = appendPadded(up.flatMap(\.scales))
        let downW = appendPadded(down.flatMap(\.packed))
        let downS = appendPadded(down.flatMap(\.scales))
        return RoutedBlob(
            bytes: bytes,
            offsets: V4ExpertOffsets(
                gateWOff: gateW, gateSOff: gateS,
                upWOff: upW, upSOff: upS,
                downWOff: downW, downSOff: downS))
    }

    private struct GPU {
        let context: MetalContext
        let kernel: MoEV4
        let blobs: [RoutedBlob]
        let routedBuffers: [MTLBuffer]
        let argumentBuffer: MTLBuffer
        let xBuffer: MTLBuffer
        let residualBuffer: MTLBuffer
        let routingBuffer: MTLBuffer
    }

    private static func makeGPU(_ fixture: Fixture) throws -> GPU {
        let context = try MetalContext()
        let kernel = try MoEV4(context: context)
        let blobs = (0..<Self.topK).map {
            makeBlob(gate: fixture.gates[$0], up: fixture.ups[$0], down: fixture.downs[$0])
        }
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == Self.topK,
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers, topK: UInt32(Self.topK)),
              let xBuffer = Fp16Buffer.make(context.device, values: fixture.x),
              let residualBuffer = Fp16Buffer.make(context.device, values: fixture.residual),
              let routingBuffer = context.device.makeBuffer(
                bytes: fixture.routingWeights,
                length: fixture.routingWeights.count * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            Issue.record("buffer allocation failed")
            throw CocoaError(.fileReadUnknown)
        }
        return GPU(context: context, kernel: kernel, blobs: blobs,
                   routedBuffers: routedBuffers, argumentBuffer: argumentBuffer,
                   xBuffer: xBuffer, residualBuffer: residualBuffer,
                   routingBuffer: routingBuffer)
    }

    private static func runPhase1(_ gpu: GPU, acts: MTLBuffer) {
        let cmd = gpu.context.queue.makeCommandBuffer()!
        gpu.kernel.encodeRoutedPhase1SwiGLU(
            commandBuffer: cmd,
            routedArgBuffer: gpu.argumentBuffer,
            routedBlobs: gpu.routedBuffers,
            routedOffsets: gpu.blobs[0].offsets,
            x: gpu.xBuffer,
            acts: acts,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
    }

    private static func runPhase2(_ gpu: GPU, acts: MTLBuffer, out: MTLBuffer) {
        let cmd = gpu.context.queue.makeCommandBuffer()!
        gpu.kernel.encodeRoutedPhase2Reduce(
            commandBuffer: cmd,
            routedArgBuffer: gpu.argumentBuffer,
            routedBlobs: gpu.routedBuffers,
            routedOffsets: gpu.blobs[0].offsets,
            acts: acts,
            routingWeights: gpu.routingBuffer,
            residual: gpu.residualBuffer,
            y: out,
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
    }

    // MARK: - Tests

    @Test func phase1FullAndSubsetMatchReference() throws {
        let fixture = Self.makeFixture(seed: 0x51, clampExpert: false)
        let gpu = try Self.makeGPU(fixture)
        let count = Self.topK * Self.intermediate
        guard let fullActs = Fp16Buffer.make(gpu.context.device, count: count),
              let splitActs = Fp16Buffer.make(gpu.context.device, count: count) else {
            Issue.record("buffer allocation failed"); return
        }

        Self.runPhase1(gpu, acts: fullActs)

        // Subset path in two waves: slots 0...2 then 3...5.
        for slots in [[UInt32](0...2), [UInt32](3...5)] {
            let slotBuffer = gpu.context.device.makeBuffer(
                bytes: slots,
                length: slots.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared)!
            let cmd = gpu.context.queue.makeCommandBuffer()!
            gpu.kernel.encodeRoutedPhase1SwiGLUSubset(
                commandBuffer: cmd,
                routedArgBuffer: gpu.argumentBuffer,
                routedBlobs: gpu.routedBuffers,
                routedOffsets: gpu.blobs[0].offsets,
                x: gpu.xBuffer,
                acts: splitActs,
                activeSlots: slotBuffer,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(Self.dimension),
                f: UInt32(Self.intermediate),
                topK: UInt32(Self.topK))
            cmd.commit()
            cmd.waitUntilCompleted()
            #expect(cmd.error == nil)
        }

        let full = Fp16Buffer.read(fullActs, count: count)
        let split = Fp16Buffer.read(splitActs, count: count)
        // The subset kernel computes identical math per row: exact equality.
        #expect(full == split)

        let reference = Self.referencePhase1(fixture)
        let rel = RelError.compute(actual: full, reference: reference)
        #expect(rel < Tolerance.fp16ChainedReduction, "phase1 rel=\(rel)")
    }

    @Test(arguments: [1, 3, 6] as [Int])
    func phase1SubsetActiveCounts(activeCount: Int) throws {
        let fixture = Self.makeFixture(seed: 0x52, clampExpert: false)
        let gpu = try Self.makeGPU(fixture)
        let count = Self.topK * Self.intermediate
        guard let fullActs = Fp16Buffer.make(gpu.context.device, count: count),
              let subsetActs = Fp16Buffer.make(gpu.context.device, count: count) else {
            Issue.record("buffer allocation failed"); return
        }
        Self.runPhase1(gpu, acts: fullActs)

        let slots = [UInt32]((0..<activeCount).map { UInt32($0) })
        let slotBuffer = gpu.context.device.makeBuffer(
            bytes: slots,
            length: slots.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared)!
        let cmd = gpu.context.queue.makeCommandBuffer()!
        gpu.kernel.encodeRoutedPhase1SwiGLUSubset(
            commandBuffer: cmd,
            routedArgBuffer: gpu.argumentBuffer,
            routedBlobs: gpu.routedBuffers,
            routedOffsets: gpu.blobs[0].offsets,
            x: gpu.xBuffer,
            acts: subsetActs,
            activeSlots: slotBuffer,
            activeSlotIndices: slots,
            activeCount: UInt32(activeCount),
            d: UInt32(Self.dimension),
            f: UInt32(Self.intermediate),
            topK: UInt32(Self.topK))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let full = Fp16Buffer.read(fullActs, count: count)
        let subset = Fp16Buffer.read(subsetActs, count: count)
        for slot in 0..<activeCount {
            let lo = slot * Self.intermediate
            let hi = lo + Self.intermediate
            #expect(Array(subset[lo..<hi]) == Array(full[lo..<hi]),
                    "slot \(slot) mismatch at activeCount=\(activeCount)")
        }
    }

    @Test func fullPipelineMatchesReference() throws {
        let fixture = Self.makeFixture(seed: 0x53, clampExpert: false)
        let gpu = try Self.makeGPU(fixture)
        let count = Self.topK * Self.intermediate
        guard let acts = Fp16Buffer.make(gpu.context.device, count: count),
              let out = Fp16Buffer.make(gpu.context.device, count: Self.dimension) else {
            Issue.record("buffer allocation failed"); return
        }
        Self.runPhase1(gpu, acts: acts)
        Self.runPhase2(gpu, acts: acts, out: out)

        let refActs = Self.referencePhase1(fixture)
        let refOut = Self.referencePhase2(fixture, acts: refActs)
        let actual = Fp16Buffer.read(out, count: Self.dimension)
        let rel = RelError.compute(actual: actual, reference: refOut)
        let maxAbs = RelError.maxAbsDiff(actual, refOut)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "pipeline rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// Clamp-exercising case: expert 0's gate/up rows drive pre-activations
    /// past ±10 (both up-clamp signs and the gate upper clamp; the deeply
    /// negative half of `up` also proves silu sees the UNclamped-below gate).
    @Test func swigluClampCasesMatchReference() throws {
        let fixture = Self.makeFixture(seed: 0x54, clampExpert: true)
        // Prove the fixture actually engages the clamps, otherwise the test
        // would pass vacuously.
        let gRow = V4Quantization.dequantizeFP4Row(fixture.gates[0][0], n: Self.dimension)
        let uRow = V4Quantization.dequantizeFP4Row(fixture.ups[0][0], n: Self.dimension)
        let gNegRow = V4Quantization.dequantizeFP4Row(fixture.gates[1][0], n: Self.dimension)
        #expect(Self.dot(gRow, fixture.x) > 10)       // gate upper clamp
        #expect(Self.dot(gNegRow, fixture.x) < -10)   // gate: NO lower clamp
        #expect(abs(Self.dot(uRow, fixture.x)) > 10)  // up clamp

        let gpu = try Self.makeGPU(fixture)
        let count = Self.topK * Self.intermediate
        guard let acts = Fp16Buffer.make(gpu.context.device, count: count),
              let out = Fp16Buffer.make(gpu.context.device, count: Self.dimension) else {
            Issue.record("buffer allocation failed"); return
        }
        Self.runPhase1(gpu, acts: acts)
        Self.runPhase2(gpu, acts: acts, out: out)

        let refActs = Self.referencePhase1(fixture)
        let gpuActs = Fp16Buffer.read(acts, count: count)
        // slot 0 carries the clamped rows; compare it directly.
        let slotActs = Array(gpuActs[0..<Self.intermediate])
        let slotRef = Array(refActs[0..<Self.intermediate])
        #expect(RelError.compute(actual: slotActs, reference: slotRef)
                < Tolerance.fp16ChainedReduction)

        let refOut = Self.referencePhase2(fixture, acts: refActs)
        let actual = Fp16Buffer.read(out, count: Self.dimension)
        let rel = RelError.compute(actual: actual, reference: refOut)
        #expect(rel < Tolerance.fp16ChainedReduction, "clamp pipeline rel=\(rel)")
    }

    // MARK: - Router

    private static func readFloats(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { ptr[$0] }
    }

    private static func routerReference(logits: [Float],
                                        bias: [Float],
                                        routeScale: Float) -> (indices: [UInt32], weights: [Float]) {
        var paired: [(sel: Float, score: Float, index: UInt32)] = []
        for e in 0..<logits.count {
            let l = logits[e]
            let sp = l > 20 ? l : log(1 + exp(l))
            let s = sqrt(max(sp, 0))
            paired.append((sel: s + bias[e], score: s, index: UInt32(e)))
        }
        paired.sort { lhs, rhs in
            lhs.sel == rhs.sel ? lhs.index < rhs.index : lhs.sel > rhs.sel
        }
        let selected = Array(paired.prefix(Self.topK))
        let sum = selected.reduce(Float(0)) { $0 + $1.score }
        return (selected.map(\.index), selected.map { $0.score / sum * routeScale })
    }

    private static func runRouter(weightsBF16: [UInt16],
                                  bias: [Float],
                                  hidden: [Float],
                                  numExperts: Int,
                                  d: Int,
                                  routeScale: Float) throws
        -> (indices: [UInt32], weights: [Float])
    {
        let context = try MetalContext()
        let kernel = try MoEV4(context: context)
        guard let wBuf = context.device.makeBuffer(
                bytes: weightsBF16,
                length: weightsBF16.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let bBuf = context.device.makeBuffer(
                bytes: bias,
                length: bias.count * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let hBuf = Fp16Buffer.make(context.device, values: hidden),
              let idxBuf = context.device.makeBuffer(
                length: Self.topK * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let wOutBuf = context.device.makeBuffer(
                length: Self.topK * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let cmd = context.queue.makeCommandBuffer() else {
            Issue.record("buffer allocation failed")
            throw CocoaError(.fileReadUnknown)
        }
        kernel.encodeRouterV4(commandBuffer: cmd,
                              weights: wBuf,
                              bias: bBuf,
                              hidden: hBuf,
                              outIndices: idxBuf,
                              outWeights: wOutBuf,
                              numExperts: UInt32(numExperts),
                              d: UInt32(d),
                              routeScale: routeScale)
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)
        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: Self.topK)
        return (indices: (0..<Self.topK).map { idxPtr[$0] },
                weights: readFloats(wOutBuf, count: Self.topK))
    }

    private static func routerLogitsReference(weightsBF16: [UInt16],
                                              hidden: [Float],
                                              numExperts: Int,
                                              d: Int) -> [Float] {
        (0..<numExperts).map { e in
            var acc: Float = 0
            for i in 0..<d {
                acc += Quantization.bf16ToFloat(weightsBF16[e * d + i]) * hidden[i]
            }
            return acc
        }
    }

    @Test func routerMatchesReferenceScoring() throws {
        let experts = 64
        let d = 128
        let routeScale: Float = 1.5
        var rng = SeedTree(0x55).key("moe-v4-router")
        let weights = (0..<experts).map { _ in
            (0..<d).map { _ in rng.uniform(-0.05, 0.05) }
        }
        let bias = (0..<experts).map { _ in rng.uniform(-0.3, 0.3) }
        let hidden = (0..<d).map { _ in Float(Float16(rng.uniform(-1.0, 1.0))) }
        let weightsBF16 = weights.flatMap { $0.map(Quantization.bf16Bits) }

        let logits = Self.routerLogitsReference(weightsBF16: weightsBF16,
                                                hidden: hidden,
                                                numExperts: experts, d: d)
        let expected = Self.routerReference(logits: logits, bias: bias,
                                            routeScale: routeScale)
        let actual = try Self.runRouter(weightsBF16: weightsBF16,
                                        bias: bias,
                                        hidden: hidden,
                                        numExperts: experts,
                                        d: d,
                                        routeScale: routeScale)
        #expect(actual.indices == expected.indices)
        let maxError = zip(actual.weights, expected.weights)
            .map { abs($0 - $1) }
            .max() ?? 0
        #expect(maxError < 5e-3)
    }

    /// Two experts with identical rows and zero bias produce identical
    /// selection scores; torch.topk order (and the kernel) take the lower
    /// index first.
    @Test func routerResolvesTiesToLowerIndex() throws {
        let experts = 8
        let d = 64
        let routeScale: Float = 1.5
        var rng = SeedTree(0x56).key("moe-v4-router-tie")
        var weights = (0..<experts).map { _ in
            (0..<d).map { _ in rng.uniform(-0.05, 0.05) }
        }
        // Experts 2 and 5 are identical, maximally-scoring rows (hidden is
        // all-positive): both must be selected, tied, lower index first.
        weights[2] = [Float](repeating: 0.05, count: d)
        weights[5] = [Float](repeating: 0.05, count: d)
        let bias = [Float](repeating: 0, count: experts)
        let hidden = (0..<d).map { _ in Float(Float16(rng.uniform(0.5, 1.0))) }
        let weightsBF16 = weights.flatMap { $0.map(Quantization.bf16Bits) }

        let logits = Self.routerLogitsReference(weightsBF16: weightsBF16,
                                                hidden: hidden,
                                                numExperts: experts, d: d)
        let expected = Self.routerReference(logits: logits, bias: bias,
                                            routeScale: routeScale)
        let actual = try Self.runRouter(weightsBF16: weightsBF16,
                                        bias: bias,
                                        hidden: hidden,
                                        numExperts: experts,
                                        d: d,
                                        routeScale: routeScale)
        #expect(actual.indices == expected.indices)
        #expect(actual.indices.first == 2)
        #expect(actual.indices.dropFirst().first == 5)
    }
}
