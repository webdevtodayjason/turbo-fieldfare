import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// GPU-vs-CPU validation of the V4F-06b grouped routed-MoE prefill path:
/// `prefill_moe_v4_grouped_phase1_gate_up_swiglu` (FP4 e2m1 + ue8m0 dequant,
/// clamped SwiGLU over gathered token rows), `prefill_moe_v4_grouped_down_scatter`
/// (FP4 down projection into token-major partials), and
/// `prefill_moe_v4_reduce_token_major` (F32 routing-weighted reduce).
///
/// The CPU reference is transcribed from the official inference code
/// (V4F-reference-notes §5): up = clamp(up, ±10), gate = min(gate, 10) (no
/// lower clamp), act = silu(gate) * up, y_token = Σ_r w_r * (w2 · act_r).
@Suite struct PrefillGroupedRoutedMoEV4Tests {
    private static let dimension = 256      // 8 FP4 groups: one full kernel block
    private static let intermediate = 224   // 7 FP4 groups: remainder path only
    private static let topK = 6

    private struct ExpertFixture {
        let gates: [V4Quantization.FP4Row]  // [F] rows over D
        let ups: [V4Quantization.FP4Row]
        let downs: [V4Quantization.FP4Row]  // [D] rows over F
        let blob: [UInt8]
        let offsets: V4ExpertOffsets
    }

    // MARK: - CPU reference

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

    /// FP4-row dot against an FP32 activation without materializing the
    /// dequantized row (same math as dequantizeFP4Row + dot, minus the
    /// per-row allocation that dominates debug-build runtime).
    private static func dotFP4Row(_ row: V4Quantization.FP4Row,
                                  _ x: [Float],
                                  n: Int) -> Float {
        let g = V4Quantization.fp4GroupSize
        var acc: Float = 0
        for group in 0..<(n / g) {
            let scale = V4Quantization.ue8m0Decode(row.scales[group])
            var dot: Float = 0
            for k in 0..<(g / 2) {
                let byte = row.packed[group * (g / 2) + k]
                dot += V4Quantization.e2m1Decode(byte & 0x0F) * x[group * g + 2 * k]
                dot += V4Quantization.e2m1Decode(byte >> 4) * x[group * g + 2 * k + 1]
            }
            acc += scale * dot
        }
        return acc
    }

    /// Full pipeline reference with the kernel's stores reproduced: acts and
    /// h2 round through Float16, while each routing-weighted down-projection
    /// rounds through Float32 before the six-route reduction adds the shared
    /// expert residual.
    private static func reference(pairs: [PrefillTokenExpertPair],
                                  experts: [ExpertFixture],
                                  hidden: [[Float]],
                                  weights: [Float],
                                  residual: [[Float]]? = nil,
                                  d: Int,
                                  f: Int) -> [Float] {
        let t = hidden.count
        var partials = [Float](repeating: 0, count: t * Self.topK * d)
        // Pairs sharing a (token, expert) produce identical activations and
        // down outputs; compute them once (matters at full geometry).
        var actsCache: [UInt64: [Float]] = [:]
        var downCache: [UInt64: [Float]] = [:]
        for pair in pairs {
            let key = UInt64(pair.token) << 32 | UInt64(pair.expert)
            let expert = experts[Int(pair.expert)]
            let acts: [Float]
            if let cached = actsCache[key] {
                acts = cached
            } else {
                let x = hidden[Int(pair.token)]
                var computed = [Float](repeating: 0, count: f)
                for ff in 0..<f {
                    let gate = dotFP4Row(expert.gates[ff], x, n: d)
                    let up = dotFP4Row(expert.ups[ff], x, n: d)
                    computed[ff] = Float(Float16(swigluAct(gate: gate, up: up)))
                }
                actsCache[key] = computed
                acts = computed
            }
            let base = (Int(pair.token) * Self.topK + Int(pair.rank)) * d
            if let cached = downCache[key] {
                partials.replaceSubrange(base..<(base + d), with: cached)
            } else {
                var down = [Float](repeating: 0, count: d)
                for dd in 0..<d {
                    let value = dotFP4Row(expert.downs[dd], acts, n: f)
                    down[dd] = value
                }
                downCache[key] = down
                partials.replaceSubrange(base..<(base + d), with: down)
            }
        }
        var h2 = [Float](repeating: 0, count: t * d)
        for token in 0..<t {
            for dd in 0..<d {
                var acc = residual?[token][dd] ?? 0
                for r in 0..<Self.topK {
                    let weighted = weights[token * Self.topK + r]
                        * partials[(token * Self.topK + r) * d + dd]
                    acc += weighted
                }
                h2[token * d + dd] = Float(Float16(acc))
            }
        }
        return h2
    }

    // MARK: - Fixture construction

    private static func makeBlob(gate: [V4Quantization.FP4Row],
                                 up: [V4Quantization.FP4Row],
                                 down: [V4Quantization.FP4Row]) -> (bytes: [UInt8],
                                                                    offsets: V4ExpertOffsets) {
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
        return (bytes, V4ExpertOffsets(
            gateWOff: gateW, gateSOff: gateS,
            upWOff: upW, upSOff: upS,
            downWOff: downW, downSOff: downS))
    }

    private static func makeExpert(seed: UInt64,
                                   label: String,
                                   d: Int,
                                   f: Int) -> ExpertFixture {
        var rng = SeedTree(seed).key(label)
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in (0..<columns).map { _ in rng.uniform(-0.4, 0.4) } }
        }
        let gates = matrix(rows: f, columns: d).map { V4Quantization.quantizeFP4Row($0) }
        let ups = matrix(rows: f, columns: d).map { V4Quantization.quantizeFP4Row($0) }
        let downs = matrix(rows: d, columns: f).map { V4Quantization.quantizeFP4Row($0) }
        let (bytes, offsets) = makeBlob(gate: gates, up: ups, down: downs)
        return ExpertFixture(gates: gates, ups: ups, downs: downs,
                             blob: bytes, offsets: offsets)
    }

    private static func makePool(numExperts: Int,
                                 seed: UInt64,
                                 d: Int,
                                 f: Int) -> [ExpertFixture] {
        (0..<numExperts).map { makeExpert(seed: seed, label: "expert-\($0)", d: d, f: f) }
    }

    /// Random FP4 payloads directly as codes (random nibbles, ue8m0 scale
    /// bytes in 2^-6...2^-2 so pre-activations span but do not saturate the
    /// SwiGLU clamps). Equally valid kernel input as quantized floats, and
    /// O(payload bytes) to generate: used for the full-geometry test where
    /// per-element quantization would dominate the suite runtime.
    private static func makeRandomCodeExpert(seed: UInt64,
                                             label: String,
                                             d: Int,
                                             f: Int) -> ExpertFixture {
        var rng = SeedTree(seed).key(label)
        func rows(_ rowCount: Int, _ n: Int) -> [V4Quantization.FP4Row] {
            (0..<rowCount).map { _ in
                var packed = [UInt8](repeating: 0, count: n / 2)
                for i in packed.indices {
                    packed[i] = UInt8.random(in: 0...255, using: &rng)
                }
                var scales = [UInt8](repeating: 0, count: n / V4Quantization.fp4GroupSize)
                for i in scales.indices {
                    scales[i] = UInt8.random(in: 121...125, using: &rng)
                }
                return V4Quantization.FP4Row(packed: packed, scales: scales)
            }
        }
        let gates = rows(f, d)
        let ups = rows(f, d)
        let downs = rows(d, f)
        let (bytes, offsets) = makeBlob(gate: gates, up: ups, down: downs)
        return ExpertFixture(gates: gates, ups: ups, downs: downs,
                             blob: bytes, offsets: offsets)
    }

    /// Route weights emulating the router output: positive scores normalized
    /// by their sum and scaled by route_scale 1.5.
    private static func makeWeights(seed: UInt64, tokens: Int) -> [Float] {
        var rng = SeedTree(seed).key("route-weights")
        var out = [Float]()
        for _ in 0..<tokens {
            let scores = (0..<Self.topK).map { _ in rng.uniform(0.05, 1.0) }
            let sum = scores.reduce(0, +)
            out.append(contentsOf: scores.map { $0 / sum * 1.5 })
        }
        return out
    }

    private static func makePairs(assignments: [[Int]]) -> [PrefillTokenExpertPair] {
        var pairs: [PrefillTokenExpertPair] = []
        for (token, experts) in assignments.enumerated() {
            precondition(experts.count == Self.topK)
            for (rank, expert) in experts.enumerated() {
                pairs.append(PrefillTokenExpertPair(token: UInt32(token),
                                                    expert: UInt32(expert),
                                                    rank: UInt32(rank),
                                                    weight: Float16(0)))
            }
        }
        return pairs
    }

    // MARK: - GPU driver

    /// Runs the tile loop exactly as the prefill runner will: one command
    /// buffer per tile (phase 1 + down scatter), drained before the next,
    /// then a single token-major reduce. Expert blob buffers carry a 64-byte
    /// header so every view offset is nonzero, exercising the argument
    /// encoder's offset plumbing.
    private static func run(pool: [ExpertFixture],
                            pairs: [PrefillTokenExpertPair],
                            hidden: [[Float]],
                            weights: [Float],
                            residual: [[Float]]? = nil,
                            numExperts: Int,
                            d: Int,
                            f: Int,
                            pairMicrobatchRows: Int = 32) throws -> (h2: [Float],
                                                                     tiles: Int,
                                                                     microbatches: [Int]) {
        let ctx = try MetalContext()
        let driver = try PrefillGroupedRoutedMoEV4(context: ctx)
        let routes = try PrefillGroupedRoutedMoEV4.groupedRoutes(
            pairs: pairs, queryCount: hidden.count, numExperts: numExperts)

        // One buffer per expert with a 64-byte header.
        var views: [TensorView] = []
        for (expertID, fixture) in pool.enumerated() {
            var bytes = [UInt8](repeating: 0xAB, count: 64)
            bytes.append(contentsOf: fixture.blob)
            guard let buffer = ctx.device.makeBuffer(
                    bytes: bytes, length: bytes.count, options: .storageModeShared) else {
                Issue.record("expert blob allocation failed")
                throw CocoaError(.fileReadUnknown)
            }
            views.append(TensorView(buffer: buffer,
                                    offset: 64,
                                    length: UInt64(fixture.blob.count),
                                    scaleOffset: 0, scaleLength: 0,
                                    biasOffset: 0, biasLength: 0,
                                    shape: (0, UInt32(expertID), 0, 0),
                                    dtype: 0))
        }

        let t = hidden.count
        guard let pairBuffer = ctx.device.makeBuffer(
                bytes: routes.sortedPairs,
                length: routes.sortedPairs.count * MemoryLayout<PrefillTokenExpertPair>.stride,
                options: .storageModeShared),
              let hiddenBuffer = Fp16Buffer.make(ctx.device, values: hidden.flatMap { $0 }),
              let partialsBuffer = ctx.device.makeBuffer(
                length: t * Self.topK * d * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let actsBuffer = Fp16Buffer.make(ctx.device,
                                               count: pairMicrobatchRows * f),
              let h2Buffer = Fp16Buffer.make(ctx.device, count: t * d),
              let residualBuffer = Fp16Buffer.make(
                ctx.device,
                values: (residual ?? Array(
                    repeating: Array(repeating: 0, count: d), count: t)).flatMap { $0 }),
              let weightsBuffer = ctx.device.makeBuffer(
                bytes: weights,
                length: weights.count * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            Issue.record("buffer allocation failed")
            throw CocoaError(.fileReadUnknown)
        }

        var microbatches: [Int] = []
        for (tileIndex, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                forTile: tileIndex, routes: routes)
            let binding = try PrefillStreamedTileBinding(
                expertIDs: expertIDs, views: expertIDs.map { views[$0] })
            try binding.validateCoversPairs(routes.sortedPairs,
                                            pairStart: Int(tile.pairStart),
                                            pairCount: Int(tile.pairCount))
            let argumentBuffer = try driver.makeStreamedArgumentBuffer(binding: binding)
            let params = PrefillGroupedRoutedMoEV4StreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(d),
                routedIntermediate: UInt32(f),
                topK: UInt32(Self.topK),
                hiddenStrideElements: UInt32(d),
                binding: binding,
                offsets: pool[0].offsets)
            guard let cmd = ctx.queue.makeCommandBuffer() else {
                Issue.record("command buffer creation failed")
                throw CocoaError(.fileReadUnknown)
            }
            let count = driver.encodeStreamedTile(
                commandBuffer: cmd,
                hidden: hiddenBuffer,
                sortedPairs: pairBuffer,
                acts: actsBuffer,
                routePartials: partialsBuffer,
                routeWeights: weightsBuffer,
                argumentBuffer: argumentBuffer,
                binding: binding,
                params: params,
                pairMicrobatchRows: pairMicrobatchRows)
            microbatches.append(count)
            cmd.commit()
            withExtendedLifetime(argumentBuffer) {
                cmd.waitUntilCompleted()
            }
            #expect(cmd.error == nil)
        }

        guard let reduceCmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("command buffer creation failed")
            throw CocoaError(.fileReadUnknown)
        }
        driver.encodeReduceTokenMajor(commandBuffer: reduceCmd,
                                      routePartials: partialsBuffer,
                                      residual: residualBuffer,
                                      h2: h2Buffer,
                                      queryCount: UInt32(t),
                                      topK: UInt32(Self.topK),
                                      d: UInt32(d))
        reduceCmd.commit()
        reduceCmd.waitUntilCompleted()
        #expect(reduceCmd.error == nil)
        return (Fp16Buffer.read(h2Buffer, count: t * d), routes.tiles.count, microbatches)
    }

    private static func makeHidden(seed: UInt64,
                                   tokens: Int,
                                   d: Int) -> [[Float]] {
        var rng = SeedTree(seed).key("hidden")
        return (0..<tokens).map { _ in
            (0..<d).map { _ in Float(Float16(rng.uniform(-0.5, 0.5))) }
        }
    }

    // MARK: - Tests

    /// Degenerate grouping: one token whose six ranks all land on a single
    /// expert (one group, one tile, one live expert in the binding).
    @Test func singleExpertSingleTokenMatchesReference() throws {
        let d = Self.dimension
        let f = Self.intermediate
        let pool = Self.makePool(numExperts: 8, seed: 0x61, d: d, f: f)
        let assignments = [[3, 3, 3, 3, 3, 3]]
        let pairs = Self.makePairs(assignments: assignments)
        let hidden = Self.makeHidden(seed: 0x62, tokens: 1, d: d)
        let weights = Self.makeWeights(seed: 0x63, tokens: 1)

        let result = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                  weights: weights, numExperts: 8, d: d, f: f)
        #expect(result.tiles == 1)
        #expect(result.microbatches == [1])

        let expected = Self.reference(pairs: pairs, experts: pool,
                                      hidden: hidden, weights: weights,
                                      d: d, f: f)
        let rel = RelError.compute(actual: result.h2, reference: expected)
        #expect(rel < Tolerance.fp16ChainedReduction, "single expert rel=\(rel)")
    }

    /// The production runner supplies the shared expert as the phase-2
    /// residual. It must join the F32 routed sum before the one final half
    /// store, matching the decode kernel rather than a separate fp16 add.
    @Test func sharedResidualIsFusedBeforeFinalHalfStore() throws {
        let d = Self.dimension
        let f = Self.intermediate
        let pool = Self.makePool(numExperts: 8, seed: 0x631, d: d, f: f)
        let pairs = Self.makePairs(assignments: [[1, 2, 3, 4, 5, 6]])
        let hidden = Self.makeHidden(seed: 0x632, tokens: 1, d: d)
        let weights = Self.makeWeights(seed: 0x633, tokens: 1)
        let residual = [(0..<d).map { index in
            Float(Float16(Float((index % 13) - 6) * 0.03125))
        }]

        let result = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                  weights: weights, residual: residual,
                                  numExperts: 8, d: d, f: f)
        let expected = Self.reference(pairs: pairs, experts: pool,
                                      hidden: hidden, weights: weights,
                                      residual: residual, d: d, f: f)
        let rel = RelError.compute(actual: result.h2, reference: expected)
        #expect(rel < Tolerance.fp16ChainedReduction, "fused residual rel=\(rel)")
    }

    /// Five tokens over 12 live experts with uneven per-expert pair counts
    /// (2...3 pairs per expert). Experts 12...15 of the pool receive zero
    /// pairs: they must not appear in groups, tiles, or bindings.
    @Test func unevenGroupingWithEmptyExpertsMatchesReference() throws {
        let d = Self.dimension
        let f = Self.intermediate
        let numExperts = 16
        let pool = Self.makePool(numExperts: numExperts, seed: 0x64, d: d, f: f)
        // Six consecutive distinct experts per token, staggered start.
        let assignments = (0..<5).map { t in
            let base = (t * 5) % 12
            return (0..<Self.topK).map { (base + $0) % 12 }
        }
        let pairs = Self.makePairs(assignments: assignments)
        let hidden = Self.makeHidden(seed: 0x65, tokens: 5, d: d)
        let weights = Self.makeWeights(seed: 0x66, tokens: 5)

        let routes = try PrefillGroupedRoutedMoEV4.groupedRoutes(
            pairs: pairs, queryCount: 5, numExperts: numExperts)
        #expect(routes.groups.count == 12)
        #expect(routes.groups.allSatisfy { $0.expert < 12 })
        #expect(routes.perExpertCounts[12...].allSatisfy { $0 == 0 })
        #expect(Set(routes.perExpertCounts.prefix(12)).isSubset(of: [2, 3]))

        let result = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                  weights: weights, numExperts: numExperts,
                                  d: d, f: f)
        #expect(result.tiles == 2)   // 12 live experts -> tiles of 8 + 4
        let expected = Self.reference(pairs: pairs, experts: pool,
                                      hidden: hidden, weights: weights,
                                      d: d, f: f)
        let rel = RelError.compute(actual: result.h2, reference: expected)
        #expect(rel < Tolerance.fp16ChainedReduction, "uneven grouping rel=\(rel)")
    }

    /// Nine live experts force a tile boundary at 8: the second tile holds a
    /// single expert and its pairs must still scatter into the shared
    /// token-major partials correctly.
    @Test func tileBoundaryWithNineExpertsMatchesReference() throws {
        let d = Self.dimension
        let f = Self.intermediate
        let numExperts = 9
        let pool = Self.makePool(numExperts: numExperts, seed: 0x67, d: d, f: f)
        let assignments = (0..<4).map { t in
            (0..<Self.topK).map { (t * 2 + $0) % numExperts }
        }
        let pairs = Self.makePairs(assignments: assignments)
        let hidden = Self.makeHidden(seed: 0x68, tokens: 4, d: d)
        let weights = Self.makeWeights(seed: 0x69, tokens: 4)

        let routes = try PrefillGroupedRoutedMoEV4.groupedRoutes(
            pairs: pairs, queryCount: 4, numExperts: numExperts)
        #expect(routes.tiles.count == 2)
        #expect(routes.tiles[0].groupCount == 8)
        #expect(routes.tiles[1].groupCount == 1)
        #expect(routes.tiles[0].pairCount + routes.tiles[1].pairCount
                == UInt32(4 * Self.topK))

        let result = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                  weights: weights, numExperts: numExperts,
                                  d: d, f: f)
        #expect(result.tiles == 2)
        let expected = Self.reference(pairs: pairs, experts: pool,
                                      hidden: hidden, weights: weights,
                                      d: d, f: f)
        let rel = RelError.compute(actual: result.h2, reference: expected)
        #expect(rel < Tolerance.fp16ChainedReduction, "tile boundary rel=\(rel)")
    }

    /// Clamp-exercising case: expert 0's gate/up rows drive pre-activations
    /// past ±10 (both up-clamp signs, the gate upper clamp, and deeply
    /// negative gates that must NOT be clamped below).
    @Test func swigluClampCasesMatchReference() throws {
        let d = Self.dimension
        let f = Self.intermediate
        let numExperts = 8
        var pool = Self.makePool(numExperts: numExperts, seed: 0x6A, d: d, f: f)
        let hidden = Self.makeHidden(seed: 0x6B, tokens: 2, d: d)
        let x0 = hidden[0]

        // Rows proportional to sign(x0) make every dot ±0.5·Σ|x0| ≈ ±30,
        // deterministic in sign: gates clamp at +10, up rows alternate both
        // clamp signs, and a second gate set goes deeply negative (silu → 0).
        let s = x0.map { $0 >= 0 ? Float(0.5) : Float(-0.5) }
        var gates = (0..<f).map { _ in s }
        let ups = (0..<f).map { row in row.isMultiple(of: 2) ? s : s.map { -$0 } }
        // Odd rows: deeply negative gate, positive-clamped up.
        for row in 0..<f where !row.isMultiple(of: 2) {
            gates[row] = s.map { -$0 }
        }
        let qGates = gates.map { V4Quantization.quantizeFP4Row($0) }
        let qUps = ups.map { V4Quantization.quantizeFP4Row($0) }
        let (bytes, offsets) = Self.makeBlob(gate: qGates,
                                             up: qUps,
                                             down: pool[0].downs)
        pool[0] = ExpertFixture(gates: qGates, ups: qUps,
                                downs: pool[0].downs, blob: bytes, offsets: offsets)

        // Prove the fixture engages the clamps, otherwise the test is vacuous.
        let gRow = V4Quantization.dequantizeFP4Row(pool[0].gates[0], n: d)
        let gNegRow = V4Quantization.dequantizeFP4Row(pool[0].gates[1], n: d)
        let uRow = V4Quantization.dequantizeFP4Row(pool[0].ups[0], n: d)
        #expect(Self.dot(gRow, x0) > 10)       // gate upper clamp
        #expect(Self.dot(gNegRow, x0) < -10)   // gate: NO lower clamp
        #expect(abs(Self.dot(uRow, x0)) > 10)  // up clamp

        // Both tokens route rank 0 to expert 0; other ranks spread out
        // (distinct experts per token).
        let assignments = (0..<2).map { t in
            [0] + (1..<Self.topK).map { ((t * 3 + $0) % 7) + 1 }
        }
        let pairs = Self.makePairs(assignments: assignments)
        let weights = Self.makeWeights(seed: 0x6C, tokens: 2)

        let result = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                  weights: weights, numExperts: numExperts,
                                  d: d, f: f)
        let expected = Self.reference(pairs: pairs, experts: pool,
                                      hidden: hidden, weights: weights,
                                      d: d, f: f)
        let rel = RelError.compute(actual: result.h2, reference: expected)
        #expect(rel < Tolerance.fp16ChainedReduction, "clamp pipeline rel=\(rel)")
    }

    /// Microbatch splitting must be bit-exact: the same tile encoded as one
    /// microbatch of 32 pair-rows vs several of 7 produces identical h2.
    @Test func microbatchSplittingIsBitExact() throws {
        let d = Self.dimension
        let f = Self.intermediate
        let numExperts = 9
        let pool = Self.makePool(numExperts: numExperts, seed: 0x6D, d: d, f: f)
        let assignments = (0..<4).map { t in
            (0..<Self.topK).map { (t * 2 + $0) % numExperts }
        }
        let pairs = Self.makePairs(assignments: assignments)
        let hidden = Self.makeHidden(seed: 0x6E, tokens: 4, d: d)
        let weights = Self.makeWeights(seed: 0x6F, tokens: 4)

        let whole = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                 weights: weights, numExperts: numExperts,
                                 d: d, f: f, pairMicrobatchRows: 32)
        let split = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                 weights: weights, numExperts: numExperts,
                                 d: d, f: f, pairMicrobatchRows: 7)
        #expect(split.microbatches.contains { $0 > 1 })
        #expect(split.h2 == whole.h2)
    }

    /// Real V4-Flash geometry (D = 4096, F = 2048) with one token over two
    /// experts; guards the fixed-shape assumptions of the kernels (D walks 16
    /// full 8-group blocks, F walks 8). Random-code payloads keep fixture
    /// generation cheap at this size.
    @Test func fullGeometrySingleTokenMatchesReference() throws {
        let d = 4096
        let f = 2048
        let pool = [Self.makeRandomCodeExpert(seed: 0x70, label: "expert-0", d: d, f: f),
                    Self.makeRandomCodeExpert(seed: 0x70, label: "expert-1", d: d, f: f)]
        let assignments = [[0, 1, 0, 1, 0, 1]]
        let pairs = Self.makePairs(assignments: assignments)
        let hidden = Self.makeHidden(seed: 0x71, tokens: 1, d: d)
        let weights = Self.makeWeights(seed: 0x72, tokens: 1)

        let result = try Self.run(pool: pool, pairs: pairs, hidden: hidden,
                                  weights: weights, numExperts: 2,
                                  d: d, f: f)
        #expect(result.tiles == 1)
        let expected = Self.reference(pairs: pairs, experts: pool,
                                      hidden: hidden, weights: weights,
                                      d: d, f: f)
        let rel = RelError.compute(actual: result.h2, reference: expected)
        #expect(rel < Tolerance.fp16ChainedReduction, "full geometry rel=\(rel)")
    }

    /// The V4 tile scheduler budget: prefetch depth 1 with 8-expert tiles
    /// must fit the 16-slot per-layer expert cache exactly (current tile +
    /// next tile being fetched).
    @Test func tileSchedulerConfigFitsSixteenSlotCache() {
        let config = PrefillGroupedRoutedMoEV4.tileSchedulerConfig
        #expect(config.tileExperts == 8)
        #expect(config.maxPendingDepth == 1)
        #expect(config.fitsSlotBudget(slotCount: 16))
        #expect(!config.fitsSlotBudget(slotCount: 15))
    }
}
