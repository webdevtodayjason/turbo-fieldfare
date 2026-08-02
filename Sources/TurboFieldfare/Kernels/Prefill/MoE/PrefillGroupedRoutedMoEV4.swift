import Foundation
import Metal

/// Buffer indices for the `prefill_moe_v4` grouped kernels. Kept private to
/// this driver; the Metal side uses the same literal indices.
private enum PrefillMoEV4BufferIndex {
    // Phase 1: grouped FP4 gate/up GEMM + clamped SwiGLU.
    enum Phase1 {
        static let hidden = 0
        static let sortedPairs = 1
        static let acts = 2
        static let routedBlobs = 3
        static let params = 4
    }
    // Phase 2: grouped FP4 down projection scattered to route partials.
    enum Down {
        static let sortedPairs = 0
        static let routePartials = 1
        static let acts = 2
        static let routedBlobs = 3
        static let params = 4
    }
    // Token-major F32-weighted reduce.
    enum Reduce {
        static let routePartials = 0
        static let routeWeights = 1
        static let h2 = 2
        static let t = 3
        static let topK = 4
        static let d = 5
    }
}

/// Per-tile parameters for the V4 grouped prefill MoE kernels. Mirrors the
/// Gemma `PrefillGroupedRoutedMoEStreamedParams` layout but with 8 local
/// expert slots and the 6-field V4 blob offsets (no biases).
struct PrefillGroupedRoutedMoEV4StreamedParams: Equatable, Sendable {
    var pairStart: UInt32
    var pairCount: UInt32
    var d: UInt32
    var routedIntermediate: UInt32
    var topK: UInt32
    var hiddenStrideElements: UInt32
    var liveExpertCount: UInt32
    var localExpert0: UInt32
    var localExpert1: UInt32
    var localExpert2: UInt32
    var localExpert3: UInt32
    var localExpert4: UInt32
    var localExpert5: UInt32
    var localExpert6: UInt32
    var localExpert7: UInt32
    var gateWOff: UInt32
    var gateSOff: UInt32
    var upWOff: UInt32
    var upSOff: UInt32
    var downWOff: UInt32
    var downSOff: UInt32

    init(pairStart: UInt32,
         pairCount: UInt32,
         d: UInt32,
         routedIntermediate: UInt32,
         topK: UInt32,
         hiddenStrideElements: UInt32,
         binding: PrefillStreamedTileBinding,
         offsets: V4ExpertOffsets) {
        precondition(binding.expertIDs.count <= PrefillGroupedRoutedMoEV4.maxTileExperts,
                     "V4 tile binding has \(binding.expertIDs.count) experts; maximum is \(PrefillGroupedRoutedMoEV4.maxTileExperts)")
        var ids = Array(repeating: UInt32.max, count: PrefillGroupedRoutedMoEV4.maxTileExperts)
        for (index, expert) in binding.expertIDs.enumerated() {
            ids[index] = UInt32(expert)
        }
        self.pairStart = pairStart
        self.pairCount = pairCount
        self.d = d
        self.routedIntermediate = routedIntermediate
        self.topK = topK
        self.hiddenStrideElements = hiddenStrideElements
        self.liveExpertCount = UInt32(binding.expertIDs.count)
        self.localExpert0 = ids[0]
        self.localExpert1 = ids[1]
        self.localExpert2 = ids[2]
        self.localExpert3 = ids[3]
        self.localExpert4 = ids[4]
        self.localExpert5 = ids[5]
        self.localExpert6 = ids[6]
        self.localExpert7 = ids[7]
        self.gateWOff = offsets.gateWOff
        self.gateSOff = offsets.gateSOff
        self.upWOff = offsets.upWOff
        self.upSOff = offsets.upSOff
        self.downWOff = offsets.downWOff
        self.downSOff = offsets.downSOff
    }
}

struct PrefillMoEV4StreamedTileArgumentBuffer {
    let buffer: MTLBuffer
}

/// Grouped routed-MoE prefill driver for DeepSeek V4-Flash (V4F-06b).
///
/// Structural mirror of `PrefillGroupedRoutedMoE` (Gemma) over the V4 FP4
/// e2m1 + ue8m0 expert format with clamped SwiGLU and F32 routing weights.
/// Differences from the Gemma path:
///
///   * Tiles hold at most `maxTileExperts = 8` experts so the 16-slot
///     per-layer expert cache can carry the in-flight tile plus the next
///     tile being fetched (`tileSchedulerConfig` encodes this budget).
///   * Phase 1 and the down projection run one SIMD group per (pair, row)
///     with the decode kernel's 8-group block structure, instead of one
///     thread per output element.
///   * The down scatter is unweighted; routing weights are applied in
///     `encodeReduceTokenMajor` from an F32 `[T * topK]` buffer (the V4
///     router emits F32 weights, so no half round-trip).
///
/// Grouping itself is format-agnostic and shared: call
/// `PrefillMoEGrouping.groupTokenExpertPairs(..., tileExpertCount: 8)` (or
/// the `groupedRoutes` convenience below). Tile fetching, slot lifetime, and
/// the prefetch scheduler reuse `PrefillStreamedTileBinding`,
/// `PrefillStreamedTileSlotLifetime`, and `PrefillRoutedTileScheduler`
/// unchanged; the runner passes `avoidingSlots` from the in-flight tile's
/// planned slots so a prefetched tile never reuses a slot that queued GPU
/// work still references.
final class PrefillGroupedRoutedMoEV4 {
    /// Experts per streamed tile. The per-layer cache has 16 slots, so one
    /// in-flight tile (8) plus one prefetched tile (8) always fits.
    static let maxTileExperts = 8

    /// V4-Flash routing geometry.
    static let topK = 6

    /// Tile scheduler config for the V4 geometry: prefetch depth 1 with
    /// 8-expert tiles against the 16-slot cache.
    static let tileSchedulerConfig = PrefillRoutedTileSchedulerConfig(
        maxPendingDepth: 1,
        tileExperts: maxTileExperts)

    private let device: MTLDevice
    private let phase1PSO: MTLComputePipelineState
    private let downPSO: MTLComputePipelineState
    private let reducePSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder

    init(context: MetalContext) throws {
        self.device = context.device
        self.phase1PSO = try context.pipeline("prefill_moe_v4_grouped_phase1_gate_up_swiglu")
        self.downPSO = try context.pipeline("prefill_moe_v4_grouped_down_scatter")
        self.reducePSO = try context.pipeline("prefill_moe_v4_reduce_token_major")
        guard let phase1Function = context.library.makeFunction(
                name: "prefill_moe_v4_grouped_phase1_gate_up_swiglu") else {
            throw MetalError.missingFunction("prefill_moe_v4_grouped_phase1_gate_up_swiglu")
        }
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(
            bufferIndex: PrefillMoEV4BufferIndex.Phase1.routedBlobs)
    }

    /// Group token/expert pairs into expert-sorted runs and 8-expert tiles.
    /// Thin typed wrapper over the shared grouping helper so callers cannot
    /// accidentally pick the Gemma 16-expert tile width.
    static func groupedRoutes(pairs: [PrefillTokenExpertPair],
                              queryCount: Int,
                              numExperts: Int) throws -> PrefillMoEGroupedRoutes {
        try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: queryCount,
            topK: topK,
            numExperts: numExperts,
            tileExpertCount: maxTileExperts)
    }

    /// Argument buffer binding the tile's expert blob views. One per tile;
    /// the runner keeps it alive until the tile's command buffer completes
    /// (slot-ownership invariant).
    func makeStreamedArgumentBuffer(binding: PrefillStreamedTileBinding) throws
        -> PrefillMoEV4StreamedTileArgumentBuffer {
        precondition(binding.views.count <= Self.maxTileExperts)
        guard let buffer = device.makeBuffer(length: routedArgEncoder.encodedLength,
                                             options: .storageModeShared) else {
            throw PrefillGroupedRoutedMoEError.allocationFailed(
                "prefill V4 streamed expert argument buffer")
        }
        buffer.label = "prefill.groupedMoeV4.streamedArgumentBuffer"
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for index in binding.views.indices {
            let view = binding.views[index]
            routedArgEncoder.setBuffer(view.buffer, offset: Int(view.offset), index: index)
        }
        return PrefillMoEV4StreamedTileArgumentBuffer(buffer: buffer)
    }

    private func validate(params: PrefillGroupedRoutedMoEV4StreamedParams,
                          binding: PrefillStreamedTileBinding) {
        precondition(params.d.isMultiple(of: 32), "D must be a multiple of 32")
        precondition(params.routedIntermediate.isMultiple(of: 32),
                     "F must be a multiple of 32")
        precondition(params.topK == UInt32(Self.topK), "topK must be \(Self.topK)")
        precondition(params.liveExpertCount == UInt32(binding.views.count))
        precondition(params.liveExpertCount >= 1 &&
                     params.liveExpertCount <= UInt32(Self.maxTileExperts))
        precondition(params.hiddenStrideElements >= params.d,
                     "hidden stride is too small")
        for offset in [params.gateWOff, params.gateSOff,
                       params.upWOff, params.upSOff,
                       params.downWOff, params.downSOff] {
            precondition(offset.isMultiple(of: 4),
                         "V4 expert blob sub-tensor offset \(offset) is not 4-aligned")
        }
    }

    /// Encode one tile: phase 1 (FP4 gate/up + clamped SwiGLU over gathered
    /// token rows) followed by the FP4 down projection scattered into
    /// token-major route partials. Returns the number of microbatches.
    ///
    /// `acts` scratch is `[pairMicrobatchRows * F]` half; `routePartials` is
    /// `[queryCount * topK * D]` half shared across all tiles of the layer.
    @discardableResult
    func encodeStreamedTile(commandBuffer: MTLCommandBuffer,
                            hidden: MTLBuffer,
                            hiddenOffset: Int = 0,
                            sortedPairs: MTLBuffer,
                            sortedPairsOffset: Int = 0,
                            acts: MTLBuffer,
                            actsOffset: Int = 0,
                            routePartials: MTLBuffer,
                            routePartialsOffset: Int = 0,
                            argumentBuffer: PrefillMoEV4StreamedTileArgumentBuffer,
                            binding: PrefillStreamedTileBinding,
                            params: PrefillGroupedRoutedMoEV4StreamedParams,
                            pairMicrobatchRows: Int = 32) -> Int {
        guard params.pairCount > 0, pairMicrobatchRows > 0 else { return 0 }
        validate(params: params, binding: binding)
        var consumed: UInt32 = 0
        var microbatchCount = 0
        while consumed < params.pairCount {
            var p = params
            p.pairStart = params.pairStart + consumed
            p.pairCount = min(UInt32(pairMicrobatchRows), params.pairCount - consumed)

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(phase1PSO)
                enc.setBuffer(hidden, offset: hiddenOffset,
                              index: PrefillMoEV4BufferIndex.Phase1.hidden)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset,
                              index: PrefillMoEV4BufferIndex.Phase1.sortedPairs)
                enc.setBuffer(acts, offset: actsOffset,
                              index: PrefillMoEV4BufferIndex.Phase1.acts)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillMoEV4BufferIndex.Phase1.routedBlobs)
                enc.setBytes(&p,
                             length: MemoryLayout<PrefillGroupedRoutedMoEV4StreamedParams>.stride,
                             index: PrefillMoEV4BufferIndex.Phase1.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                let rows = Int(p.pairCount) * Int(p.routedIntermediate)
                enc.dispatchThreadgroups(
                    MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }

            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.setComputePipelineState(downPSO)
                enc.setBuffer(sortedPairs, offset: sortedPairsOffset,
                              index: PrefillMoEV4BufferIndex.Down.sortedPairs)
                enc.setBuffer(routePartials, offset: routePartialsOffset,
                              index: PrefillMoEV4BufferIndex.Down.routePartials)
                enc.setBuffer(acts, offset: actsOffset,
                              index: PrefillMoEV4BufferIndex.Down.acts)
                enc.setBuffer(argumentBuffer.buffer, offset: 0,
                              index: PrefillMoEV4BufferIndex.Down.routedBlobs)
                enc.setBytes(&p,
                             length: MemoryLayout<PrefillGroupedRoutedMoEV4StreamedParams>.stride,
                             index: PrefillMoEV4BufferIndex.Down.params)
                for view in binding.views {
                    enc.useResource(view.buffer, usage: .read)
                }
                let rows = Int(p.pairCount) * Int(p.d)
                enc.dispatchThreadgroups(
                    MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
            }

            consumed += p.pairCount
            microbatchCount += 1
        }
        return microbatchCount
    }

    /// Token-major routing-weighted reduce over all tiles' partials. Encode
    /// once per layer after every tile command buffer has drained.
    /// `routeWeights` is the router's F32 `[queryCount * topK]` output;
    /// `h2` receives `[queryCount * D]` half.
    func encodeReduceTokenMajor(commandBuffer: MTLCommandBuffer,
                                routePartials: MTLBuffer,
                                routePartialsOffset: Int = 0,
                                routeWeights: MTLBuffer,
                                routeWeightsOffset: Int = 0,
                                h2: MTLBuffer,
                                h2Offset: Int = 0,
                                queryCount: UInt32,
                                topK: UInt32,
                                d: UInt32) {
        precondition(queryCount > 0, "queryCount must be positive")
        precondition(topK == UInt32(Self.topK), "topK must be \(Self.topK)")
        precondition(d > 0, "D must be positive")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(reducePSO)
        enc.setBuffer(routePartials, offset: routePartialsOffset,
                      index: PrefillMoEV4BufferIndex.Reduce.routePartials)
        enc.setBuffer(routeWeights, offset: routeWeightsOffset,
                      index: PrefillMoEV4BufferIndex.Reduce.routeWeights)
        enc.setBuffer(h2, offset: h2Offset, index: PrefillMoEV4BufferIndex.Reduce.h2)
        var tVar = queryCount
        var topKVar = topK
        var dVar = d
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.stride,
                     index: PrefillMoEV4BufferIndex.Reduce.t)
        enc.setBytes(&topKVar, length: MemoryLayout<UInt32>.stride,
                     index: PrefillMoEV4BufferIndex.Reduce.topK)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.stride,
                     index: PrefillMoEV4BufferIndex.Reduce.d)
        enc.dispatchThreads(MTLSize(width: Int(d), height: Int(queryCount), depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }
}
