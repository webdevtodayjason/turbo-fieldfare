import Foundation
import Metal

public struct V4GroupedRoutedMoEPrefillOutput {
    public let routes: PrefillMoEGroupedRoutes
    public let sortedPairs: MTLBuffer
    public let routePartials: MTLBuffer
    public let output: MTLBuffer
}

struct V4GroupedRoutedMoEPrefillTileRecord: Equatable, Sendable {
    var tileIndex: Int
    var expertIDs: [Int]
    var plannedSlots: [Int]
    var avoidingSlotsForNextPlan: [Int]
    var prefetchedNextTile: Int?
}

struct V4GroupedRoutedMoEPrefillPlan: Equatable, Sendable {
    var routes: PrefillMoEGroupedRoutes
    var records: [V4GroupedRoutedMoEPrefillTileRecord]
}

protocol V4GroupedRoutedMoEPrefillExpertProvider {
    func routedExpertV4Offsets(layer: Int) -> V4ExpertOffsets
    func planRoutedExperts(layer: Int, experts: [Int], avoidingSlots: Set<Int>) throws -> RoutedExpertFetchPlan
    func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView]
}

extension V4Model: V4GroupedRoutedMoEPrefillExpertProvider {}

private final class V4GroupedRoutedMoEPrefillProviderBox: @unchecked Sendable {
    let provider: V4GroupedRoutedMoEPrefillExpertProvider
    init(_ provider: V4GroupedRoutedMoEPrefillExpertProvider) { self.provider = provider }
    func fetch(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try await provider.fetchRoutedExperts(plan: plan)
    }
}

enum V4GroupedRoutedMoEPrefillAdapterError: Error, Equatable, CustomStringConvertible {
    case allocationFailed(String)
    case invalidRouteBuffers(ids: Int, weights: Int, queryCount: Int, topK: Int)
    case commandBufferUnavailable
    case preplannedFetchMismatch(tile: Int)

    var description: String {
        switch self {
        case .allocationFailed(let label): return "failed to allocate \(label)"
        case .invalidRouteBuffers(let ids, let weights, let queryCount, let topK):
            return "expected \(queryCount * topK) token-major route ids/weights, got ids=\(ids), weights=\(weights)"
        case .commandBufferUnavailable: return "failed to create command buffer"
        case .preplannedFetchMismatch(let tile): return "preplanned fetch does not match tile \(tile)"
        }
    }
}

/// Reusable orchestration adapter for V4 grouped routed-MoE prefill.
///
/// This host-side adapter normalizes hash-routed and router-produced
/// token-major expert IDs to the same pair stream, groups into <=8-expert
/// tiles, uses the shared prefetch scheduler, protects slot lifetime with
/// `avoidingSlots`, encodes phase1/down per tile, then reduces token-major.
final class V4GroupedRoutedMoEPrefillAdapter {
    private let context: MetalContext
    private let moe: PrefillGroupedRoutedMoEV4
    private let scheduler = PrefillRoutedTileScheduler(config: PrefillGroupedRoutedMoEV4.tileSchedulerConfig)

    init(context: MetalContext, moe: PrefillGroupedRoutedMoEV4? = nil) throws {
        self.context = context
        self.moe = try moe ?? PrefillGroupedRoutedMoEV4(context: context)
    }

    static func makePairs(tokenMajorExpertIDs: [UInt32], tokenMajorWeights: [Float], queryCount: Int,
                          topK: Int = PrefillGroupedRoutedMoEV4.topK) throws -> [PrefillTokenExpertPair] {
        guard tokenMajorExpertIDs.count == queryCount * topK, tokenMajorWeights.count == queryCount * topK else {
            throw V4GroupedRoutedMoEPrefillAdapterError.invalidRouteBuffers(
                ids: tokenMajorExpertIDs.count, weights: tokenMajorWeights.count, queryCount: queryCount, topK: topK)
        }
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(tokenMajorExpertIDs.count)
        for token in 0..<queryCount {
            for rank in 0..<topK {
                let index = token * topK + rank
                pairs.append(PrefillTokenExpertPair(token: UInt32(token), expert: tokenMajorExpertIDs[index],
                                                    rank: UInt32(rank), weight: Float16(tokenMajorWeights[index])))
            }
        }
        return pairs
    }

    func encode(model: V4Model, hidden: MTLBuffer, hiddenOffset: Int = 0,
                tokenMajorExpertIDs: [UInt32], tokenMajorWeights: [Float],
                residual: MTLBuffer? = nil, residualOffset: Int = 0,
                layer: Int, chunkSize queryCount: Int, d: Int, routedIntermediate: Int,
                hiddenStrideElements: Int? = nil, pairMicrobatchRows: Int = 32) async throws
        -> V4GroupedRoutedMoEPrefillOutput {
        let pairs = try Self.makePairs(tokenMajorExpertIDs: tokenMajorExpertIDs,
                                       tokenMajorWeights: tokenMajorWeights,
                                       queryCount: queryCount)
        guard let routeWeights = makeBuffer(bytes: tokenMajorWeights, label: "prefill.groupedMoeV4.routeWeights") else {
            throw V4GroupedRoutedMoEPrefillAdapterError.allocationFailed("prefill V4 route weights")
        }
        return try await encode(provider: model, hidden: hidden, hiddenOffset: hiddenOffset, pairs: pairs,
                                routeWeights: routeWeights, routeWeightsOffset: 0,
                                residual: residual, residualOffset: residualOffset,
                                layer: layer, queryCount: queryCount, numExperts: model.config.numExperts,
                                d: d, routedIntermediate: routedIntermediate,
                                hiddenStrideElements: hiddenStrideElements ?? d,
                                pairMicrobatchRows: pairMicrobatchRows)
    }

    func encode(provider: V4GroupedRoutedMoEPrefillExpertProvider, hidden: MTLBuffer, hiddenOffset: Int = 0,
                pairs: [PrefillTokenExpertPair], routeWeights: MTLBuffer, routeWeightsOffset: Int = 0,
                residual: MTLBuffer? = nil, residualOffset: Int = 0,
                layer: Int, queryCount: Int, numExperts: Int, d: Int, routedIntermediate: Int,
                hiddenStrideElements: Int, pairMicrobatchRows: Int = 32) async throws
        -> V4GroupedRoutedMoEPrefillOutput {
        let routes = try PrefillGroupedRoutedMoEV4.groupedRoutes(pairs: pairs, queryCount: queryCount, numExperts: numExperts)
        guard let sortedPairs = makeBuffer(bytes: routes.sortedPairs, label: "prefill.groupedMoeV4.sortedPairs") else {
            throw V4GroupedRoutedMoEPrefillAdapterError.allocationFailed("prefill V4 sorted route pairs")
        }
        let routePartials = try makeEmptyBuffer(length: queryCount * PrefillGroupedRoutedMoEV4.topK * d * MemoryLayout<Float>.stride,
                                                label: "prefill.groupedMoeV4.routePartials")
        let output = try makeEmptyBuffer(length: queryCount * d * MemoryLayout<Float16>.stride,
                                         label: "prefill.groupedMoeV4.output")
        let reductionResidual = try residual ?? makeEmptyBuffer(
            length: queryCount * d * MemoryLayout<Float16>.stride,
            label: "prefill.groupedMoeV4.zeroResidual")
        let acts = try makeEmptyBuffer(length: max(1, min(pairMicrobatchRows, max(1, routes.maxPairsPerTile))) * routedIntermediate * MemoryLayout<Float16>.stride,
                                       label: "prefill.groupedMoeV4.actsScratch")
        try await encodeTiles(provider: provider, routes: routes, hidden: hidden, hiddenOffset: hiddenOffset,
                              sortedPairs: sortedPairs, routePartials: routePartials, acts: acts,
                              output: output, routeWeights: routeWeights, routeWeightsOffset: routeWeightsOffset,
                              residual: reductionResidual, residualOffset: residualOffset,
                              layer: layer, queryCount: queryCount, d: d, routedIntermediate: routedIntermediate,
                              hiddenStrideElements: hiddenStrideElements, pairMicrobatchRows: pairMicrobatchRows)
        return V4GroupedRoutedMoEPrefillOutput(routes: routes, sortedPairs: sortedPairs,
                                               routePartials: routePartials, output: output)
    }

    private func encodeTiles(provider: V4GroupedRoutedMoEPrefillExpertProvider, routes: PrefillMoEGroupedRoutes,
                             hidden: MTLBuffer, hiddenOffset: Int, sortedPairs: MTLBuffer,
                             routePartials: MTLBuffer, acts: MTLBuffer, output: MTLBuffer,
                             routeWeights: MTLBuffer, routeWeightsOffset: Int,
                             residual: MTLBuffer, residualOffset: Int,
                             layer: Int, queryCount: Int,
                             d: Int, routedIntermediate: Int, hiddenStrideElements: Int,
                             pairMicrobatchRows: Int) async throws {
        let offsets = provider.routedExpertV4Offsets(layer: layer)
        let fetchProvider = V4GroupedRoutedMoEPrefillProviderBox(provider)
        var lifetime = PrefillStreamedTileSlotLifetime()
        var pending: (tileIndex: Int, plan: RoutedExpertFetchPlan, task: Task<[TensorView], Error>)?

        func planForTile(_ tileIndex: Int, avoidingSlots: Set<Int>) throws -> RoutedExpertFetchPlan {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
            return try provider.planRoutedExperts(layer: layer, experts: expertIDs, avoidingSlots: avoidingSlots)
        }

        if !routes.tiles.isEmpty {
            let firstPlan = try planForTile(0, avoidingSlots: [])
            pending = (0, firstPlan, Task { try await fetchProvider.fetch(plan: firstPlan) })
        }

        for tileIndex in routes.tiles.indices {
            guard let current = pending, current.tileIndex == tileIndex else {
                throw V4GroupedRoutedMoEPrefillAdapterError.preplannedFetchMismatch(tile: tileIndex)
            }
            let plannedSlots = current.plan.assignedSlots
            try lifetime.begin(tileIndex: tileIndex, plannedSlots: plannedSlots)

            var nextPending: (tileIndex: Int, plan: RoutedExpertFetchPlan, task: Task<[TensorView], Error>)?
            let nextIndex = tileIndex + 1
            if nextIndex < routes.tiles.count {
                let decision = scheduler.decide(PrefillRoutedTileSchedulerInput(
                    hasPendingTile: true, pendingDepth: 1, pendingAssignedSlots: plannedSlots,
                    avoidingSlotPlanAvailable: true))
                if case .prefetchNext(let avoidingSlots) = decision {
                    let nextPlan = try planForTile(nextIndex, avoidingSlots: Set(avoidingSlots))
                    nextPending = (nextIndex, nextPlan, Task { try await fetchProvider.fetch(plan: nextPlan) })
                }
            }

            let views = try await current.task.value
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes)
            guard current.plan.experts == expertIDs else {
                throw V4GroupedRoutedMoEPrefillAdapterError.preplannedFetchMismatch(tile: tileIndex)
            }
            let binding = try PrefillStreamedTileBinding(expertIDs: expertIDs, views: views)
            try binding.validateCoversPairs(routes.sortedPairs, pairStart: Int(routes.tiles[tileIndex].pairStart),
                                            pairCount: Int(routes.tiles[tileIndex].pairCount))
            let argumentBuffer = try moe.makeStreamedArgumentBuffer(binding: binding)
            let commandBuffer = try makeCommandBuffer()
            let tile = routes.tiles[tileIndex]
            let params = PrefillGroupedRoutedMoEV4StreamedParams(
                pairStart: tile.pairStart, pairCount: tile.pairCount, d: UInt32(d),
                routedIntermediate: UInt32(routedIntermediate), topK: UInt32(PrefillGroupedRoutedMoEV4.topK),
                hiddenStrideElements: UInt32(hiddenStrideElements), binding: binding, offsets: offsets)
            moe.encodeStreamedTile(commandBuffer: commandBuffer, hidden: hidden, hiddenOffset: hiddenOffset,
                                   sortedPairs: sortedPairs, acts: acts, routePartials: routePartials,
                                   routeWeights: routeWeights, routeWeightsOffset: routeWeightsOffset,
                                   argumentBuffer: argumentBuffer, binding: binding, params: params,
                                   pairMicrobatchRows: pairMicrobatchRows)
            commandBuffer.commit()
            await commandBuffer.completed()
            if let error = commandBuffer.error { throw error }
            try lifetime.complete(tileIndex: tileIndex)
            pending = nextPending
        }

        let reduceCommandBuffer = try makeCommandBuffer()
        moe.encodeReduceTokenMajor(commandBuffer: reduceCommandBuffer, routePartials: routePartials,
                                   residual: residual, residualOffset: residualOffset,
                                   h2: output, queryCount: UInt32(queryCount),
                                   topK: UInt32(PrefillGroupedRoutedMoEV4.topK), d: UInt32(d))
        reduceCommandBuffer.commit()
        await reduceCommandBuffer.completed()
        if let error = reduceCommandBuffer.error { throw error }
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw V4GroupedRoutedMoEPrefillAdapterError.commandBufferUnavailable
        }
        return commandBuffer
    }

    private func makeEmptyBuffer(length: Int, label: String) throws -> MTLBuffer {
        guard let buffer = context.device.makeBuffer(length: max(1, length), options: .storageModeShared) else {
            throw V4GroupedRoutedMoEPrefillAdapterError.allocationFailed(label)
        }
        buffer.label = label
        memset(buffer.contents(), 0, max(1, length))
        return buffer
    }

    private func makeBuffer<T>(bytes: [T], label: String) -> MTLBuffer? {
        guard !bytes.isEmpty else {
            let buffer = context.device.makeBuffer(length: 1, options: .storageModeShared)
            buffer?.label = label
            return buffer
        }
        let length = bytes.count * MemoryLayout<T>.stride
        let buffer = bytes.withUnsafeBufferPointer { ptr in
            context.device.makeBuffer(bytes: ptr.baseAddress!, length: length, options: .storageModeShared)
        }
        buffer?.label = label
        return buffer
    }
}

struct V4GroupedRoutedMoEPrefillOrchestrationSimulator {
    var scheduler = PrefillRoutedTileScheduler(config: PrefillGroupedRoutedMoEV4.tileSchedulerConfig)

    func plan(tokenMajorExpertIDs: [UInt32], tokenMajorWeights: [Float], queryCount: Int,
              numExperts: Int, slotPlans: [[Int]]) throws -> V4GroupedRoutedMoEPrefillPlan {
        let pairs = try V4GroupedRoutedMoEPrefillAdapter.makePairs(tokenMajorExpertIDs: tokenMajorExpertIDs,
                                                                  tokenMajorWeights: tokenMajorWeights,
                                                                  queryCount: queryCount)
        let routes = try PrefillGroupedRoutedMoEV4.groupedRoutes(pairs: pairs, queryCount: queryCount, numExperts: numExperts)
        var lifetime = PrefillStreamedTileSlotLifetime()
        var records: [V4GroupedRoutedMoEPrefillTileRecord] = []
        for tileIndex in routes.tiles.indices {
            let plannedSlots = slotPlans[tileIndex]
            try lifetime.begin(tileIndex: tileIndex, plannedSlots: plannedSlots)
            var avoiding: [Int] = []
            var prefetched: Int?
            if tileIndex + 1 < routes.tiles.count {
                let decision = scheduler.decide(PrefillRoutedTileSchedulerInput(
                    hasPendingTile: true, pendingDepth: 1, pendingAssignedSlots: plannedSlots,
                    avoidingSlotPlanAvailable: true))
                if case .prefetchNext(let avoidingSlots) = decision {
                    avoiding = avoidingSlots
                    prefetched = tileIndex + 1
                }
            }
            records.append(V4GroupedRoutedMoEPrefillTileRecord(
                tileIndex: tileIndex,
                expertIDs: try PrefillStreamedTileBinding.expertIDs(forTile: tileIndex, routes: routes),
                plannedSlots: plannedSlots,
                avoidingSlotsForNextPlan: avoiding,
                prefetchedNextTile: prefetched))
            try lifetime.complete(tileIndex: tileIndex)
        }
        return V4GroupedRoutedMoEPrefillPlan(routes: routes, records: records)
    }
}
