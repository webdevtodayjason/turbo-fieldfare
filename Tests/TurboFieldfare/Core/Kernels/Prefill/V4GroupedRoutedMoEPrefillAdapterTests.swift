import Testing
@testable import TurboFieldfare

@Suite struct V4GroupedRoutedMoEPrefillAdapterTests {
    private static let topK = PrefillGroupedRoutedMoEV4.topK

    private static func tokenMajorIDs(_ rows: [[UInt32]]) -> [UInt32] {
        rows.flatMap { row in
            precondition(row.count == topK)
            return row
        }
    }

    private static func weights(queryCount: Int) -> [Float] {
        (0..<(queryCount * topK)).map { Float($0 + 1) / 100 }
    }

    @Test func groupsNineExpertsAcrossEightExpertTileBoundary() throws {
        let ids = Self.tokenMajorIDs([
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 0, 1, 2]
        ])
        let plan = try V4GroupedRoutedMoEPrefillOrchestrationSimulator().plan(
            tokenMajorExpertIDs: ids,
            tokenMajorWeights: Self.weights(queryCount: 2),
            queryCount: 2,
            numExperts: 16,
            slotPlans: [Array(0..<8), [8]])

        #expect(plan.routes.groups.map(\.expert) == Array(0...8).map(UInt32.init))
        #expect(plan.routes.tiles.count == 2)
        #expect(plan.routes.tiles[0].groupCount == 8)
        #expect(plan.routes.tiles[1].groupCount == 1)
        #expect(plan.records[0].expertIDs == Array(0...7))
        #expect(plan.records[1].expertIDs == [8])
        #expect(plan.records[0].prefetchedNextTile == 1)
        #expect(plan.records[0].avoidingSlotsForNextPlan == Array(0..<8))
    }

    @Test func hitMissOverlapPassesAvoidingSlotsToPrefetch() throws {
        let ids = Self.tokenMajorIDs([
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0, 1]
        ])
        let plan = try V4GroupedRoutedMoEPrefillOrchestrationSimulator().plan(
            tokenMajorExpertIDs: ids,
            tokenMajorWeights: Self.weights(queryCount: 2),
            queryCount: 2,
            numExperts: 16,
            slotPlans: [[0, 1, 2, 3, 4, 5, 12, 13], [6, 7]])

        #expect(plan.records.count == 2)
        #expect(plan.records[0].expertIDs == Array(0...7))
        #expect(plan.records[1].expertIDs == [8, 9])
        #expect(plan.records[0].plannedSlots == [0, 1, 2, 3, 4, 5, 12, 13])
        #expect(plan.records[0].avoidingSlotsForNextPlan == [0, 1, 2, 3, 4, 5, 12, 13])
        #expect(Set(plan.records[0].plannedSlots).isDisjoint(with: Set(plan.records[1].plannedSlots)))
    }

    @Test func slotLifetimeRejectsReuseBeforeCompletion() throws {
        var lifetime = PrefillStreamedTileSlotLifetime()
        try lifetime.begin(tileIndex: 0, plannedSlots: [0, 1, 2])
        #expect(throws: PrefillStreamedTileLifetimeError.slotReuseBeforeCompletion(
            tileIndex: 1,
            conflictingTileIndex: 0,
            slots: [2])) {
            try lifetime.begin(tileIndex: 1, plannedSlots: [2, 3, 4])
        }
        try lifetime.complete(tileIndex: 0)
        try lifetime.begin(tileIndex: 1, plannedSlots: [2, 3, 4])
    }

    @Test func hashAndRouterIDsNormalizeIdenticallyAndDeterministically() throws {
        let hashProduced = Self.tokenMajorIDs([
            [4, 2, 7, 0, 1, 3],
            [8, 5, 6, 4, 2, 7]
        ])
        let weights = Self.weights(queryCount: 2)
        let hashPairs = try V4GroupedRoutedMoEPrefillAdapter.makePairs(
            tokenMajorExpertIDs: hashProduced,
            tokenMajorWeights: weights,
            queryCount: 2)
        let routerPairs = try V4GroupedRoutedMoEPrefillAdapter.makePairs(
            tokenMajorExpertIDs: hashProduced,
            tokenMajorWeights: weights,
            queryCount: 2)
        #expect(hashPairs == routerPairs)

        let first = try PrefillGroupedRoutedMoEV4.groupedRoutes(pairs: hashPairs, queryCount: 2, numExperts: 16)
        let second = try PrefillGroupedRoutedMoEV4.groupedRoutes(pairs: routerPairs, queryCount: 2, numExperts: 16)
        #expect(first == second)
        #expect(first.sortedPairs == second.sortedPairs)
        #expect(first.tiles == second.tiles)
    }
}
