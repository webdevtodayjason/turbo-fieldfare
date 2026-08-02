import Testing
@testable import TurboFieldfare

@Suite struct V4ChunkedPrefillStateLayoutTests {
    @Test func realGeometryUsesCorrectElementWidths() {
        let layout = V4ChunkedPrefillState.Layout(
            rows: 128,
            dim: 4096,
            streams: 4,
            heads: 64,
            headDim: 512,
            indexHeadDim: 128,
            experts: 256,
            topK: 6,
            ffn: 2048)

        #expect(layout.tokenIDsBytes == 128 * 4)
        #expect(layout.streamBytes == 128 * 4 * 4096 * 4)
        #expect(layout.branchBytes == 128 * 4096 * 4)
        #expect(layout.xNormBytes == 128 * 4096 * 2)
        #expect(layout.qBytes == 128 * 64 * 512 * 2)
        #expect(layout.indexWeightsBytes == 128 * 64 * 4)
        #expect(layout.compressorBytes == 128 * 1024 * 4)
        #expect(layout.indexerCompressorBytes == 128 * 256 * 4)
        #expect(layout.routerLogitsBytes == 128 * 256 * 4)
        #expect(layout.routeWeightsBytes == 128 * 6 * 4)
        #expect(layout.sharedIntermediateBytes == 128 * 2048 * 2)
        #expect(layout.lowRankBytes == 128 * 8192 * 2)
    }
}
