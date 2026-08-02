import Testing
@testable import TurboFieldfare

@Suite struct V4ChunkedPrefillStateLayoutTests {
    @Test func realGeometryUsesCorrectElementWidths() {
        let layout = V4ChunkedPrefillState.Layout(rows: 128,
                                                  dim: 4096,
                                                  ffn: 2048,
                                                  experts: 256,
                                                  layers: 62)

        #expect(layout.tokenIDsBytes == 128 * 4)
        #expect(layout.streamBytes == 128 * 4 * 4096 * 4)
        #expect(layout.branchBytes == 128 * 4096 * 4)
        #expect(layout.xNormBytes == 128 * 4096 * 2)
        #expect(layout.qBytes == 128 * 64 * 512 * 2)
        #expect(layout.indexQBytes == 128 * 64 * 128 * 2)
        #expect(layout.indexWeightsBytes == 128 * 64 * 4)
        #expect(layout.windowKVBytes == 128 * 512 * 2)
        #expect(layout.compressorKVBytes == 128 * 1024 * 4)
        #expect(layout.compressorGateBytes == 128 * 1024 * 4)
        #expect(layout.indexerCompressorKVBytes == 128 * 256 * 4)
        #expect(layout.indexerCompressorGateBytes == 128 * 256 * 4)
        #expect(layout.attnBytes == 128 * 64 * 512 * 2)
        #expect(layout.lowRankBytes == 128 * 8192 * 2)
        #expect(layout.oProjBytes == 128 * 4096 * 2)
        #expect(layout.routerLogitsBytes == 128 * 256 * 4)
        #expect(layout.routeIDsBytes == 128 * 6 * 4)
        #expect(layout.routeWeightsBytes == 128 * 6 * 4)
        #expect(layout.sharedGateBytes == 128 * 2048 * 2)
        #expect(layout.sharedUpBytes == 128 * 2048 * 2)
        #expect(layout.sharedActBytes == 128 * 2048 * 2)
        #expect(layout.sharedOutBytes == 128 * 4096 * 2)
        #expect(layout.routedPlusSharedBytes == 128 * 4096 * 2)

        #expect(layout.streamBytes != 128 * 4 * 4096 * 2)
        #expect(layout.branchBytes != 128 * 4096 * 2)
        #expect(layout.indexWeightsBytes != 128 * 64 * 2)
        #expect(layout.compressorKVBytes != 128 * 1024 * 2)
        #expect(layout.compressorGateBytes != 128 * 1024 * 2)
        #expect(layout.indexerCompressorKVBytes != 128 * 256 * 2)
        #expect(layout.indexerCompressorGateBytes != 128 * 256 * 2)
        #expect(layout.routerLogitsBytes != 128 * 256 * 2)
        #expect(layout.routeWeightsBytes != 128 * 6 * 2)
    }

    @Test func v4LayoutSupportsAChunkLargerThanItsSlidingWindow() {
        let layout = V4ChunkedPrefillState.Layout(rows: 256,
                                                  dim: 4096,
                                                  ffn: 2048,
                                                  experts: 256,
                                                  layers: 43)
        #expect(layout.rows == 256)
        #expect(layout.qBytes == 256 * 64 * 512 * MemoryLayout<Float16>.stride)
    }
}
