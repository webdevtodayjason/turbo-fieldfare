import Testing
import Metal
@testable import TurboFieldfare

@Suite struct V4CompressorAccumulatorStoreTests {
    @Test func accumulatorBuffersMatchReferenceInitialState() throws {
        let ctx = try MetalContext()
        let store = try V4CompressorAccumulatorStore(device: ctx.device,
                                                     layerKinds: [.csa, .hca])

        let csa = store.csa(layer: 0)
        #expect(Self.read(csa.prevKV, offset: 0) == 0)
        #expect(Self.read(csa.prevGate, offset: 0) == -.infinity)
        #expect(Self.read(csa.idxPrevKV, offset: 0) == 0)
        #expect(Self.read(csa.idxPrevGate, offset: 0) == -.infinity)
        #expect(Self.read(csa.prevGate,
                          offset: V4CompressorAccumulatorStore.csaRows * V4CompressorAccumulatorStore.csaWidth - 1) == -.infinity)
        #expect(Self.read(csa.idxPrevGate,
                          offset: V4CompressorAccumulatorStore.csaRows * V4CompressorAccumulatorStore.csaIndexerWidth - 1) == -.infinity)
        #expect(Self.read(csa.curKV, offset: V4CompressorAccumulatorStore.csaRows * V4CompressorAccumulatorStore.csaWidth - 1) == 0)
        #expect(Self.read(csa.idxCurGate, offset: V4CompressorAccumulatorStore.csaRows * V4CompressorAccumulatorStore.csaIndexerWidth - 1) == 0)

        let hca = store.hca(layer: 1)
        #expect(Self.read(hca.ringKV, offset: 0) == 0)
        #expect(Self.read(hca.ringGate, offset: V4CompressorAccumulatorStore.hcaRows * V4CompressorAccumulatorStore.hcaWidth - 1) == 0)
    }

    @Test func csaLayersKeepDistinctAccumulatorRows() throws {
        let ctx = try MetalContext()
        let store = try V4CompressorAccumulatorStore(
            device: ctx.device,
            layerKinds: [.csa, .csa, .passthrough])

        let layer0 = store.csa(layer: 0)
        let layer1 = store.csa(layer: 1)
        Self.writeRow(layer0.curKV, row: 2, width: V4CompressorAccumulatorStore.csaWidth, value: 11)
        Self.writeRow(layer1.curKV, row: 2, width: V4CompressorAccumulatorStore.csaWidth, value: 22)
        Self.writeRow(layer0.curGate, row: 1, width: V4CompressorAccumulatorStore.csaWidth, value: 33)
        Self.writeRow(layer1.curGate, row: 1, width: V4CompressorAccumulatorStore.csaWidth, value: 44)

        #expect(Self.read(layer0.curKV, offset: 2 * V4CompressorAccumulatorStore.csaWidth) == 11)
        #expect(Self.read(layer1.curKV, offset: 2 * V4CompressorAccumulatorStore.csaWidth) == 22)
        #expect(Self.read(layer0.curGate, offset: V4CompressorAccumulatorStore.csaWidth) == 33)
        #expect(Self.read(layer1.curGate, offset: V4CompressorAccumulatorStore.csaWidth) == 44)
    }

    @Test func hcaLayersKeepDistinctAccumulatorRows() throws {
        let ctx = try MetalContext()
        let store = try V4CompressorAccumulatorStore(
            device: ctx.device,
            layerKinds: [.hca, .passthrough, .hca])

        let layer0 = store.hca(layer: 0)
        let layer2 = store.hca(layer: 2)
        Self.writeRow(layer0.ringKV, row: 127, width: V4CompressorAccumulatorStore.hcaWidth, value: 101)
        Self.writeRow(layer2.ringKV, row: 127, width: V4CompressorAccumulatorStore.hcaWidth, value: 202)
        Self.writeRow(layer0.ringGate, row: 64, width: V4CompressorAccumulatorStore.hcaWidth, value: 303)
        Self.writeRow(layer2.ringGate, row: 64, width: V4CompressorAccumulatorStore.hcaWidth, value: 404)

        #expect(Self.read(layer0.ringKV, offset: 127 * V4CompressorAccumulatorStore.hcaWidth) == 101)
        #expect(Self.read(layer2.ringKV, offset: 127 * V4CompressorAccumulatorStore.hcaWidth) == 202)
        #expect(Self.read(layer0.ringGate, offset: 64 * V4CompressorAccumulatorStore.hcaWidth) == 303)
        #expect(Self.read(layer2.ringGate, offset: 64 * V4CompressorAccumulatorStore.hcaWidth) == 404)
    }

    @Test func csaRollCurrentToPreviousAffectsOnlyRequestedLayer() throws {
        let ctx = try MetalContext()
        let store = try V4CompressorAccumulatorStore(device: ctx.device, layerKinds: [.csa, .csa])
        let layer0 = store.csa(layer: 0)
        let layer1 = store.csa(layer: 1)

        Self.fill(layer0.curKV, value: 10)
        Self.fill(layer0.curGate, value: 11)
        Self.fill(layer0.idxCurKV, value: 12)
        Self.fill(layer0.idxCurGate, value: 13)
        Self.fill(layer1.curKV, value: 20)
        Self.fill(layer1.curGate, value: 21)
        Self.fill(layer1.idxCurKV, value: 22)
        Self.fill(layer1.idxCurGate, value: 23)
        Self.fill(layer0.prevKV, value: -1)
        Self.fill(layer0.prevGate, value: -1)
        Self.fill(layer0.idxPrevKV, value: -1)
        Self.fill(layer0.idxPrevGate, value: -1)
        Self.fill(layer1.prevKV, value: -2)
        Self.fill(layer1.prevGate, value: -2)
        Self.fill(layer1.idxPrevKV, value: -2)
        Self.fill(layer1.idxPrevGate, value: -2)

        guard let cb = ctx.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        store.rollCSA(commandBuffer: cb, layer: 0)
        cb.commit()
        cb.waitUntilCompleted()

        #expect(Self.read(layer0.prevKV, offset: 0) == 10)
        #expect(Self.read(layer0.prevGate, offset: 0) == 11)
        #expect(Self.read(layer0.idxPrevKV, offset: 0) == 12)
        #expect(Self.read(layer0.idxPrevGate, offset: 0) == 13)
        #expect(Self.read(layer1.prevKV, offset: 0) == -2)
        #expect(Self.read(layer1.prevGate, offset: 0) == -2)
        #expect(Self.read(layer1.idxPrevKV, offset: 0) == -2)
        #expect(Self.read(layer1.idxPrevGate, offset: 0) == -2)
    }

    private static func writeRow(_ buffer: MTLBuffer, row: Int, width: Int, value: Float) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        ptr[row * width] = value
    }

    private static func fill(_ buffer: MTLBuffer, value: Float) {
        let ptr = buffer.contents().assumingMemoryBound(to: Float.self)
        let count = buffer.length / MemoryLayout<Float>.stride
        for i in 0..<count { ptr[i] = value }
    }

    private static func read(_ buffer: MTLBuffer, offset: Int) -> Float {
        buffer.contents().assumingMemoryBound(to: Float.self)[offset]
    }
}
