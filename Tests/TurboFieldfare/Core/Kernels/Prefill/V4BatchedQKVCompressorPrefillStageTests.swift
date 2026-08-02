import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct V4BatchedQKVCompressorPrefillStageTests {
    private static let dim = 4096
    private static let qLora = 1024
    private static let heads = 64
    private static let headDim = 512
    private static let indexDim = 128
    private static let rows = 3
    private static let startPosition = 126

    private static func randomFP8(m: Int, n: Int, rng: inout SplitMix64) -> V4Quantization.FP8BlockMatrix {
        let codes = (0..<(m * n)).map { _ -> UInt8 in
            var b = UInt8(clamping: Int(rng.uniform(0, 256))) & 0x7F
            if b == 0x7F { b = 0x7E }
            if rng.uniform(0, 1) < 0.5 { b |= 0x80 }
            return b
        }
        let scales = (0..<((m / 128) * (n / 128))).map { _ -> UInt8 in
            UInt8(clamping: 127 + Int(rng.uniform(-6, -2)))
        }
        return V4Quantization.FP8BlockMatrix(m: m, n: n, codes: codes, scales: scales)
    }

    private static func bytes<T>(_ device: MTLDevice, _ values: [T]) -> MTLBuffer? {
        device.makeBuffer(bytes: values, length: values.count * MemoryLayout<T>.stride,
                          options: .storageModeShared)
    }

    private static func makeWeights(device: MTLDevice, includeIndexer: Bool) throws -> V4QKVEpilogue.Weights {
        var rng = SeedTree(includeIndexer ? 0xB471_C5A : 0xB471).key("batched-qkv")
        let wqA = randomFP8(m: qLora, n: dim, rng: &rng)
        let wqB = randomFP8(m: heads * headDim, n: qLora, rng: &rng)
        let wkv = randomFP8(m: headDim, n: dim, rng: &rng)
        let qGamma = (0..<qLora).map { _ in rng.uniform(0.5, 1.5) }
        let kvGamma = (0..<headDim).map { _ in rng.uniform(0.5, 1.5) }
        guard let wqAC = bytes(device, wqA.codes), let wqAS = bytes(device, wqA.scales),
              let wqBC = bytes(device, wqB.codes), let wqBS = bytes(device, wqB.scales),
              let wkvC = bytes(device, wkv.codes), let wkvS = bytes(device, wkv.scales),
              let qGB = bytes(device, qGamma), let kvGB = bytes(device, kvGamma) else {
            throw MetalError.noDevice
        }
        var weights = V4QKVEpilogue.Weights(wqA: (wqAC, wqAS), wqB: (wqBC, wqBS),
                                            qNormGamma: qGB,
                                            windowWKV: (wkvC, wkvS), kvNormGamma: kvGB)
        if includeIndexer {
            let widx = randomFP8(m: heads * indexDim, n: qLora, rng: &rng)
            guard let widxC = bytes(device, widx.codes), let widxS = bytes(device, widx.scales) else {
                throw MetalError.noDevice
            }
            weights.indexerWqB = (widxC, widxS)
        }
        return weights
    }

    private static func runParity(includeIndexer: Bool) throws {
        var rng = SeedTree(includeIndexer ? 0x1DE_C5A : 0x1DE).key("hidden")
        let hidden = (0..<(rows * dim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let ctx = try MetalContext()
        let weights = try makeWeights(device: ctx.device, includeIndexer: includeIndexer)
        let stage = try V4BatchedQKVCompressorPrefillStage(device: ctx.device, maxRows: rows)
        let epi = try V4QKVEpilogue(device: ctx.device)

        guard let hiddenBuf = Fp16Buffer.make(ctx.device, halves: hidden),
              let batchedQ = Fp16Buffer.make(ctx.device, count: rows * heads * headDim),
              let batchedKV = Fp16Buffer.make(ctx.device, count: rows * headDim),
              let refQ = Fp16Buffer.make(ctx.device, count: rows * heads * headDim),
              let refKV = Fp16Buffer.make(ctx.device, count: rows * headDim) else {
            Issue.record("alloc failed"); return
        }
        let batchedIndexQ = includeIndexer ? Fp16Buffer.make(ctx.device, count: rows * heads * indexDim) : nil
        let refIndexQ = includeIndexer ? Fp16Buffer.make(ctx.device, count: rows * heads * indexDim) : nil
        let refIndexScratch = includeIndexer ? Fp16Buffer.make(ctx.device, count: heads * indexDim) : nil
        let indexWeightsIn = includeIndexer ? bytes(ctx.device, (0..<(rows * heads)).map { Float($0) / 17 }) : nil
        let indexWeightsOut = includeIndexer ? ctx.device.makeBuffer(length: rows * heads * MemoryLayout<Float>.stride,
                                                                    options: .storageModeShared) : nil
        if includeIndexer && (batchedIndexQ == nil || refIndexQ == nil || refIndexScratch == nil || indexWeightsIn == nil || indexWeightsOut == nil) {
            Issue.record("index alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        var outputs = V4BatchedQKVCompressorPrefillStage.Outputs(qOut: batchedQ, windowKVOut: batchedKV)
        outputs.indexQOut = batchedIndexQ
        outputs.indexWeightsOut = indexWeightsOut
        stage.encode(commandBuffer: cb,
                     hiddenRows: hiddenBuf,
                     rowCount: rows,
                     startPosition: startPosition,
                     weights: weights,
                     rope: .compressedLayer,
                     outputs: outputs,
                     indexerPerHeadWeights: indexWeightsIn.map { .init(buffer: $0) })
        for row in 0..<rows {
            epi.encodeDecode(commandBuffer: cb,
                             x: hiddenBuf,
                             xOffset: row * dim * MemoryLayout<Float16>.stride,
                             position: startPosition + row,
                             weights: weights,
                             rope: .compressedLayer,
                             qOut: refQ,
                             qOutOffset: row * heads * headDim * MemoryLayout<Float16>.stride,
                             windowSlot: .init(buffer: refKV,
                                               offset: row * headDim * MemoryLayout<Float16>.stride),
                             indexQOut: refIndexScratch)
            if let refIndexQ, let refIndexScratch {
                // encodeDecode writes indexQOut at offset zero. Copy that row into
                // place before the next serial reference row overwrites it.
                let blit = cb.makeBlitCommandEncoder()
                blit?.copy(from: refIndexScratch, sourceOffset: 0,
                           to: refIndexQ, destinationOffset: row * heads * indexDim * MemoryLayout<Float16>.stride,
                           size: heads * indexDim * MemoryLayout<Float16>.stride)
                blit?.endEncoding()
            }
        }
        cb.commit(); cb.waitUntilCompleted()

        let batchedQValues = Fp16Buffer.read(batchedQ, count: rows * heads * headDim)
        let refQValues = Fp16Buffer.read(refQ, count: rows * heads * headDim)
        #expect(batchedQValues == refQValues, "batched Q must be bit-exact to decode")
        let batchedKVValues = Fp16Buffer.read(batchedKV, count: rows * headDim)
        let refKVValues = Fp16Buffer.read(refKV, count: rows * headDim)
        #expect(batchedKVValues == refKVValues, "batched window KV must be bit-exact to decode")
        if includeIndexer, let batchedIndexQ, let refIndexQ, let indexWeightsIn, let indexWeightsOut {
            let idxRel = RelError.compute(actual: Fp16Buffer.read(batchedIndexQ, count: rows * heads * indexDim),
                                          reference: Fp16Buffer.read(refIndexQ, count: rows * heads * indexDim))
            #expect(idxRel < Tolerance.fp16ChainedReduction, "index q rel=\(idxRel)")
            let got = (0..<(rows * heads)).map { indexWeightsOut.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
            let ref = (0..<(rows * heads)).map { indexWeightsIn.contents().load(fromByteOffset: $0 * 4, as: Float.self) }
            #expect(got == ref)
        }
    }

    @Test func passthroughBatchedRows_matchSerialDecodeAcrossWindowBoundary() throws {
        try Self.runParity(includeIndexer: false)
    }

    @Test func csaIndexerBatchedRows_matchSerialDecodeAcrossWindowBoundary() throws {
        try Self.runParity(includeIndexer: true)
    }
}
