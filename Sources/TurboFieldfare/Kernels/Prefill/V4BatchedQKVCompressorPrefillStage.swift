import Foundation
import Metal

/// Batched Q/KV/compressor projection stage for V4 chunked prefill.
///
/// This class is orchestration only. It composes the existing batched prefill
/// projection kernels (`V4PrefillProj`), batched boundary kernels
/// (`V4PrefillBoundary`), and the decode epilogue's exposed per-head renorm
/// kernel. It does not duplicate the serial `V4QKVEpilogue.encodeDecode` path.
///
/// Input hidden rows are normalized fp16 `[rows, 4096]`. Outputs use decode's
/// staging layout:
/// - `qOut`: `[rows, 64, 512]` fp16 after q_a -> RMSNorm -> q_b -> per-head
///   renorm -> interleaved trailing RoPE.
/// - `windowKVOut`: `[rows, 512]` fp16 after projection -> gamma RMSNorm ->
///   interleaved trailing RoPE.
/// - `indexQOut`: optional `[rows, 64, 128]` fp16 after indexer projection from
///   shared q latent and RoPE.
/// - compressor outputs: optional `[rows, compressorOutDim]` fp32 for WKV/WGate.
/// - `indexWeightsOut`: optional caller-provided per-head fp32 weights copied
///   to `[rows, 64]` when supplied.
final class V4BatchedQKVCompressorPrefillStage {
    static let dim = V4QKVEpilogue.dim
    static let qLoraRank = V4QKVEpilogue.qLoraRank
    static let numQHeads = V4QKVEpilogue.numQHeads
    static let headDim = V4QKVEpilogue.headDim
    static let indexHeadDim = V4QKVEpilogue.indexHeadDim

    struct Outputs {
        var qOut: MTLBuffer
        var qOutOffset: Int = 0
        var windowKVOut: MTLBuffer
        var windowKVOutOffset: Int = 0
        var compressorWKVOut: MTLBuffer? = nil
        var compressorWKVOutOffset: Int = 0
        var compressorWGateOut: MTLBuffer? = nil
        var compressorWGateOutOffset: Int = 0
        var indexQOut: MTLBuffer? = nil
        var indexQOutOffset: Int = 0
        var indexWeightsOut: MTLBuffer? = nil
        var indexWeightsOutOffset: Int = 0
    }

    struct IndexerPerHeadWeights {
        var buffer: MTLBuffer
        var offset: Int = 0
    }

    private let maxRows: Int
    private let proj: V4PrefillProj
    private let serialGlue: V4ChunkedPrefillGlue
    private let boundary: V4PrefillBoundary
    private let epilogue: V4QKVEpilogue
    private let qr: MTLBuffer
    private let positions: MTLBuffer
    private let repeatedHeadPositions: MTLBuffer

    init(device: MTLDevice, maxRows: Int = 128) throws {
        precondition(maxRows > 0)
        self.maxRows = maxRows
        self.proj = try V4PrefillProj(device: device)
        self.serialGlue = try V4ChunkedPrefillGlue(device: device)
        self.boundary = try V4PrefillBoundary(device: device, maxRows: maxRows * Self.numQHeads)
        self.epilogue = try V4QKVEpilogue(device: device)
        // Decode stores q_a/window_wkv projections as fp16 before RMSNorm.
        // Keep the same quantization point and fp16 row stride for strict
        // parity. Packing half rows into fp32 scratch aliases later rows.
        guard let qr = device.makeBuffer(length: maxRows * Self.qLoraRank * MemoryLayout<Float16>.stride,
                                         options: .storageModeShared),
              let positions = device.makeBuffer(length: maxRows * MemoryLayout<Float>.stride,
                                                options: .storageModeShared),
              let repeated = device.makeBuffer(length: maxRows * Self.numQHeads * MemoryLayout<Float>.stride,
                                               options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        qr.label = "v4batched-qkv.qr"
        positions.label = "v4batched-qkv.positions"
        repeated.label = "v4batched-qkv.repeatedHeadPositions"
        self.qr = qr
        self.positions = positions
        self.repeatedHeadPositions = repeated
    }

    func encode(commandBuffer cb: MTLCommandBuffer,
                hiddenRows: MTLBuffer,
                hiddenRowsOffset: Int = 0,
                rowCount: Int,
                startPosition: Int,
                weights: V4QKVEpilogue.Weights,
                rope: V4RoPE.Config,
                outputs: Outputs,
                indexerPerHeadWeights: IndexerPerHeadWeights? = nil,
                normEps: Float = 1e-6) {
        precondition(rowCount > 0 && rowCount <= maxRows)
        precondition(startPosition >= 0)
        precondition(hiddenRowsOffset % 8 == 0, "FP8 row reads need 8-aligned hiddenRowsOffset")
        if weights.compressorWKV != nil || weights.compressorWGate != nil {
            precondition(weights.compressorWKV != nil && weights.compressorWGate != nil)
            precondition(weights.compressorOutDim == 1024 || weights.compressorOutDim == 512)
            precondition(outputs.compressorWKVOut != nil && outputs.compressorWGateOut != nil)
        }
        if weights.indexerWqB != nil { precondition(outputs.indexQOut != nil) }
        if indexerPerHeadWeights != nil { precondition(outputs.indexWeightsOut != nil) }

        fillPositions(rowCount: rowCount, startPosition: startPosition)

        proj.encodeFP8GEMM(commandBuffer: cb,
                           weights: weights.wqA.codes, weightsOffset: weights.wqACodesOffset,
                           scales: weights.wqA.scales, scalesOffset: weights.wqAScalesOffset,
                           x: hiddenRows, xOffset: hiddenRowsOffset,
                           out: qr, rows: rowCount, m: Self.qLoraRank, n: Self.dim,
                           outFP16: true)
        boundary.encodeRMSNormF16(commandBuffer: cb,
                                  x: qr, gamma: weights.qNormGamma,
                                  gammaOffset: weights.qNormGammaOffset,
                                  out: qr, rows: rowCount, n: Self.qLoraRank,
                                  eps: normEps, useGamma: true)

        proj.encodeFP8GEMM(commandBuffer: cb,
                           weights: weights.wqB.codes, weightsOffset: weights.wqBCodesOffset,
                           scales: weights.wqB.scales, scalesOffset: weights.wqBScalesOffset,
                           x: qr, out: outputs.qOut, outOffset: outputs.qOutOffset,
                           rows: rowCount, m: Self.numQHeads * Self.headDim, n: Self.qLoraRank,
                           outFP16: true)
        epilogue.encodePerHeadRenorm(commandBuffer: cb,
                                     buf: outputs.qOut, bufOffset: outputs.qOutOffset,
                                     heads: rowCount * Self.numQHeads,
                                     headDim: Self.headDim, eps: normEps)
        boundary.encodeRoPE(commandBuffer: cb,
                            x: outputs.qOut, xOffset: outputs.qOutOffset,
                            positions: repeatedHeadPositions,
                            rows: rowCount * Self.numQHeads,
                            width: Self.headDim,
                            ropeDim: 64,
                            inverse: false,
                            config: rope)

        proj.encodeFP8GEMM(commandBuffer: cb,
                           weights: weights.windowWKV.codes, weightsOffset: weights.windowWKVCodesOffset,
                           scales: weights.windowWKV.scales, scalesOffset: weights.windowWKVScalesOffset,
                           x: hiddenRows, xOffset: hiddenRowsOffset,
                           out: outputs.windowKVOut,
                           outOffset: outputs.windowKVOutOffset,
                           rows: rowCount, m: Self.headDim, n: Self.dim,
                           outFP16: true)
        boundary.encodeRMSNormF16(commandBuffer: cb,
                                  x: outputs.windowKVOut,
                                  xOffset: outputs.windowKVOutOffset,
                                  gamma: weights.kvNormGamma,
                                  gammaOffset: weights.kvNormGammaOffset,
                                  out: outputs.windowKVOut,
                                  outOffset: outputs.windowKVOutOffset,
                                  rows: rowCount, n: Self.headDim,
                                  eps: normEps, useGamma: true)
        boundary.encodeRoPE(commandBuffer: cb,
                            x: outputs.windowKVOut, xOffset: outputs.windowKVOutOffset,
                            positions: positions,
                            rows: rowCount,
                            width: Self.headDim,
                            ropeDim: 64,
                            inverse: false,
                            config: rope)
        epilogue.encodeWindowKVQAT(commandBuffer: cb,
                                   buf: outputs.windowKVOut,
                                   bufOffset: outputs.windowKVOutOffset,
                                   rows: rowCount)

        if let wkv = weights.compressorWKV, let wgate = weights.compressorWGate,
           let wkvOut = outputs.compressorWKVOut, let wgateOut = outputs.compressorWGateOut {
            serialGlue.encodeBF16GEMMSerialOrder(
                commandBuffer: cb,
                weights: wkv, weightsOffset: weights.compressorWKVOffset,
                hidden: hiddenRows, hiddenOffset: hiddenRowsOffset,
                output: wkvOut, outputOffset: outputs.compressorWKVOutOffset,
                rows: rowCount, outputRows: weights.compressorOutDim, dim: Self.dim)
            serialGlue.encodeBF16GEMMSerialOrder(
                commandBuffer: cb,
                weights: wgate, weightsOffset: weights.compressorWGateOffset,
                hidden: hiddenRows, hiddenOffset: hiddenRowsOffset,
                output: wgateOut, outputOffset: outputs.compressorWGateOutOffset,
                rows: rowCount, outputRows: weights.compressorOutDim, dim: Self.dim)
        }

        if let indexer = weights.indexerWqB, let indexQOut = outputs.indexQOut {
            proj.encodeFP8GEMM(commandBuffer: cb,
                               weights: indexer.codes, weightsOffset: weights.indexerWqBCodesOffset,
                               scales: indexer.scales, scalesOffset: weights.indexerWqBScalesOffset,
                               x: qr,
                               out: indexQOut, outOffset: outputs.indexQOutOffset,
                               rows: rowCount, m: Self.numQHeads * Self.indexHeadDim, n: Self.qLoraRank,
                               outFP16: true)
            boundary.encodeRoPE(commandBuffer: cb,
                                x: indexQOut, xOffset: outputs.indexQOutOffset,
                                positions: repeatedHeadPositions,
                                rows: rowCount * Self.numQHeads,
                                width: Self.indexHeadDim,
                                ropeDim: 64,
                                inverse: false,
                                config: rope)
            epilogue.encodeIndexerQAT(commandBuffer: cb,
                                      buf: indexQOut,
                                      bufOffset: outputs.indexQOutOffset,
                                      rows: rowCount * Self.numQHeads)
        }

        if let source = indexerPerHeadWeights, let destination = outputs.indexWeightsOut {
            let bytes = rowCount * Self.numQHeads * MemoryLayout<Float>.stride
            if source.buffer !== destination || source.offset != outputs.indexWeightsOutOffset {
                let blit = cb.makeBlitCommandEncoder()
                blit?.copy(from: source.buffer, sourceOffset: source.offset,
                           to: destination, destinationOffset: outputs.indexWeightsOutOffset,
                           size: bytes)
                blit?.endEncoding()
            }
        }
    }

    private func fillPositions(rowCount: Int, startPosition: Int) {
        let posPtr = positions.contents().bindMemory(to: Float.self, capacity: maxRows)
        let repPtr = repeatedHeadPositions.contents().bindMemory(to: Float.self,
                                                                 capacity: maxRows * Self.numQHeads)
        for row in 0..<rowCount {
            let p = Float(startPosition + row)
            posPtr[row] = p
            let base = row * Self.numQHeads
            for head in 0..<Self.numQHeads { repPtr[base + head] = p }
        }
    }

    var qrBuffer: MTLBuffer { qr }
}
