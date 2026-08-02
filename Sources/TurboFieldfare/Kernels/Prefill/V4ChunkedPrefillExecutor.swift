import Foundation
import Metal

/// Orchestrates V4 chunked prefill for one layer using only existing cache,
/// compressor, and decode kernels. Rows are encoded in causal order into the
/// caller's command buffer: write current window KV, stage/flush completed
/// compressed groups, then decode with tokenCount = startPosition + row + 1.
///
/// The executor does not advance `CompressedKVCacheManager.position`. Layer-major
/// prefill must run the same `[startPosition, startPosition + rowCount)` chunk
/// through every layer, then the caller advances the shared cache cursor once by
/// `rowCount` after all layers complete.
///
/// Create one executor instance per layer. CSA/HCA staging accumulators are held
/// by the executor and must not be shared across layers.
final class V4ChunkedPrefillExecutor {
    struct Inputs {
        let q: MTLBuffer; let qOffset: Int
        let windowKV: MTLBuffer; let windowKVOffset: Int
        let indexQ: MTLBuffer?; let indexQOffset: Int
        let indexWeights: MTLBuffer?; let indexWeightsOffset: Int
        let compressorKV: MTLBuffer?; let compressorKVOffset: Int
        let compressorGate: MTLBuffer?; let compressorGateOffset: Int
        let indexerCompressorKV: MTLBuffer?; let indexerCompressorKVOffset: Int
        let indexerCompressorGate: MTLBuffer?; let indexerCompressorGateOffset: Int
        let sinks: MTLBuffer; let sinksOffset: Int
        let output: MTLBuffer; let outputOffset: Int
        let rowStrideQ: Int, rowStrideWindowKV: Int, rowStrideIndexQ: Int
        let rowStrideIndexWeights: Int, rowStrideCompressor: Int, rowStrideOutput: Int

        init(q: MTLBuffer, qOffset: Int = 0,
             windowKV: MTLBuffer, windowKVOffset: Int = 0,
             indexQ: MTLBuffer? = nil, indexQOffset: Int = 0,
             indexWeights: MTLBuffer? = nil, indexWeightsOffset: Int = 0,
             compressorKV: MTLBuffer? = nil, compressorKVOffset: Int = 0,
             compressorGate: MTLBuffer? = nil, compressorGateOffset: Int = 0,
             indexerCompressorKV: MTLBuffer? = nil, indexerCompressorKVOffset: Int = 0,
             indexerCompressorGate: MTLBuffer? = nil, indexerCompressorGateOffset: Int = 0,
             sinks: MTLBuffer, sinksOffset: Int = 0,
             output: MTLBuffer, outputOffset: Int = 0,
             rowStrideQ: Int = V4Attention.numQHeads * V4Attention.headDim * 2,
             rowStrideWindowKV: Int = V4Attention.headDim * 2,
             rowStrideIndexQ: Int = V4Attention.indexHeads * V4Attention.indexHeadDim * 2,
             rowStrideIndexWeights: Int = V4Attention.indexHeads * 4,
             rowStrideCompressor: Int = 0,
             rowStrideOutput: Int = V4Attention.numQHeads * V4Attention.headDim * 2) {
            self.q = q; self.qOffset = qOffset; self.windowKV = windowKV; self.windowKVOffset = windowKVOffset
            self.indexQ = indexQ; self.indexQOffset = indexQOffset; self.indexWeights = indexWeights; self.indexWeightsOffset = indexWeightsOffset
            self.compressorKV = compressorKV; self.compressorKVOffset = compressorKVOffset; self.compressorGate = compressorGate; self.compressorGateOffset = compressorGateOffset
            self.indexerCompressorKV = indexerCompressorKV; self.indexerCompressorKVOffset = indexerCompressorKVOffset
            self.indexerCompressorGate = indexerCompressorGate; self.indexerCompressorGateOffset = indexerCompressorGateOffset
            self.sinks = sinks; self.sinksOffset = sinksOffset; self.output = output; self.outputOffset = outputOffset
            self.rowStrideQ = rowStrideQ; self.rowStrideWindowKV = rowStrideWindowKV; self.rowStrideIndexQ = rowStrideIndexQ
            self.rowStrideIndexWeights = rowStrideIndexWeights; self.rowStrideCompressor = rowStrideCompressor; self.rowStrideOutput = rowStrideOutput
        }
    }

    struct CompressorWeights {
        let ape: MTLBuffer; let apeOffset: Int
        let gamma: MTLBuffer; let gammaOffset: Int
        let indexerAPE: MTLBuffer?; let indexerAPEOffset: Int
        let indexerGamma: MTLBuffer?; let indexerGammaOffset: Int
        init(ape: MTLBuffer, apeOffset: Int = 0, gamma: MTLBuffer, gammaOffset: Int = 0,
             indexerAPE: MTLBuffer? = nil, indexerAPEOffset: Int = 0,
             indexerGamma: MTLBuffer? = nil, indexerGammaOffset: Int = 0) {
            self.ape = ape; self.apeOffset = apeOffset; self.gamma = gamma; self.gammaOffset = gammaOffset
            self.indexerAPE = indexerAPE; self.indexerAPEOffset = indexerAPEOffset
            self.indexerGamma = indexerGamma; self.indexerGammaOffset = indexerGammaOffset
        }
    }

    private let cache: CompressedKVCacheManager, attention: V4Attention, hca: V4HCACompressor, glue: V4DecodeGlue
    private let config: V4CacheConfig
    private let csaPrevKV, csaCurKV, csaPrevGate, csaCurGate: MTLBuffer
    private let idxPrevKV, idxCurKV, idxPrevGate, idxCurGate: MTLBuffer
    private let hcaKV, hcaGate: MTLBuffer

    init(device: MTLDevice, cache: CompressedKVCacheManager, attention: V4Attention,
         hca: V4HCACompressor? = nil, glue: V4DecodeGlue? = nil) throws {
        self.cache = cache; self.attention = attention; self.config = cache.config
        self.hca = try hca ?? V4HCACompressor(device: device)
        self.glue = try glue ?? V4DecodeGlue(context: V4ShaderLibrary.context(for: device))
        func make(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else { throw ModelError.residentBufferWrapFailed }
            b.label = label
            memset(b.contents(), 0, bytes)
            return b
        }
        csaPrevKV = try make(4 * 1024 * 4, "v4prefill.csa.prevKV"); csaCurKV = try make(4 * 1024 * 4, "v4prefill.csa.curKV")
        csaPrevGate = try make(4 * 1024 * 4, "v4prefill.csa.prevGate"); csaCurGate = try make(4 * 1024 * 4, "v4prefill.csa.curGate")
        idxPrevKV = try make(4 * 256 * 4, "v4prefill.idx.prevKV"); idxCurKV = try make(4 * 256 * 4, "v4prefill.idx.curKV")
        idxPrevGate = try make(4 * 256 * 4, "v4prefill.idx.prevGate"); idxCurGate = try make(4 * 256 * 4, "v4prefill.idx.curGate")
        hcaKV = try make(128 * 512 * 4, "v4prefill.hca.kv"); hcaGate = try make(128 * 512 * 4, "v4prefill.hca.gate")
    }

    func encode(commandBuffer cb: MTLCommandBuffer, layer: Int, startPosition: Int, rowCount: Int,
                inputs: Inputs, compressorWeights: CompressorWeights? = nil,
                rope: V4RoPE.Config = .compressedLayer, normEps: Float = 1e-6) {
        precondition(startPosition == cache.position); precondition(rowCount >= 0)
        let kind = cache.layerKind(layer)
        if kind != .passthrough { precondition(inputs.compressorKV != nil && inputs.compressorGate != nil && compressorWeights != nil && inputs.rowStrideCompressor > 0) }
        if kind == .csa { precondition(inputs.indexQ != nil && inputs.indexWeights != nil && inputs.indexerCompressorKV != nil && inputs.indexerCompressorGate != nil && compressorWeights?.indexerAPE != nil && compressorWeights?.indexerGamma != nil) }
        for row in 0..<rowCount {
            let pos = startPosition + row
            let win = cache.windowSlot(layer: layer, position: pos)
            copy(cb, inputs.windowKV, inputs.windowKVOffset + row * inputs.rowStrideWindowKV, win.buffer, win.offset, config.headDim * 2)
            stageCompressor(cb, layer, kind, pos, row, inputs)
            if kind != .passthrough && cache.completesGroup(layer: layer, tokenPosition: pos) { flush(cb, layer, kind, pos, compressorWeights!, rope, normEps) }
            let tokenCount = pos + 1
            let nVisible = cache.visibleGroupCount(layer: layer, windowStart: max(0, tokenCount - config.window), tokenCount: tokenCount)
            cache.assertDisjointCoverage(layer: layer, groupCount: nVisible, tokenCount: tokenCount)
            decode(cb, layer, kind, tokenCount, nVisible, row, inputs)
        }
    }

    private func stageCompressor(_ cb: MTLCommandBuffer, _ layer: Int, _ kind: V4LayerKind, _ pos: Int, _ row: Int, _ i: Inputs) {
        guard kind != .passthrough else { return }
        let r = pos % config.compressRatio(layer: layer)
        let kv = i.compressorKVOffset + row * i.rowStrideCompressor, gate = i.compressorGateOffset + row * i.rowStrideCompressor
        if kind == .csa {
            copy(cb, i.compressorKV!, kv, csaCurKV, r * 1024 * 4, 1024 * 4); copy(cb, i.compressorGate!, gate, csaCurGate, r * 1024 * 4, 1024 * 4)
            copy(cb, i.indexerCompressorKV!, i.indexerCompressorKVOffset + row * 256 * 4, idxCurKV, r * 256 * 4, 256 * 4)
            copy(cb, i.indexerCompressorGate!, i.indexerCompressorGateOffset + row * 256 * 4, idxCurGate, r * 256 * 4, 256 * 4)
        } else {
            copy(cb, i.compressorKV!, kv, hcaKV, r * 512 * 4, 512 * 4); copy(cb, i.compressorGate!, gate, hcaGate, r * 512 * 4, 512 * 4)
        }
    }

    private func flush(_ cb: MTLCommandBuffer, _ layer: Int, _ kind: V4LayerKind, _ pos: Int, _ w: CompressorWeights, _ rope: V4RoPE.Config, _ eps: Float) {
        let group = cache.groupIndex(layer: layer, tokenPosition: pos), slot = cache.compressedSlot(layer: layer, group: group)
        let ropePos = UInt32(cache.ropePosition(layer: layer, group: group))
        if kind == .csa {
            attention.encodeCSACompressGroup(commandBuffer: cb, prevKV: csaPrevKV, curKV: csaCurKV, prevGate: csaPrevGate, curGate: csaCurGate,
                                             ape: w.ape, apeOffset: w.apeOffset, gamma: w.gamma, gammaOffset: w.gammaOffset,
                                             outValues: slot.values.buffer, valuesOffset: slot.values.offset, outScales: slot.scales.buffer, scalesOffset: slot.scales.offset,
                                             outRope: slot.rope.buffer, ropeOffset: slot.rope.offset, ropePosition: ropePos, ropeTheta: rope.theta,
                                             yarnFactor: rope.yarnFactor, originalSeqLen: rope.originalSeqLen, betaFast: rope.betaFast, betaSlow: rope.betaSlow,
                                             useYarn: rope.useYarn, normEps: eps)
            let idx = cache.indexerSlot(layer: layer, group: group)
            glue.encodeIndexerCompressGroup(commandBuffer: cb, prevKV: idxPrevKV, curKV: idxCurKV, prevGate: idxPrevGate, curGate: idxCurGate,
                                            ape: w.indexerAPE!, apeOffset: w.indexerAPEOffset, gamma: w.indexerGamma!, gammaOffset: w.indexerGammaOffset,
                                            out: idx.buffer, outOffset: idx.offset, ropePosition: ropePos, rope: rope, normEps: eps)
            copy(cb, csaCurKV, 0, csaPrevKV, 0, 4 * 1024 * 4); copy(cb, csaCurGate, 0, csaPrevGate, 0, 4 * 1024 * 4)
            copy(cb, idxCurKV, 0, idxPrevKV, 0, 4 * 256 * 4); copy(cb, idxCurGate, 0, idxPrevGate, 0, 4 * 256 * 4)
        } else if kind == .hca {
            hca.encodeGroup(commandBuffer: cb, kv: hcaKV, gate: hcaGate, ape: w.ape, apeOffset: w.apeOffset, gamma: w.gamma, gammaOffset: w.gammaOffset,
                            outValues: slot.values.buffer, valuesOffset: slot.values.offset, outScales: slot.scales.buffer, scalesOffset: slot.scales.offset,
                            outRope: slot.rope.buffer, ropeOffset: slot.rope.offset, ropePosition: ropePos, rope: rope, normEps: eps)
        }
    }

    private func decode(_ cb: MTLCommandBuffer, _ layer: Int, _ kind: V4LayerKind, _ tokenCount: Int, _ nVisible: Int, _ row: Int, _ i: Inputs) {
        let qOff = i.qOffset + row * i.rowStrideQ, outOff = i.outputOffset + row * i.rowStrideOutput, window = cache.windowBuffer(layer: layer)
        switch kind {
        case .passthrough:
            attention.encodeWindowMQADecode(commandBuffer: cb, q: i.q, qOffset: qOff, windowK: window, tokenCount: tokenCount, sinks: i.sinks, sinksOffset: i.sinksOffset, out: i.output, outOffset: outOff)
        case .csa:
            let base = cache.compressedSlot(layer: layer, group: 0)
            attention.encodeCSADecode(commandBuffer: cb, q: i.q, qOffset: qOff, indexQ: i.indexQ!, indexQOffset: i.indexQOffset + row * i.rowStrideIndexQ,
                                      indexKV: cache.indexerBuffer(layer: layer), indexWeights: i.indexWeights!, indexWeightsOffset: i.indexWeightsOffset + row * i.rowStrideIndexWeights,
                                      nVisible: nVisible, compressedValues: base.values.buffer, compressedScales: base.scales.buffer, compressedRope: base.rope.buffer,
                                      windowK: window, tokenCount: tokenCount, sinks: i.sinks, sinksOffset: i.sinksOffset, out: i.output, outOffset: outOff)
        case .hca:
            let base = cache.compressedSlot(layer: layer, group: 0)
            attention.encodeHCADecode(commandBuffer: cb, q: i.q, qOffset: qOff, nVisible: nVisible, compressedValues: base.values.buffer,
                                      compressedScales: base.scales.buffer, compressedRope: base.rope.buffer, windowK: window,
                                      tokenCount: tokenCount, sinks: i.sinks, sinksOffset: i.sinksOffset, out: i.output, outOffset: outOff)
        }
    }

    private func copy(_ cb: MTLCommandBuffer, _ src: MTLBuffer, _ srcOff: Int, _ dst: MTLBuffer, _ dstOff: Int, _ bytes: Int) {
        guard let blit = cb.makeBlitCommandEncoder() else { return }
        blit.copy(from: src, sourceOffset: srcOff, to: dst, destinationOffset: dstOff, size: bytes); blit.endEncoding()
    }
}
