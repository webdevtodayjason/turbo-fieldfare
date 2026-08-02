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
/// Create one executor instance per layer. All executors may share one
/// `V4CompressorAccumulatorStore`; the store keeps independent CSA/HCA state
/// for every layer so prefill and decode can continue the same partial group.
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
    private let accumulators: V4CompressorAccumulatorStore

    init(device: MTLDevice, cache: CompressedKVCacheManager, attention: V4Attention,
         hca: V4HCACompressor? = nil, glue: V4DecodeGlue? = nil,
         accumulators: V4CompressorAccumulatorStore? = nil) throws {
        let cacheConfig = cache.config
        self.cache = cache; self.attention = attention; self.config = cacheConfig
        self.hca = try hca ?? V4HCACompressor(device: device)
        self.glue = try glue ?? V4DecodeGlue(context: V4ShaderLibrary.context(for: device))
        self.accumulators = try accumulators ?? V4CompressorAccumulatorStore(
            device: device,
            layerKinds: cacheConfig.compressRatios.indices.map { cacheConfig.kind(layer: $0) })
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
            let state = accumulators.csa(layer: layer)
            copy(cb, i.compressorKV!, kv, state.curKV, r * 1024 * 4, 1024 * 4); copy(cb, i.compressorGate!, gate, state.curGate, r * 1024 * 4, 1024 * 4)
            copy(cb, i.indexerCompressorKV!, i.indexerCompressorKVOffset + row * 256 * 4, state.idxCurKV, r * 256 * 4, 256 * 4)
            copy(cb, i.indexerCompressorGate!, i.indexerCompressorGateOffset + row * 256 * 4, state.idxCurGate, r * 256 * 4, 256 * 4)
        } else {
            let state = accumulators.hca(layer: layer)
            copy(cb, i.compressorKV!, kv, state.ringKV, r * 512 * 4, 512 * 4); copy(cb, i.compressorGate!, gate, state.ringGate, r * 512 * 4, 512 * 4)
        }
    }

    private func flush(_ cb: MTLCommandBuffer, _ layer: Int, _ kind: V4LayerKind, _ pos: Int, _ w: CompressorWeights, _ rope: V4RoPE.Config, _ eps: Float) {
        let group = cache.groupIndex(layer: layer, tokenPosition: pos), slot = cache.compressedSlot(layer: layer, group: group)
        let ropePos = UInt32(cache.ropePosition(layer: layer, group: group))
        if kind == .csa {
            let state = accumulators.csa(layer: layer)
            attention.encodeCSACompressGroup(commandBuffer: cb, prevKV: state.prevKV, curKV: state.curKV, prevGate: state.prevGate, curGate: state.curGate,
                                             ape: w.ape, apeOffset: w.apeOffset, gamma: w.gamma, gammaOffset: w.gammaOffset,
                                             outValues: slot.values.buffer, valuesOffset: slot.values.offset, outScales: slot.scales.buffer, scalesOffset: slot.scales.offset,
                                             outRope: slot.rope.buffer, ropeOffset: slot.rope.offset, ropePosition: ropePos, ropeTheta: rope.theta,
                                             yarnFactor: rope.yarnFactor, originalSeqLen: rope.originalSeqLen, betaFast: rope.betaFast, betaSlow: rope.betaSlow,
                                             useYarn: rope.useYarn, normEps: eps)
            let idx = cache.indexerSlot(layer: layer, group: group)
            glue.encodeIndexerCompressGroup(commandBuffer: cb, prevKV: state.idxPrevKV, curKV: state.idxCurKV, prevGate: state.idxPrevGate, curGate: state.idxCurGate,
                                            ape: w.indexerAPE!, apeOffset: w.indexerAPEOffset, gamma: w.indexerGamma!, gammaOffset: w.indexerGammaOffset,
                                            out: idx.buffer, outOffset: idx.offset, ropePosition: ropePos, rope: rope, normEps: eps)
            accumulators.rollCSA(commandBuffer: cb, layer: layer)
        } else if kind == .hca {
            let state = accumulators.hca(layer: layer)
            hca.encodeGroup(commandBuffer: cb, kv: state.ringKV, gate: state.ringGate, ape: w.ape, apeOffset: w.apeOffset, gamma: w.gamma, gammaOffset: w.gammaOffset,
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
