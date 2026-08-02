import Foundation
import Metal

/// Reusable resources for V4 layer-major chunked prefill.
///
/// All buffers are sized for `maxRows` and are reused across prompt chunks.
/// Element widths are explicit here because the projection stages intentionally
/// mix fp32 accumulation rows with fp16 activation rows.
final class V4ChunkedPrefillState {
    struct Layout: Equatable, Sendable {
        let rows: Int
        let dim: Int
        let streams: Int
        let heads: Int
        let headDim: Int
        let indexHeadDim: Int
        let experts: Int
        let topK: Int
        let ffn: Int

        var tokenIDsBytes: Int { rows * MemoryLayout<UInt32>.stride }
        var streamBytes: Int { rows * streams * dim * MemoryLayout<Float>.stride }
        var branchBytes: Int { rows * dim * MemoryLayout<Float>.stride }
        var xNormBytes: Int { rows * dim * MemoryLayout<Float16>.stride }
        var qBytes: Int { rows * heads * headDim * MemoryLayout<Float16>.stride }
        var indexQBytes: Int { rows * heads * indexHeadDim * MemoryLayout<Float16>.stride }
        var indexWeightsBytes: Int { rows * heads * MemoryLayout<Float>.stride }
        var windowKVBytes: Int { rows * headDim * MemoryLayout<Float16>.stride }
        var compressorBytes: Int { rows * 1024 * MemoryLayout<Float>.stride }
        var indexerCompressorBytes: Int { rows * 256 * MemoryLayout<Float>.stride }
        var attnBytes: Int { qBytes }
        var lowRankBytes: Int { rows * 8192 * MemoryLayout<Float16>.stride }
        var hiddenF16Bytes: Int { rows * dim * MemoryLayout<Float16>.stride }
        var routerLogitsBytes: Int { rows * experts * MemoryLayout<Float>.stride }
        var routeIDsBytes: Int { rows * topK * MemoryLayout<UInt32>.stride }
        var routeWeightsBytes: Int { rows * topK * MemoryLayout<Float>.stride }
        var sharedIntermediateBytes: Int { rows * ffn * MemoryLayout<Float16>.stride }
    }

    let maxRows: Int
    let layout: Layout
    let context: MetalContext
    let boundary: V4PrefillBoundary
    let qkvStage: V4BatchedQKVCompressorPrefillStage
    let proj: V4PrefillProj
    let glue: V4ChunkedPrefillGlue
    let routedMoE: V4GroupedRoutedMoEPrefillAdapter
    let executors: [V4ChunkedPrefillExecutor]

    let tokenIDs: MTLBuffer
    let stream: MTLBuffer
    let branch: MTLBuffer
    let xNorm: MTLBuffer
    let q: MTLBuffer
    let indexQ: MTLBuffer
    let indexWeights: MTLBuffer
    let windowKV: MTLBuffer
    let compressorKV: MTLBuffer
    let compressorGate: MTLBuffer
    let indexerCompressorKV: MTLBuffer
    let indexerCompressorGate: MTLBuffer
    let attn: MTLBuffer
    let lowRank: MTLBuffer
    let oProj: MTLBuffer
    let routerLogits: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let sharedGate: MTLBuffer
    let sharedUp: MTLBuffer
    let sharedAct: MTLBuffer
    let sharedOut: MTLBuffer
    let combinedFFN: MTLBuffer

    init(model: V4Model,
         cache: CompressedKVCacheManager,
         attention: V4Attention,
         maxRows: Int = PrefillRuntimeConfig.maxChunkTokens) throws {
        precondition(maxRows > 0 && maxRows <= PrefillRuntimeConfig.maxChunkTokens)
        self.maxRows = maxRows
        let cfg = model.config
        self.layout = Layout(rows: maxRows,
                             dim: cfg.hiddenSize,
                             streams: cfg.hcMult,
                             heads: cfg.numHeads,
                             headDim: cfg.headDim,
                             indexHeadDim: V4Attention.indexHeadDim,
                             experts: cfg.numExperts,
                             topK: cfg.topKExperts,
                             ffn: cfg.moeIntermediateSize)
        let context = try V4ShaderLibrary.context(for: model.device)
        self.context = context
        self.boundary = try V4PrefillBoundary(device: model.device, maxRows: maxRows)
        self.qkvStage = try V4BatchedQKVCompressorPrefillStage(device: model.device,
                                                               maxRows: maxRows)
        self.proj = try V4PrefillProj(device: model.device)
        self.glue = try V4ChunkedPrefillGlue(device: model.device)
        self.routedMoE = try V4GroupedRoutedMoEPrefillAdapter(context: context)
        self.executors = try (0..<cfg.numLayers).map { _ in
            try V4ChunkedPrefillExecutor(device: model.device,
                                         cache: cache,
                                         attention: attention)
        }

        func make(_ bytes: Int, _ label: String, zero: Bool = true) throws -> MTLBuffer {
            guard let buffer = model.device.makeBuffer(length: max(1, bytes),
                                                       options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            if zero { memset(buffer.contents(), 0, buffer.length) }
            return buffer
        }

        tokenIDs = try make(layout.tokenIDsBytes, "v4prefill.tokenIDs")
        stream = try make(layout.streamBytes, "v4prefill.stream")
        branch = try make(layout.branchBytes, "v4prefill.branch")
        xNorm = try make(layout.xNormBytes, "v4prefill.xNorm")
        q = try make(layout.qBytes, "v4prefill.q")
        indexQ = try make(layout.indexQBytes, "v4prefill.indexQ")
        indexWeights = try make(layout.indexWeightsBytes, "v4prefill.indexWeights")
        windowKV = try make(layout.windowKVBytes, "v4prefill.windowKV")
        compressorKV = try make(layout.compressorBytes, "v4prefill.compressorKV")
        compressorGate = try make(layout.compressorBytes, "v4prefill.compressorGate")
        indexerCompressorKV = try make(layout.indexerCompressorBytes,
                                      "v4prefill.indexerCompressorKV")
        indexerCompressorGate = try make(layout.indexerCompressorBytes,
                                        "v4prefill.indexerCompressorGate")
        attn = try make(layout.attnBytes, "v4prefill.attn")
        lowRank = try make(layout.lowRankBytes, "v4prefill.lowRank")
        oProj = try make(layout.hiddenF16Bytes, "v4prefill.oProj")
        routerLogits = try make(layout.routerLogitsBytes, "v4prefill.routerLogits")
        routeIDs = try make(layout.routeIDsBytes, "v4prefill.routeIDs")
        routeWeights = try make(layout.routeWeightsBytes, "v4prefill.routeWeights")
        sharedGate = try make(layout.sharedIntermediateBytes, "v4prefill.sharedGate")
        sharedUp = try make(layout.sharedIntermediateBytes, "v4prefill.sharedUp")
        sharedAct = try make(layout.sharedIntermediateBytes, "v4prefill.sharedAct")
        sharedOut = try make(layout.hiddenF16Bytes, "v4prefill.sharedOut")
        combinedFFN = try make(layout.hiddenF16Bytes, "v4prefill.combinedFFN")
    }

    func upload(tokens: ArraySlice<Int32>) {
        precondition(tokens.count <= maxRows)
        let ptr = tokenIDs.contents().assumingMemoryBound(to: UInt32.self)
        for (row, token) in tokens.enumerated() {
            ptr[row] = UInt32(bitPattern: token)
        }
    }
}
