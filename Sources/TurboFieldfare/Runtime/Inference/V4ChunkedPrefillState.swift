import Foundation
import Metal

/// Reusable resources for V4 layer-major chunked prefill.
///
/// All buffers are sized for `maxRows` and are reused across prompt chunks.
/// Element widths are explicit because the projection stages intentionally mix
/// fp32 accumulation rows with fp16 activation rows.
internal final class V4ChunkedPrefillState {
    internal struct Layout: Equatable, Sendable {
        internal let rows: Int
        internal let dim: Int
        internal let ffn: Int
        internal let experts: Int
        internal let layers: Int

        internal static let streams = 4
        internal static let heads = 64
        internal static let headDim = 512
        internal static let indexHeadDim = 128
        internal static let csaCompressorDim = 1024
        internal static let indexerCompressorDim = 256
        internal static let lowRankDim = 8192
        internal static let topK = 6

        internal init(rows: Int, dim: Int, ffn: Int, experts: Int, layers: Int) {
            precondition(rows > 0 && rows <= 128)
            self.rows = rows
            self.dim = dim
            self.ffn = ffn
            self.experts = experts
            self.layers = layers
        }

        internal var tokenIDsBytes: Int { rows * MemoryLayout<UInt32>.stride }
        internal var streamBytes: Int { rows * Self.streams * dim * MemoryLayout<Float>.stride }
        internal var branchBytes: Int { rows * dim * MemoryLayout<Float>.stride }
        internal var xNormBytes: Int { rows * dim * MemoryLayout<Float16>.stride }
        internal var qBytes: Int { rows * Self.heads * Self.headDim * MemoryLayout<Float16>.stride }
        internal var indexQBytes: Int { rows * Self.heads * Self.indexHeadDim * MemoryLayout<Float16>.stride }
        internal var indexWeightsBytes: Int { rows * Self.heads * MemoryLayout<Float>.stride }
        internal var positionsBytes: Int { rows * MemoryLayout<Float>.stride }
        internal var repeatedHeadPositionsBytes: Int { rows * Self.heads * MemoryLayout<Float>.stride }
        internal var windowKVBytes: Int { rows * Self.headDim * MemoryLayout<Float16>.stride }
        internal var compressorKVBytes: Int { rows * Self.csaCompressorDim * MemoryLayout<Float>.stride }
        internal var compressorGateBytes: Int { rows * Self.csaCompressorDim * MemoryLayout<Float>.stride }
        internal var indexerCompressorKVBytes: Int { rows * Self.indexerCompressorDim * MemoryLayout<Float>.stride }
        internal var indexerCompressorGateBytes: Int { rows * Self.indexerCompressorDim * MemoryLayout<Float>.stride }
        internal var attnBytes: Int { rows * Self.heads * Self.headDim * MemoryLayout<Float16>.stride }
        internal var lowRankBytes: Int { rows * Self.lowRankDim * MemoryLayout<Float16>.stride }
        internal var oProjBytes: Int { rows * dim * MemoryLayout<Float16>.stride }
        internal var routerLogitsBytes: Int { rows * experts * MemoryLayout<Float>.stride }
        internal var routeIDsBytes: Int { rows * Self.topK * MemoryLayout<UInt32>.stride }
        internal var routeWeightsBytes: Int { rows * Self.topK * MemoryLayout<Float>.stride }
        internal var sharedGateBytes: Int { rows * ffn * MemoryLayout<Float16>.stride }
        internal var sharedUpBytes: Int { rows * ffn * MemoryLayout<Float16>.stride }
        internal var sharedActBytes: Int { rows * ffn * MemoryLayout<Float16>.stride }
        internal var sharedOutBytes: Int { rows * dim * MemoryLayout<Float16>.stride }
        internal var routedPlusSharedBytes: Int { rows * dim * MemoryLayout<Float16>.stride }
    }

    internal let model: V4Model
    internal let cache: CompressedKVCacheManager
    internal let attention: V4Attention
    internal let maxRows: Int
    internal let layout: Layout

    internal let context: MetalContext
    internal let queue: MTLCommandQueue
    internal let boundary: V4PrefillBoundary
    internal let qkvStage: V4BatchedQKVCompressorPrefillStage
    internal let proj: V4PrefillProj
    internal let glue: V4ChunkedPrefillGlue
    internal let routedMoE: V4GroupedRoutedMoEPrefillAdapter
    internal let executors: [V4ChunkedPrefillExecutor]

    internal let tokenIDs: MTLBuffer              // UInt32 [T]
    internal let stream: MTLBuffer                // fp32 [T, 4, D]
    internal let branch: MTLBuffer                // fp32 [T, D]
    internal let xNorm: MTLBuffer                 // fp16 [T, D]
    internal let q: MTLBuffer                     // fp16 [T, 64, 512]
    internal let indexQ: MTLBuffer                // fp16 [T, 64, 128]
    internal let indexWeights: MTLBuffer          // fp32 [T, 64]
    internal let positions: MTLBuffer             // fp32 [T]
    internal let repeatedHeadPositions: MTLBuffer // fp32 [T, 64]
    internal let windowKV: MTLBuffer              // fp16 [T, 512]
    internal let compressorKV: MTLBuffer          // fp32 [T, 1024]
    internal let compressorGate: MTLBuffer        // fp32 [T, 1024]
    internal let indexerCompressorKV: MTLBuffer   // fp32 [T, 256]
    internal let indexerCompressorGate: MTLBuffer // fp32 [T, 256]
    internal let attn: MTLBuffer                  // fp16 [T, 64, 512]
    internal let lowRank: MTLBuffer               // fp16 [T, 8192]
    internal let oProj: MTLBuffer                 // fp16 [T, D]
    internal let routerLogits: MTLBuffer          // fp32 [T, numExperts]
    internal let routeIDs: MTLBuffer              // UInt32 [T, 6]
    internal let routeWeights: MTLBuffer          // fp32 [T, 6]
    internal let sharedGate: MTLBuffer            // fp16 [T, F]
    internal let sharedUp: MTLBuffer              // fp16 [T, F]
    internal let sharedAct: MTLBuffer             // fp16 [T, F]
    internal let sharedOut: MTLBuffer             // fp16 [T, D]
    internal let routedPlusShared: MTLBuffer      // fp16 [T, D]

    internal init(model: V4Model,
                  cache: CompressedKVCacheManager,
                  attention: V4Attention,
                  compressorAccumulators: V4CompressorAccumulatorStore,
                  maxRows: Int = 128) throws {
        precondition(maxRows > 0 && maxRows <= 128)
        self.model = model
        self.cache = cache
        self.attention = attention
        self.maxRows = maxRows

        let cfg = model.config
        self.layout = Layout(rows: maxRows,
                             dim: cfg.hiddenSize,
                             ffn: cfg.moeIntermediateSize,
                             experts: cfg.numExperts,
                             layers: cfg.numLayers)

        let context = try V4ShaderLibrary.context(for: model.device)
        self.context = context
        self.queue = context.queue
        self.boundary = try V4PrefillBoundary(device: model.device, maxRows: maxRows)
        self.qkvStage = try V4BatchedQKVCompressorPrefillStage(device: model.device, maxRows: maxRows)
        self.proj = try V4PrefillProj(device: model.device)
        self.glue = try V4ChunkedPrefillGlue(device: model.device)
        self.routedMoE = try V4GroupedRoutedMoEPrefillAdapter(context: context)
        self.executors = try (0..<cfg.numLayers).map { _ in
            try V4ChunkedPrefillExecutor(device: model.device,
                                         cache: cache,
                                         attention: attention,
                                         accumulators: compressorAccumulators)
        }

        self.tokenIDs = try Self.makeUInt32Buffer(device: model.device, count: maxRows, label: "v4prefill.tokenIDs")
        self.stream = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.streams * cfg.hiddenSize, label: "v4prefill.stream")
        self.branch = try Self.makeFloatBuffer(device: model.device, count: maxRows * cfg.hiddenSize, label: "v4prefill.branch")
        self.xNorm = try Self.makeFloat16Buffer(device: model.device, count: maxRows * cfg.hiddenSize, label: "v4prefill.xNorm")
        self.q = try Self.makeFloat16Buffer(device: model.device, count: maxRows * Layout.heads * Layout.headDim, label: "v4prefill.q")
        self.indexQ = try Self.makeFloat16Buffer(device: model.device, count: maxRows * Layout.heads * Layout.indexHeadDim, label: "v4prefill.indexQ")
        self.indexWeights = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.heads, label: "v4prefill.indexWeights")
        self.positions = try Self.makeFloatBuffer(device: model.device, count: maxRows, label: "v4prefill.positions")
        self.repeatedHeadPositions = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.heads, label: "v4prefill.repeatedHeadPositions")
        self.windowKV = try Self.makeFloat16Buffer(device: model.device, count: maxRows * Layout.headDim, label: "v4prefill.windowKV")
        self.compressorKV = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.csaCompressorDim, label: "v4prefill.compressorKV")
        self.compressorGate = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.csaCompressorDim, label: "v4prefill.compressorGate")
        self.indexerCompressorKV = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.indexerCompressorDim, label: "v4prefill.indexerCompressorKV")
        self.indexerCompressorGate = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.indexerCompressorDim, label: "v4prefill.indexerCompressorGate")
        self.attn = try Self.makeFloat16Buffer(device: model.device, count: maxRows * Layout.heads * Layout.headDim, label: "v4prefill.attn")
        self.lowRank = try Self.makeFloat16Buffer(device: model.device, count: maxRows * Layout.lowRankDim, label: "v4prefill.lowRank")
        self.oProj = try Self.makeFloat16Buffer(device: model.device, count: maxRows * cfg.hiddenSize, label: "v4prefill.oProj")
        self.routerLogits = try Self.makeFloatBuffer(device: model.device, count: maxRows * cfg.numExperts, label: "v4prefill.routerLogits")
        self.routeIDs = try Self.makeUInt32Buffer(device: model.device, count: maxRows * Layout.topK, label: "v4prefill.routeIDs")
        self.routeWeights = try Self.makeFloatBuffer(device: model.device, count: maxRows * Layout.topK, label: "v4prefill.routeWeights")
        self.sharedGate = try Self.makeFloat16Buffer(device: model.device, count: maxRows * cfg.moeIntermediateSize, label: "v4prefill.sharedGate")
        self.sharedUp = try Self.makeFloat16Buffer(device: model.device, count: maxRows * cfg.moeIntermediateSize, label: "v4prefill.sharedUp")
        self.sharedAct = try Self.makeFloat16Buffer(device: model.device, count: maxRows * cfg.moeIntermediateSize, label: "v4prefill.sharedAct")
        self.sharedOut = try Self.makeFloat16Buffer(device: model.device, count: maxRows * cfg.hiddenSize, label: "v4prefill.sharedOut")
        self.routedPlusShared = try Self.makeFloat16Buffer(device: model.device, count: maxRows * cfg.hiddenSize, label: "v4prefill.routedPlusShared")
    }

    internal func uploadTokenIDs(_ ids: [UInt32]) {
        precondition(ids.count <= maxRows)
        let byteCount = ids.count * MemoryLayout<UInt32>.stride
        tokenIDs.contents().copyMemory(from: ids, byteCount: byteCount)
        if byteCount < tokenIDs.length {
            memset(tokenIDs.contents().advanced(by: byteCount), 0, tokenIDs.length - byteCount)
        }
    }

    internal func fillPositions(startPosition: Int, rowCount: Int) {
        precondition(rowCount <= maxRows)
        let positions = positions.contents().assumingMemoryBound(to: Float.self)
        let repeated = repeatedHeadPositions.contents().assumingMemoryBound(to: Float.self)
        for row in 0..<rowCount {
            let position = Float(startPosition + row)
            positions[row] = position
            for head in 0..<Layout.heads {
                repeated[row * Layout.heads + head] = position
            }
        }
    }

    internal static func makeFloatBuffer(device: MTLDevice, count: Int, label: String) throws -> MTLBuffer {
        try makeBuffer(device: device, bytes: count * MemoryLayout<Float>.stride, label: label)
    }

    internal static func makeFloat16Buffer(device: MTLDevice, count: Int, label: String) throws -> MTLBuffer {
        try makeBuffer(device: device, bytes: count * MemoryLayout<Float16>.stride, label: label)
    }

    internal static func makeUInt32Buffer(device: MTLDevice, count: Int, label: String) throws -> MTLBuffer {
        try makeBuffer(device: device, bytes: count * MemoryLayout<UInt32>.stride, label: label)
    }

    private static func makeBuffer(device: MTLDevice, bytes: Int, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        buffer.label = label
        memset(buffer.contents(), 0, bytes)
        return buffer
    }
}
