import Foundation
import Metal

/// Trailing-slice partial RoPE for DeepSeek V4-Flash (V4F-03, work-order
/// item 2.5). Wraps `v4b_rope_trailing` in `attention_v4b.metal`.
///
/// Only the trailing `ropeDim` (64) channels of each row rotate, as
/// slice-internal NeoX pairs `(width - ropeDim + i, width - ropeDim/2 + i)`.
/// One kernel covers every rotation site in the decode path:
///
/// - **q / window KV**: forward rotation at the token position.
/// - **compressed entries**: forward rotation at the group-start position
///   (the caller passes `CompressedKVCacheManager.ropePosition`; the CSA/HCA
///   compressor kernels already fold this in — this entry point is for
///   standalone/passthrough writes).
/// - **attention output de-rotation**: `inverse: true` applies the complex
///   conjugate at the query position (`apply_rotary_emb(..., inverse=True)`
///   in the reference — an approximation the model was trained with; recon
///   note #4 says do not "fix" it).
///
/// Per-layer frequency config (recon note #5): ratio-4/128 layers use
/// `compress_rope_theta` 160000 + YaRN (factor 16) for the *whole* layer
/// including the window branch; ratio-0 layers (0/1/42) use plain theta
/// 10000 with YaRN disabled.
final class V4RoPE {
    /// Frequency-table configuration for one layer.
    struct Config: Sendable, Equatable {
        var theta: Float
        var yarnFactor: Float
        var originalSeqLen: Float
        var betaFast: Float
        var betaSlow: Float
        var useYarn: Bool

        /// CSA/HCA layers: compress theta + YaRN on the entire layer.
        static let compressedLayer = Config(theta: 160_000,
                                            yarnFactor: 16,
                                            originalSeqLen: 65_536,
                                            betaFast: 32,
                                            betaSlow: 1,
                                            useYarn: true)
        /// Ratio-0 layers (0/1/42): base theta, YaRN disabled.
        static let passthroughLayer = Config(theta: 10_000,
                                             yarnFactor: 16,
                                             originalSeqLen: 65_536,
                                             betaFast: 32,
                                             betaSlow: 1,
                                             useYarn: false)

        /// Config for a layer kind from `V4CacheConfig.kind(layer:)`.
        static func forLayer(_ kind: V4LayerKind) -> Config {
            kind == .passthrough ? .passthroughLayer : .compressedLayer
        }
    }

    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.pipeline = try V4ShaderLibrary().pipeline(
            device: device,
            module: "attention_v4b",
            subdirectory: "Metal/Attention",
            name: "v4b_rope_trailing")
    }

    /// Rotate (or de-rotate) the trailing rope slice of `x` in place.
    ///
    /// - `x`: [rows, width] FP16.
    /// - `position`: rotation position as a float — signed/fractional values
    ///   are legal (the negative-position de-rotation path).
    /// - `inverse`: conjugate rotation (output de-rotation).
    func encode(commandBuffer cb: MTLCommandBuffer,
                x: MTLBuffer, xOffset: Int = 0,
                rows: Int,
                width: Int = 512,
                ropeDim: Int = 64,
                position: Float,
                inverse: Bool = false,
                config: Config = .compressedLayer) {
        precondition(rows > 0 && width > ropeDim && ropeDim % 2 == 0)
        precondition(xOffset % MemoryLayout<Float16>.size == 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(x, offset: xOffset, index: 0)
        var r = UInt32(rows)
        var w = UInt32(width)
        var rd = UInt32(ropeDim)
        var pos = position
        var inv = UInt32(inverse ? 1 : 0)
        var theta = config.theta
        var factor = config.yarnFactor
        var orig = config.originalSeqLen
        var bf = config.betaFast
        var bs = config.betaSlow
        var uy = UInt32(config.useYarn ? 1 : 0)
        enc.setBytes(&r, length: 4, index: 1)
        enc.setBytes(&w, length: 4, index: 2)
        enc.setBytes(&rd, length: 4, index: 3)
        enc.setBytes(&pos, length: 4, index: 4)
        enc.setBytes(&inv, length: 4, index: 5)
        enc.setBytes(&theta, length: 4, index: 6)
        enc.setBytes(&factor, length: 4, index: 7)
        enc.setBytes(&orig, length: 4, index: 8)
        enc.setBytes(&bf, length: 4, index: 9)
        enc.setBytes(&bs, length: 4, index: 10)
        enc.setBytes(&uy, length: 4, index: 11)
        let total = rows * (ropeDim / 2)
        let tg = min(256, Int(pipeline.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }
}
