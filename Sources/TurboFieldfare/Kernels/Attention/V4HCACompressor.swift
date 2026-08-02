import Foundation
import Metal

/// HCA compressor for DeepSeek V4-Flash (V4F-03, work-order item 2.3/HCA
/// side). Wraps `v4b_hca_compress_group` in `attention_v4b.metal`, mirroring
/// the committed CSA compressor (`V4Attention.encodeCSACompressGroup`).
///
/// Non-overlapped 128:1 pooling (recon §3): one compressed entry per 128
/// tokens, `x[d] = sum_j softmax_j(wgate[j,d] + ape[j,d]) * wkv[j,d]` over
/// the 128-token group, then RMSNorm(512, gamma), partial RoPE on the
/// trailing 64 dims at the group-start position (compress theta + YaRN), and
/// the split FP8/FP16 quantize of `V4KVLayout`. HCA layers have no indexer.
final class V4HCACompressor {
    static let groupSize = 128
    static let headDim = 512

    private let pipeline: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.pipeline = try V4ShaderLibrary().pipeline(
            device: device,
            module: "attention_v4b",
            subdirectory: "Metal/Attention",
            name: "v4b_hca_compress_group")
    }

    /// Flush one HCA compressed entry (decode group completion, or per-group
    /// in prefill). `kv`/`gate`/`ape` are the group's [128, 512] fp32
    /// projection rows. Writes the split FP8/FP16 entry via the slot offsets
    /// from `CompressedKVCacheManager.compressedSlot`.
    func encodeGroup(commandBuffer cb: MTLCommandBuffer,
                     kv: MTLBuffer, kvOffset: Int = 0,
                     gate: MTLBuffer, gateOffset: Int = 0,
                     ape: MTLBuffer, apeOffset: Int = 0,
                     gamma: MTLBuffer, gammaOffset: Int = 0,
                     outValues: MTLBuffer, valuesOffset: Int,
                     outScales: MTLBuffer, scalesOffset: Int,
                     outRope: MTLBuffer, ropeOffset: Int,
                     ropePosition: UInt32,
                     rope: V4RoPE.Config = .compressedLayer,
                     normEps: Float = 1e-6) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(kv, offset: kvOffset, index: 0)
        enc.setBuffer(gate, offset: gateOffset, index: 1)
        enc.setBuffer(ape, offset: apeOffset, index: 2)
        enc.setBuffer(gamma, offset: gammaOffset, index: 3)
        enc.setBuffer(outValues, offset: valuesOffset, index: 4)
        enc.setBuffer(outScales, offset: scalesOffset, index: 5)
        enc.setBuffer(outRope, offset: ropeOffset, index: 6)
        var pos = ropePosition
        var theta = rope.theta
        var factor = rope.yarnFactor
        var orig = rope.originalSeqLen
        var bf = rope.betaFast
        var bs = rope.betaSlow
        var uy = UInt32(rope.useYarn ? 1 : 0)
        var eps = normEps
        enc.setBytes(&pos, length: 4, index: 7)
        enc.setBytes(&theta, length: 4, index: 8)
        enc.setBytes(&factor, length: 4, index: 9)
        enc.setBytes(&orig, length: 4, index: 10)
        enc.setBytes(&bf, length: 4, index: 11)
        enc.setBytes(&bs, length: 4, index: 12)
        enc.setBytes(&uy, length: 4, index: 13)
        enc.setBytes(&eps, length: 4, index: 14)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
        enc.endEncoding()
    }
}
