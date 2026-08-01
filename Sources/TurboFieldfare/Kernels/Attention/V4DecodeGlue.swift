import Foundation
import Metal

/// Swift wrapper for the V4F-04 decode-graph glue kernels in
/// `attention_v4c.metal`, plus thin re-bindings of two committed kernels
/// the runner needs outside their owner wrappers (`router_v4_gemv_bf16`
/// from `moe_v4` for indexer/router logits, `v4b_gemv_f32` from
/// `attention_v4b` for the fp32 lm_head). All pipelines come from the
/// shared `MetalContext` library.
///
/// Covers the pieces the committed modules lack: BF16 embed broadcast into
/// the fp32 mHC stream, fp32->fp16 RMSNorm at the mHC branch boundary,
/// shared-expert clamped SwiGLU, indexer per-head weight pre-scale,
/// hash-layer routing weights at the tid2eid indices (recon §6), and the
/// CSA indexer compressed-entry flush.
final class V4DecodeGlue {
    private let embedPSO: MTLComputePipelineState
    private let rmsnormPSO: MTLComputePipelineState
    private let swigluPSO: MTLComputePipelineState
    private let scalePSO: MTLComputePipelineState
    private let hashWeightsPSO: MTLComputePipelineState
    private let indexerCompressPSO: MTLComputePipelineState
    private let bf16GemvPSO: MTLComputePipelineState
    private let gemvF32PSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.embedPSO = try context.pipeline("v4c_embed_broadcast")
        self.rmsnormPSO = try context.pipeline("v4c_rmsnorm_f32in")
        self.swigluPSO = try context.pipeline("v4c_swiglu_act")
        self.scalePSO = try context.pipeline("v4c_scale_f32")
        self.hashWeightsPSO = try context.pipeline("v4c_router_weights_at_indices")
        self.indexerCompressPSO = try context.pipeline("v4c_indexer_compress_group")
        self.bf16GemvPSO = try context.pipeline("router_v4_gemv_bf16")
        self.gemvF32PSO = try context.pipeline("v4b_gemv_f32")
    }

    /// BF16 embedding row broadcast into the fp32 mHC residual stream:
    /// `out[s*dim + i] = table[token*dim + i]` for s in 0..<streams.
    func encodeEmbedBroadcast(commandBuffer cb: MTLCommandBuffer,
                              table: MTLBuffer, tableOffset: Int = 0,
                              out: MTLBuffer,
                              token: UInt32,
                              dim: UInt32,
                              streams: UInt32) {
        var tok = token
        var dimension = dim
        var s = streams
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(embedPSO)
        enc.setBuffer(table, offset: tableOffset, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&tok, length: 4, index: 2)
        enc.setBytes(&dimension, length: 4, index: 3)
        enc.setBytes(&s, length: 4, index: 4)
        let flat = Int(streams * dim)
        let w = min(256, Int(embedPSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: flat, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// RMSNorm with fp32 input + fp32 gamma, fp16 out (mHC branch boundary).
    func encodeRMSNormF32In(commandBuffer cb: MTLCommandBuffer,
                            x: MTLBuffer, xOffset: Int = 0,
                            w: MTLBuffer, wOffset: Int = 0,
                            out: MTLBuffer, outOffset: Int = 0,
                            dim: UInt32,
                            eps: Float) {
        var dimension = dim
        var e = eps
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(rmsnormPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(w, offset: wOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        enc.setBytes(&dimension, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Clamped SwiGLU (shared expert): act = silu(min(gate, limit)) *
    /// clamp(up, +/-limit).
    func encodeSwiGLUAct(commandBuffer cb: MTLCommandBuffer,
                         gate: MTLBuffer,
                         up: MTLBuffer,
                         act: MTLBuffer,
                         n: UInt32,
                         limit: Float) {
        var count = n
        var lim = limit
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(swigluPSO)
        enc.setBuffer(gate, offset: 0, index: 0)
        enc.setBuffer(up, offset: 0, index: 1)
        enc.setBuffer(act, offset: 0, index: 2)
        enc.setBytes(&count, length: 4, index: 3)
        enc.setBytes(&lim, length: 4, index: 4)
        let w = min(256, Int(swigluPSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// buf[i] *= scale (indexer per-head weights pre-scale).
    func encodeScaleF32(commandBuffer cb: MTLCommandBuffer,
                        buf: MTLBuffer, bufOffset: Int = 0,
                        scale: Float,
                        n: UInt32) {
        var sc = scale
        var count = n
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(scalePSO)
        enc.setBuffer(buf, offset: bufOffset, index: 0)
        enc.setBytes(&sc, length: 4, index: 1)
        enc.setBytes(&count, length: 4, index: 2)
        let w = min(256, Int(scalePSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Hash-layer routing weights (recon §6): sqrtsoftplus scores gathered
    /// at the caller-uploaded tid2eid indices, L1-normalized, scaled by
    /// `routeScale`. No bias on hash layers.
    func encodeRouterWeightsAtIndices(commandBuffer cb: MTLCommandBuffer,
                                      logits: MTLBuffer,
                                      indices: MTLBuffer,
                                      outWeights: MTLBuffer,
                                      k: UInt32,
                                      routeScale: Float) {
        var count = k
        var scale = routeScale
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hashWeightsPSO)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(indices, offset: 0, index: 1)
        enc.setBuffer(outWeights, offset: 0, index: 2)
        enc.setBytes(&count, length: 4, index: 3)
        enc.setBytes(&scale, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// BF16-weight GEMV, fp16 x, fp32 out (`router_v4_gemv_bf16` from
    /// `moe_v4`). Used for indexer weights_proj and hash-layer router
    /// logits, which `MoEV4.encodeRouterV4` does not expose standalone.
    /// Requires D % 64 == 0.
    func encodeBF16GEMV(commandBuffer cb: MTLCommandBuffer,
                        weights: MTLBuffer, weightsOffset: Int = 0,
                        x: MTLBuffer,
                        out: MTLBuffer,
                        m: UInt32,
                        d: UInt32) {
        precondition(d % 64 == 0, "router_v4_gemv_bf16 requires D % 64 == 0")
        var rows = m
        var cols = d
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(bf16GemvPSO)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&rows, length: 4, index: 3)
        enc.setBytes(&cols, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(m) + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// FP32-weight GEMV (`v4b_gemv_f32` from `attention_v4b`): fp32 lm_head
    /// and any fp32 projection the QKV epilogue does not own. Requires
    /// N % 128 == 0 and a 16-byte-aligned weights base.
    func encodeGemvF32(commandBuffer cb: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       x: MTLBuffer,
                       out: MTLBuffer, outOffset: Int = 0,
                       m: UInt32,
                       n: UInt32) {
        precondition(n % 128 == 0, "v4b_gemv_f32 requires N % 128 == 0")
        precondition(weightsOffset % 16 == 0, "v4b_gemv_f32 needs a 16-aligned weights base")
        var rows = m
        var cols = n
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(gemvF32PSO)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(x, offset: 0, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        enc.setBytes(&rows, length: 4, index: 3)
        enc.setBytes(&cols, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(m) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// CSA lightning-indexer compressed-entry flush (128-dim overlapped
    /// pooling + RMSNorm + group-start partial RoPE, FP16 out).
    func encodeIndexerCompressGroup(commandBuffer cb: MTLCommandBuffer,
                                    prevKV: MTLBuffer, prevKVOffset: Int = 0,
                                    curKV: MTLBuffer, curKVOffset: Int = 0,
                                    prevGate: MTLBuffer, prevGateOffset: Int = 0,
                                    curGate: MTLBuffer, curGateOffset: Int = 0,
                                    ape: MTLBuffer, apeOffset: Int = 0,
                                    gamma: MTLBuffer, gammaOffset: Int = 0,
                                    out: MTLBuffer, outOffset: Int = 0,
                                    ropePosition: UInt32,
                                    rope: V4RoPE.Config,
                                    normEps: Float) {
        var pos = ropePosition
        var theta = rope.theta
        var factor = rope.yarnFactor
        var orig = rope.originalSeqLen
        var bf = rope.betaFast
        var bs = rope.betaSlow
        var uy = UInt32(rope.useYarn ? 1 : 0)
        var eps = normEps
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(indexerCompressPSO)
        enc.setBuffer(prevKV, offset: prevKVOffset, index: 0)
        enc.setBuffer(curKV, offset: curKVOffset, index: 1)
        enc.setBuffer(prevGate, offset: prevGateOffset, index: 2)
        enc.setBuffer(curGate, offset: curGateOffset, index: 3)
        enc.setBuffer(ape, offset: apeOffset, index: 4)
        enc.setBuffer(gamma, offset: gammaOffset, index: 5)
        enc.setBuffer(out, offset: outOffset, index: 6)
        enc.setBytes(&pos, length: 4, index: 7)
        enc.setBytes(&theta, length: 4, index: 8)
        enc.setBytes(&factor, length: 4, index: 9)
        enc.setBytes(&orig, length: 4, index: 10)
        enc.setBytes(&bf, length: 4, index: 11)
        enc.setBytes(&bs, length: 4, index: 12)
        enc.setBytes(&uy, length: 4, index: 13)
        enc.setBytes(&eps, length: 4, index: 14)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}
