import Foundation
import Metal

/// mHC (manifold Hyper-Connections) layer-boundary kernels for DeepSeek
/// V4-Flash (V4F-03, work-order item 2.6). Wraps `v4b_hc_params`,
/// `v4b_hc_pre`, and `v4b_hc_post` in `attention_v4b.metal`.
///
/// The residual stream is 4 x `dim` fp32. Per sublayer boundary:
///
/// 1. `encodeParams` derives the dynamic 4x4 mixing parameters from the
///    RMS-normalized flattened state: one 24-wide projection, sigmoid clamps
///    (`pre = sigmoid + eps`, `post = 2 * sigmoid`), and the eps-biased
///    Sinkhorn on `comb` (1 row-softmax + 20 column norms + 19 row norms —
///    recon note #10, exact ordering).
/// 2. `encodePre` gathers the branch input `y = sum_j pre[j] * x[j]`
///    (the caller then applies the sublayer's RMSNorm and the sublayer).
/// 3. `encodePost` merges: `out[k] = post[k] * sublayer + sum_j comb[j,k] *
///    residual[j]` — the comb gather is by COLUMN (recon note #10).
///
/// `encodeHeadParams` covers the pre-only `ParallelHead.hc_head` variant
/// (4-wide fn, single scale): `y = sum pre * x` then the final RMSNorm.
final class V4HyperConnections {
    static let hcMult = 4
    /// pre[4] | post[4] | comb[16] fp32.
    static let paramsFloats = 24

    private let device: MTLDevice
    private let paramsPSO: MTLComputePipelineState
    private let prePSO: MTLComputePipelineState
    private let postPSO: MTLComputePipelineState

    /// Scratch holding the latest `encodeParams` output (hazard-tracked in
    /// the same command buffer; read back in tests only).
    let paramsBuffer: MTLBuffer

    init(device: MTLDevice) throws {
        self.device = device
        let library = V4ShaderLibrary()
        self.paramsPSO = try library.pipeline(device: device,
                                              module: "attention_v4b",
                                              subdirectory: "Metal/Attention",
                                              name: "v4b_hc_params")
        self.prePSO = try library.pipeline(device: device,
                                           module: "attention_v4b",
                                           subdirectory: "Metal/Attention",
                                           name: "v4b_hc_pre")
        self.postPSO = try library.pipeline(device: device,
                                            module: "attention_v4b",
                                            subdirectory: "Metal/Attention",
                                            name: "v4b_hc_post")
        guard let params = device.makeBuffer(
            length: Self.paramsFloats * MemoryLayout<Float>.size,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.paramsBuffer = params
    }

    /// Dynamic parameters for one sublayer boundary.
    ///
    /// - `x`: flattened stream, [4 * dim] fp32.
    /// - `hcFn`: [24, 4 * dim] fp32 (`hc_fn`).
    /// - `hcBase`: [24] fp32 (`hc_base`).
    /// - `hcScale`: [3] fp32 (`hc_scale`).
    func encodeParams(commandBuffer cb: MTLCommandBuffer,
                      x: MTLBuffer, xOffset: Int = 0,
                      hcFn: MTLBuffer, hcFnOffset: Int = 0,
                      hcBase: MTLBuffer, hcBaseOffset: Int = 0,
                      hcScale: MTLBuffer, hcScaleOffset: Int = 0,
                      dim: Int,
                      normEps: Float = 1e-6,
                      hcEps: Float = 1e-6) {
        encodeParamsImpl(commandBuffer: cb,
                         x: x, xOffset: xOffset,
                         hcFn: hcFn, hcFnOffset: hcFnOffset,
                         hcBase: hcBase, hcBaseOffset: hcBaseOffset,
                         hcScale: hcScale, hcScaleOffset: hcScaleOffset,
                         dim: dim, mixCount: 24,
                         normEps: normEps, hcEps: hcEps)
    }

    /// Pre-only head variant (`ParallelHead.hc_head`): `hcFn` is [4, 4*dim],
    /// `hcBase` [4], `hcScale` [1]. Only `pre` (paramsBuffer[0..4)) is
    /// written; follow with `encodePre` and the final RMSNorm.
    func encodeHeadParams(commandBuffer cb: MTLCommandBuffer,
                          x: MTLBuffer, xOffset: Int = 0,
                          hcFn: MTLBuffer, hcFnOffset: Int = 0,
                          hcBase: MTLBuffer, hcBaseOffset: Int = 0,
                          hcScale: MTLBuffer, hcScaleOffset: Int = 0,
                          dim: Int,
                          normEps: Float = 1e-6,
                          hcEps: Float = 1e-6) {
        encodeParamsImpl(commandBuffer: cb,
                         x: x, xOffset: xOffset,
                         hcFn: hcFn, hcFnOffset: hcFnOffset,
                         hcBase: hcBase, hcBaseOffset: hcBaseOffset,
                         hcScale: hcScale, hcScaleOffset: hcScaleOffset,
                         dim: dim, mixCount: 4,
                         normEps: normEps, hcEps: hcEps)
    }

    private func encodeParamsImpl(commandBuffer cb: MTLCommandBuffer,
                                  x: MTLBuffer, xOffset: Int,
                                  hcFn: MTLBuffer, hcFnOffset: Int,
                                  hcBase: MTLBuffer, hcBaseOffset: Int,
                                  hcScale: MTLBuffer, hcScaleOffset: Int,
                                  dim: Int, mixCount: Int,
                                  normEps: Float, hcEps: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(paramsPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(hcFn, offset: hcFnOffset, index: 1)
        enc.setBuffer(hcBase, offset: hcBaseOffset, index: 2)
        enc.setBuffer(hcScale, offset: hcScaleOffset, index: 3)
        enc.setBuffer(paramsBuffer, offset: 0, index: 4)
        var d = UInt32(dim)
        var mc = UInt32(mixCount)
        var ne = normEps
        var he = hcEps
        enc.setBytes(&d, length: 4, index: 5)
        enc.setBytes(&mc, length: 4, index: 6)
        enc.setBytes(&ne, length: 4, index: 7)
        enc.setBytes(&he, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 768, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Branch input gather: `y[d] = sum_j pre[j] * x[j*dim + d]` (fp32).
    /// Consumes the params produced by the latest `encodeParams`/`encodeHeadParams`.
    func encodePre(commandBuffer cb: MTLCommandBuffer,
                   x: MTLBuffer, xOffset: Int = 0,
                   out: MTLBuffer, outOffset: Int = 0,
                   dim: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(prePSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(paramsBuffer, offset: 0, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim)
        enc.setBytes(&d, length: 4, index: 3)
        let tg = min(256, Int(prePSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Boundary merge: `out[k*dim+d] = post[k] * sublayer[d] +
    /// sum_j comb[j,k] * residual[j*dim+d]`. `out` may alias `residual`
    /// (per-thread disjoint slots make the in-place stream update safe).
    func encodePost(commandBuffer cb: MTLCommandBuffer,
                    residual: MTLBuffer, residualOffset: Int = 0,
                    sublayer: MTLBuffer, sublayerOffset: Int = 0,
                    out: MTLBuffer, outOffset: Int = 0,
                    dim: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(postPSO)
        enc.setBuffer(residual, offset: residualOffset, index: 0)
        enc.setBuffer(sublayer, offset: sublayerOffset, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var d = UInt32(dim)
        enc.setBytes(&d, length: 4, index: 4)
        let tg = min(256, Int(postPSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }
}
