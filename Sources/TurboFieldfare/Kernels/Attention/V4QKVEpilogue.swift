import Foundation
import Metal

/// Q/KV LoRA projection path for DeepSeek V4-Flash decode (V4F-03,
/// work-order item 2.8). Composes the committed FP8 block GEMV (V4F-02) with
/// the new `attention_v4b` boundary kernels:
///
/// **Q path** (recon §2):
///   `qr = RMSNorm(wq_a(x))` — wq_a FP8 [1024, 4096], learned gamma;
///   `q  = wq_b(qr)` — wq_b FP8 [64*512, 1024];
///   weight-free per-head RMS renorm over each 512-dim head (recon note #6);
///   trailing-64 partial RoPE at the token position (`V4RoPE`).
///
/// **Window KV path**: `kv = RMSNorm(wkv(x))` — wkv FP8 [512, 4096] — with
/// trailing-slice RoPE, written directly into the cache manager's window
/// ring slot (GEMV -> RMSNorm -> RoPE all in place on the slot: a clearly
/// safe epilogue fusion, one buffer, no scratch round-trip).
///
/// **Compressor feed**: wkv/wgate fp32 GEMVs (`v4b_gemv_f32`) producing the
/// group's projection rows for the CSA/HCA compressor kernels. CSA layers
/// use out-dim 1024, HCA 512.
///
/// **Indexer feed** (optional, CSA only): `index_q = wq_b_index(qr)` FP8
/// [64*128, 1024] with trailing-slice RoPE. The reference's Hadamard + FP4
/// QAT sim is NOT applied here (write-side concern of the indexer store;
/// recorded as a gap in the V4F-03 report).
final class V4QKVEpilogue {
    static let dim = 4096
    static let qLoraRank = 1024
    static let numQHeads = 64
    static let headDim = 512
    static let indexHeadDim = 128

    /// Per-layer projection weights. FP8 tensors are the V4F-02 layout:
    /// `[M, N]` e4m3 codes plus a ue8m0 grid of `[ceil(M/128), N/128]`.
    struct Weights {
        var wqA: (codes: MTLBuffer, scales: MTLBuffer)      // [1024, 4096]
        var wqB: (codes: MTLBuffer, scales: MTLBuffer)      // [32768, 1024]
        var qNormGamma: MTLBuffer                           // [1024] fp32
        var windowWKV: (codes: MTLBuffer, scales: MTLBuffer) // [512, 4096]
        var kvNormGamma: MTLBuffer                          // [512] fp32
        /// Compressor projections (fp32 [compOutDim, 4096]); nil on
        /// ratio-0 layers.
        var compressorWKV: MTLBuffer?
        var compressorWGate: MTLBuffer?
        var compressorOutDim: Int = 0
        /// Indexer up-projection FP8 [64*128, 1024]; nil on non-CSA layers.
        var indexerWqB: (codes: MTLBuffer, scales: MTLBuffer)?
    }

    /// Where the token's window KV lands (from
    /// `CompressedKVCacheManager.windowSlot(layer:position:)`).
    struct WindowSlot {
        var buffer: MTLBuffer
        var offset: Int
    }

    private let fp8PSO: MTLComputePipelineState
    private let rmsnormPSO: MTLComputePipelineState
    private let renormPSO: MTLComputePipelineState
    private let ropePSO: MTLComputePipelineState
    private let gemvF32PSO: MTLComputePipelineState

    /// Latent scratch: RMS-normalized qr [1024] fp16 (also feeds the
    /// indexer up-projection, per the shared-`qr` reference structure).
    private let qr: MTLBuffer

    init(device: MTLDevice) throws {
        let library = V4ShaderLibrary()
        self.fp8PSO = try library.pipeline(device: device,
                                           module: "dequant_v4",
                                           subdirectory: "Metal/Quant",
                                           name: "dequant_fp8_e4m3_gemv_simd")
        self.rmsnormPSO = try library.pipeline(device: device,
                                               module: "attention_v4b",
                                               subdirectory: "Metal/Attention",
                                               name: "v4b_rmsnorm")
        self.renormPSO = try library.pipeline(device: device,
                                              module: "attention_v4b",
                                              subdirectory: "Metal/Attention",
                                              name: "v4b_perhead_renorm")
        self.ropePSO = try library.pipeline(device: device,
                                            module: "attention_v4b",
                                            subdirectory: "Metal/Attention",
                                            name: "v4b_rope_trailing")
        self.gemvF32PSO = try library.pipeline(device: device,
                                               module: "attention_v4b",
                                               subdirectory: "Metal/Attention",
                                               name: "v4b_gemv_f32")
        guard let qr = device.makeBuffer(
            length: Self.qLoraRank * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.qr = qr
    }

    /// Encode one decode token's Q/KV projections. Everything lands in one
    /// command buffer; Metal hazard-tracking orders the stages.
    ///
    /// - `x`: hidden state [4096] fp16 (the mHC pre-gather output, normed).
    /// - `position`: absolute token position (RoPE phase).
    /// - `qOut`: [64, 512] fp16 queries, renormed + RoPE'd.
    /// - `windowSlot`: window-ring write target; nil to skip the window KV.
    /// - `compressorWKVOut`/`compressorWGateOut`: fp32 [compressorOutDim]
    ///   projection outputs; required when the weights carry compressor
    ///   tensors.
    /// - `indexQOut`: [64, 128] fp16 indexer queries (CSA layers only).
    func encodeDecode(commandBuffer cb: MTLCommandBuffer,
                      x: MTLBuffer, xOffset: Int = 0,
                      position: Int,
                      weights: Weights,
                      rope: V4RoPE.Config,
                      qOut: MTLBuffer, qOutOffset: Int = 0,
                      windowSlot: WindowSlot? = nil,
                      compressorWKVOut: MTLBuffer? = nil,
                      compressorWGateOut: MTLBuffer? = nil,
                      indexQOut: MTLBuffer? = nil,
                      normEps: Float = 1e-6) {
        precondition(xOffset % 8 == 0, "FP8 GEMV x reads need 8-aligned xOffset")
        // Q: wq_a -> qr scratch, RMSNorm in place, wq_b -> qOut, renorm, RoPE.
        encodeFP8GEMV(commandBuffer: cb,
                      codes: weights.wqA.codes, scales: weights.wqA.scales,
                      x: x, xOffset: xOffset,
                      y: qr, yOffset: 0,
                      m: Self.qLoraRank, n: Self.dim)
        encodeRMSNorm(commandBuffer: cb,
                      buf: qr, gamma: weights.qNormGamma,
                      n: Self.qLoraRank, eps: normEps, useGamma: true)
        encodeFP8GEMV(commandBuffer: cb,
                      codes: weights.wqB.codes, scales: weights.wqB.scales,
                      x: qr, xOffset: 0,
                      y: qOut, yOffset: qOutOffset,
                      m: Self.numQHeads * Self.headDim, n: Self.qLoraRank)
        encodePerHeadRenorm(commandBuffer: cb,
                            buf: qOut, bufOffset: qOutOffset,
                            heads: Self.numQHeads, headDim: Self.headDim, eps: normEps)
        encodeRoPE(commandBuffer: cb,
                   buf: qOut, bufOffset: qOutOffset,
                   rows: Self.numQHeads, width: Self.headDim,
                   position: Float(position), inverse: false, config: rope)

        // Window KV: straight into the ring slot, normed + RoPE'd in place.
        if let slot = windowSlot {
            encodeFP8GEMV(commandBuffer: cb,
                          codes: weights.windowWKV.codes,
                          scales: weights.windowWKV.scales,
                          x: x, xOffset: xOffset,
                          y: slot.buffer, yOffset: slot.offset,
                          m: Self.headDim, n: Self.dim)
            encodeRMSNorm(commandBuffer: cb,
                          buf: slot.buffer, bufOffset: slot.offset,
                          gamma: weights.kvNormGamma,
                          n: Self.headDim, eps: normEps, useGamma: true)
            encodeRoPE(commandBuffer: cb,
                       buf: slot.buffer, bufOffset: slot.offset,
                       rows: 1, width: Self.headDim,
                       position: Float(position), inverse: false, config: rope)
        }

        // Compressor feed (fp32 projections).
        if let wkv = weights.compressorWKV, let wgate = weights.compressorWGate {
            precondition(weights.compressorOutDim > 0 &&
                         compressorWKVOut != nil && compressorWGateOut != nil,
                         "compressor weights require outputs and an out-dim")
            encodeF32GEMV(commandBuffer: cb,
                          w: wkv, x: x, xOffset: xOffset,
                          y: compressorWKVOut!,
                          m: weights.compressorOutDim, n: Self.dim)
            encodeF32GEMV(commandBuffer: cb,
                          w: wgate, x: x, xOffset: xOffset,
                          y: compressorWGateOut!,
                          m: weights.compressorOutDim, n: Self.dim)
        }

        // Indexer queries from the shared qr (no weight-free renorm).
        if let indexWqB = weights.indexerWqB, let indexQOut {
            encodeFP8GEMV(commandBuffer: cb,
                          codes: indexWqB.codes, scales: indexWqB.scales,
                          x: qr, xOffset: 0,
                          y: indexQOut, yOffset: 0,
                          m: Self.numQHeads * Self.indexHeadDim,
                          n: Self.qLoraRank)
            encodeRoPE(commandBuffer: cb,
                       buf: indexQOut, bufOffset: 0,
                       rows: Self.numQHeads, width: Self.indexHeadDim,
                       position: Float(position), inverse: false, config: rope)
        }
    }

    // MARK: - Stage encoders

    private func encodeFP8GEMV(commandBuffer cb: MTLCommandBuffer,
                               codes: MTLBuffer, scales: MTLBuffer,
                               x: MTLBuffer, xOffset: Int,
                               y: MTLBuffer, yOffset: Int,
                               m: Int, n: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(fp8PSO)
        enc.setBuffer(codes, offset: 0, index: 0)
        enc.setBuffer(scales, offset: 0, index: 1)
        enc.setBuffer(x, offset: xOffset, index: 2)
        enc.setBuffer(y, offset: yOffset, index: 3)
        var mv = UInt32(m)
        var nv = UInt32(n)
        enc.setBytes(&mv, length: 4, index: 4)
        enc.setBytes(&nv, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// In-place RMSNorm (`buf` read and written).
    private func encodeRMSNorm(commandBuffer cb: MTLCommandBuffer,
                               buf: MTLBuffer, bufOffset: Int = 0,
                               gamma: MTLBuffer,
                               n: Int, eps: Float, useGamma: Bool) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(rmsnormPSO)
        enc.setBuffer(buf, offset: bufOffset, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(buf, offset: bufOffset, index: 2)
        var nv = UInt32(n)
        var e = eps
        var ug = UInt32(useGamma ? 1 : 0)
        enc.setBytes(&nv, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.setBytes(&ug, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Weight-free per-head RMS renorm, in place (internal for tests).
    func encodePerHeadRenorm(commandBuffer cb: MTLCommandBuffer,
                             buf: MTLBuffer, bufOffset: Int = 0,
                             heads: Int, headDim: Int, eps: Float) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(renormPSO)
        enc.setBuffer(buf, offset: bufOffset, index: 0)
        var hd = UInt32(headDim)
        var e = eps
        enc.setBytes(&hd, length: 4, index: 1)
        enc.setBytes(&e, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: heads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeRoPE(commandBuffer cb: MTLCommandBuffer,
                            buf: MTLBuffer, bufOffset: Int,
                            rows: Int, width: Int,
                            position: Float, inverse: Bool,
                            config: V4RoPE.Config) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(ropePSO)
        enc.setBuffer(buf, offset: bufOffset, index: 0)
        var r = UInt32(rows)
        var w = UInt32(width)
        var rd = UInt32(64)
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
        let total = rows * 32
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    private func encodeF32GEMV(commandBuffer cb: MTLCommandBuffer,
                               w: MTLBuffer,
                               x: MTLBuffer, xOffset: Int,
                               y: MTLBuffer,
                               m: Int, n: Int) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(gemvF32PSO)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(x, offset: xOffset, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var mv = UInt32(m)
        var nv = UInt32(n)
        enc.setBytes(&mv, length: 4, index: 3)
        enc.setBytes(&nv, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (m + 7) / 8, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Read back the latent qr scratch (tests only; syncs on the CB).
    var qrBuffer: MTLBuffer { qr }
}
