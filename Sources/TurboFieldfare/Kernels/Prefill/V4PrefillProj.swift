import Foundation
import Metal

/// Batched projection / window-attention kernels for DeepSeek V4-Flash
/// chunked prefill (V4F-06c, work-order A2). Wraps `prefill_proj_v4.metal`:
///
/// - `encodeFP8GEMM`: batched FP8 e4m3 + ue8m0 128x128 block GEMM,
///   `y[r, m] = sum_n W[m, n] * x[r, n]`. Covers wq_a (M=1024, N=4096),
///   wq_b (M=32768, N=1024), wkv (M=512, N=4096), indexer wq_b (M=8192,
///   N=1024), and the o-proj up stage wo_b (M=4096, N=8192).
/// - `encodeBF16GEMM`: batched BF16 GEMM for the compressor projections
///   (M=1024 CSA / 512 HCA, N=4096) and the router gate (M=256, N=4096).
/// - `encodeWindowRingWrite`: lands chunk KV rows at ring slots
///   `(startPosition + i) % window`, matching
///   `CompressedKVCacheManager.windowSlot` addressing.
/// - `encodeWindowMQAPrefill`: batched causal window MQA, 64 heads x 512,
///   per-head sinks in the denominator only; row i attends
///   `ring[0..<prefixCount] + chunkKV[0...i]`.
/// - `encodeGroupedOProjDown` / `encodeOProjUp`: the 8-group o-projection
///   (group slices, NOT one flat GEMM — design-note pitfall 676731b).
///
/// Module registration note: `prefill_proj_v4` is registered in
/// `MetalContext.shaderModules` at integration time; until then pipelines
/// fall back to a standalone compile of the module from the bundle, so this
/// wrapper works in both states.
final class V4PrefillProj {
    static let numQHeads = 64
    static let headDim = 512
    static let window = 128
    static let threadsPerGroup = 256

    /// O-projection geometry (matches V4OutputProjection).
    static let oProjGroups = 8
    static let oProjGroupDim = 4096        // (64 * 512) / 8
    static let oProjLoraRank = 1024
    static let oProjHidden = 4096

    /// Attention scale applied after the QK dot (512^-0.5, matches decode).
    static let softmaxScale: Float = 1.0 / Float(512).squareRoot()

    /// Activation-column tile staged per GEMM iteration (kernel constant).
    static let nChunk = 2048

    private let device: MTLDevice
    private let fp8PSO: MTLComputePipelineState
    private let bf16PSO: MTLComputePipelineState
    private let ringWritePSO: MTLComputePipelineState
    private let attnPSO: MTLComputePipelineState
    private let groupedPSO: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.device = device
        let resolve: (String) throws -> MTLComputePipelineState = { name in
            if let pso = try? V4ShaderLibrary().pipeline(
                device: device,
                module: "prefill_proj_v4",
                subdirectory: "Metal/Prefill",
                name: name) {
                return pso
            }
            // Pre-integration fallback: the module is not yet in
            // `MetalContext.shaderModules`; compile it standalone from the
            // bundle resource.
            guard let url = Bundle.module.url(forResource: "prefill_proj_v4",
                                              withExtension: "metal",
                                              subdirectory: "Metal/Prefill") else {
                throw MetalError.missingShaderResource("prefill_proj_v4")
            }
            let src = try String(contentsOf: url, encoding: .utf8)
            let opts = MTLCompileOptions()
            opts.languageVersion = .version4_0
            let library: MTLLibrary
            do {
                library = try device.makeLibrary(source: src, options: opts)
            } catch {
                throw MetalError.libraryCompileFailed("\(error)")
            }
            guard let fn = library.makeFunction(name: name) else {
                throw MetalError.missingFunction(name)
            }
            return try device.makeComputePipelineState(function: fn)
        }
        self.fp8PSO = try resolve("v4pp_fp8_block_gemm")
        self.bf16PSO = try resolve("v4pp_bf16_gemm")
        self.ringWritePSO = try resolve("v4pp_window_ring_write")
        self.attnPSO = try resolve("v4pp_window_mqa_prefill")
        self.groupedPSO = try resolve("v4pp_fp8_grouped_gemm")
    }

    // MARK: - 1. Batched FP8 block GEMM

    /// `out[r, m] = sum_n W[m, n] * x[r, n]` for `rows` activation rows.
    ///
    /// - `weights`: [M, N] e4m3 row-major; `scales`: [ceil(M/128), N/128]
    ///   ue8m0 grid row-major (the `dequant_v4` layout).
    /// - `x`: [rows, N] fp16.
    /// - `out`: [rows, M]; fp32 when `outFP16 == false`, fp16 otherwise.
    /// - Requires `n % 128 == 0`, `m % 8 == 0`, and 4-byte-aligned weight
    ///   offsets (uchar4 fast loads).
    func encodeFP8GEMM(commandBuffer cb: MTLCommandBuffer,
                       weights: MTLBuffer, weightsOffset: Int = 0,
                       scales: MTLBuffer, scalesOffset: Int = 0,
                       x: MTLBuffer, xOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       rows: Int, m: Int, n: Int,
                       outFP16: Bool) {
        precondition(rows > 0 && m > 0 && n > 0)
        precondition(n % 128 == 0, "FP8 block GEMM requires N % 128 == 0")
        precondition(m % 8 == 0, "FP8 block GEMM requires M % 8 == 0")
        precondition(weightsOffset % 4 == 0, "FP8 GEMM needs 4-aligned weights")
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(fp8PSO)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(x, offset: xOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var mm = UInt32(m)
        var nn = UInt32(n)
        var fp16 = UInt32(outFP16 ? 1 : 0)
        enc.setBytes(&mm, length: 4, index: 5)
        enc.setBytes(&nn, length: 4, index: 6)
        enc.setBytes(&fp16, length: 4, index: 7)
        let tg = min(Self.threadsPerGroup, Int(fp8PSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(
            MTLSize(width: rows * (m / 8), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - 2. Batched BF16 GEMM

    /// `out[r, m] = sum_n W[m, n] * x[r, n]` over bf16 weights (compressor
    /// projections, router gate). Same buffer contract as `encodeFP8GEMM`
    /// minus the scale grid; fp32 output is the intended mode.
    /// Requires `n % 4 == 0`, `m % 8 == 0`, 4-byte-aligned weight offsets.
    func encodeBF16GEMM(commandBuffer cb: MTLCommandBuffer,
                        weights: MTLBuffer, weightsOffset: Int = 0,
                        x: MTLBuffer, xOffset: Int = 0,
                        out: MTLBuffer, outOffset: Int = 0,
                        rows: Int, m: Int, n: Int,
                        outFP16: Bool) {
        precondition(rows > 0 && m > 0 && n > 0)
        precondition(n % 4 == 0, "BF16 GEMM requires N % 4 == 0")
        precondition(m % 8 == 0, "BF16 GEMM requires M % 8 == 0")
        precondition(weightsOffset % 4 == 0, "BF16 GEMM needs 4-aligned weights")
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(bf16PSO)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(x, offset: xOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var mm = UInt32(m)
        var nn = UInt32(n)
        var fp16 = UInt32(outFP16 ? 1 : 0)
        enc.setBytes(&mm, length: 4, index: 4)
        enc.setBytes(&nn, length: 4, index: 5)
        enc.setBytes(&fp16, length: 4, index: 6)
        let tg = min(Self.threadsPerGroup, Int(bf16PSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(
            MTLSize(width: rows * (m / 8), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - 3. Batched window ring write

    /// Write `rows` KV rows into the window ring: row i lands at slot
    /// `(startPosition + i) % window`, matching
    /// `CompressedKVCacheManager.windowPhysicalSlot`.
    ///
    /// - `kv`: [rows, headDim] fp16 staged chunk KV.
    /// - `ring`: [window, headDim] fp16 layer ring (`windowBuffer(layer:)`).
    /// - `startPosition`: absolute position of chunk row 0 (the cache
    ///   manager's position at chunk start).
    func encodeWindowRingWrite(commandBuffer cb: MTLCommandBuffer,
                               kv: MTLBuffer, kvOffset: Int = 0,
                               ring: MTLBuffer, ringOffset: Int = 0,
                               startPosition: Int, rows: Int,
                               headDim: Int = 512,
                               window: Int = 128) {
        precondition(rows > 0 && headDim > 0 && window > 0)
        precondition(startPosition >= 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(ringWritePSO)
        enc.setBuffer(kv, offset: kvOffset, index: 0)
        enc.setBuffer(ring, offset: ringOffset, index: 1)
        var sp = UInt32(startPosition)
        var r = UInt32(rows)
        var hd = UInt32(headDim)
        var w = UInt32(window)
        enc.setBytes(&sp, length: 4, index: 2)
        enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.setBytes(&w, length: 4, index: 5)
        let total = rows * headDim
        let tg = min(Self.threadsPerGroup, Int(ringWritePSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - 3. Batched causal window MQA prefill attention

    /// Causal window MQA over one chunk: row i attends exactly
    /// `ring[0..<prefixCount]` (tokens flushed by earlier chunks, at ring
    /// slots 0..prefixCount-1) followed by `chunkKV[0...i]` (previous chunk
    /// rows plus itself). Per-head sinks enter the denominator only, matching
    /// `v4_sink_combine`. FP16 in/out.
    ///
    /// - `q`: [rows, 64, 512] fp16 (post per-head norms + RoPE).
    /// - `windowK`: [window, 512] fp16 ring; the first `prefixCount` slots
    ///   are the visible prefix.
    /// - `chunkKV`: [rows, 512] fp16 staged chunk KV (shared K == V).
    /// - `sinks`: [64] fp32 per-head sink logits.
    /// - `out`: [rows, 64, 512] fp16.
    func encodeWindowMQAPrefill(commandBuffer cb: MTLCommandBuffer,
                                q: MTLBuffer, qOffset: Int = 0,
                                windowK: MTLBuffer, windowKOffset: Int = 0,
                                prefixCount: Int,
                                chunkKV: MTLBuffer, chunkKVOffset: Int = 0,
                                rows: Int,
                                sinks: MTLBuffer, sinksOffset: Int = 0,
                                out: MTLBuffer, outOffset: Int = 0) {
        precondition(rows > 0)
        precondition(prefixCount >= 0 && prefixCount <= Self.window)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(attnPSO)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(windowK, offset: windowKOffset, index: 1)
        enc.setBuffer(chunkKV, offset: chunkKVOffset, index: 2)
        enc.setBuffer(sinks, offset: sinksOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var p = UInt32(prefixCount)
        var sc = Self.softmaxScale
        enc.setBytes(&p, length: 4, index: 5)
        enc.setBytes(&sc, length: 4, index: 6)
        let tg = min(Self.threadsPerGroup, Int(attnPSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(
            MTLSize(width: rows * Self.numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    // MARK: - 4. Batched grouped o-projection

    /// Stage 1 (grouped down): for each row,
    /// `lowRank[r, g*1024 + i] = dot(woA[g*1024 + i, :], attn[r, g*4096 .. +4096])`.
    /// The 8 groups are separate slices of the input, NOT one flat GEMM.
    ///
    /// - `attn`: [rows, 32768] fp16 attention output (post de-rotation).
    /// - `woAWeights`: [8192, 4096] e4m3 group-major; `woAScales`: [64, 32]
    ///   ue8m0 grid.
    /// - `lowRank`: [rows, 8192] fp16 output (feeds `encodeOProjUp`).
    func encodeGroupedOProjDown(commandBuffer cb: MTLCommandBuffer,
                                attn: MTLBuffer, attnOffset: Int = 0,
                                woAWeights: MTLBuffer, woAWeightsOffset: Int = 0,
                                woAScales: MTLBuffer, woAScalesOffset: Int = 0,
                                rows: Int,
                                lowRank: MTLBuffer, lowRankOffset: Int = 0) {
        precondition(rows > 0)
        precondition(woAWeightsOffset % 4 == 0, "FP8 grouped GEMM needs 4-aligned weights")
        let m = Self.oProjGroups * Self.oProjLoraRank
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(groupedPSO)
        enc.setBuffer(woAWeights, offset: woAWeightsOffset, index: 0)
        enc.setBuffer(woAScales, offset: woAScalesOffset, index: 1)
        enc.setBuffer(attn, offset: attnOffset, index: 2)
        enc.setBuffer(lowRank, offset: lowRankOffset, index: 3)
        var mm = UInt32(m)
        var dd = UInt32(Self.oProjGroupDim)
        var rr = UInt32(Self.oProjLoraRank)
        enc.setBytes(&mm, length: 4, index: 4)
        enc.setBytes(&dd, length: 4, index: 5)
        enc.setBytes(&rr, length: 4, index: 6)
        let tg = min(Self.threadsPerGroup, Int(groupedPSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreadgroups(
            MTLSize(width: rows * (m / 8), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Stage 2 (up): `out[r, :] = wo_b x lowRank[r, :]` via the batched FP8
    /// block GEMM. `woBWeights`: [4096, 8192] e4m3; `woBScales`: [32, 64]
    /// ue8m0 grid; `lowRank`: [rows, 8192] fp16; `out`: [rows, 4096] fp16.
    func encodeOProjUp(commandBuffer cb: MTLCommandBuffer,
                       lowRank: MTLBuffer, lowRankOffset: Int = 0,
                       woBWeights: MTLBuffer, woBWeightsOffset: Int = 0,
                       woBScales: MTLBuffer, woBScalesOffset: Int = 0,
                       rows: Int,
                       out: MTLBuffer, outOffset: Int = 0) {
        encodeFP8GEMM(commandBuffer: cb,
                      weights: woBWeights, weightsOffset: woBWeightsOffset,
                      scales: woBScales, scalesOffset: woBScalesOffset,
                      x: lowRank, xOffset: lowRankOffset,
                      out: out, outOffset: outOffset,
                      rows: rows,
                      m: Self.oProjHidden,
                      n: Self.oProjGroups * Self.oProjLoraRank,
                      outFP16: true)
    }
}
