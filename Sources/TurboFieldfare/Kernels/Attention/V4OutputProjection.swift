import Foundation
import Metal

/// Grouped output projection for DeepSeek V4-Flash (V4F-03, work-order item
/// 2.7). Composition of two GEMVs:
///
/// 1. **Down (grouped)**: the attention output `o` [64*512 = 32768] fp16 is
///    viewed as 8 groups of 4096; `wo_a` (bf16, dequantized at convert time
///    per recon §2) is viewed as [8, 1024, 4096] and projects each group to
///    `o_lora_rank` 1024 -> `lowRank` [8192] fp16 (`v4b_grouped_gemv_bf16`).
/// 2. **Up (summed)**: `wo_b` (FP8 128x128-block, [4096, 8192]) sums the
///    group low-rank outputs into hidden 4096 (the existing
///    `dequant_fp8_e4m3_gemv_simd` from V4F-02).
///
/// No epilogue fusion is taken: the output de-rotation runs before this
/// projection (`V4RoPE`, per reference order) and the mHC post-merge runs
/// after it (`V4HyperConnections.encodePost`); both are already fused into
/// their own kernels, so there is nothing clearly safe left to fold in here.
final class V4OutputProjection {
    static let groups = 8
    static let groupDim = 4096          // (64 * 512) / 8
    static let loraRank = 1024
    static let hidden = 4096

    private let groupedPSO: MTLComputePipelineState
    private let fp8PSO: MTLComputePipelineState
    /// Grouped-stage scratch: [8 * 1024] fp16 (hazard-tracked in one CB).
    private let lowRank: MTLBuffer

    init(device: MTLDevice) throws {
        let library = V4ShaderLibrary()
        self.groupedPSO = try library.pipeline(device: device,
                                               module: "attention_v4b",
                                               subdirectory: "Metal/Attention",
                                               name: "v4b_grouped_gemv_bf16")
        self.fp8PSO = try library.pipeline(device: device,
                                           module: "dequant_v4",
                                           subdirectory: "Metal/Quant",
                                           name: "dequant_fp8_e4m3_gemv_simd")
        guard let scratch = device.makeBuffer(
            length: Self.groups * Self.loraRank * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.lowRank = scratch
    }

    /// Encode the full projection: `o` [32768] fp16 -> `out` [4096] fp16.
    ///
    /// - `woA`: [8*1024, 4096] bf16 (row-major, group-major).
    /// - `woBWeights`: [4096, 8192] e4m3; `woBScales`: [32, 64] ue8m0 grid.
    func encode(commandBuffer cb: MTLCommandBuffer,
                o: MTLBuffer, oOffset: Int = 0,
                woA: MTLBuffer, woAOffset: Int = 0,
                woBWeights: MTLBuffer, woBWeightsOffset: Int = 0,
                woBScales: MTLBuffer, woBScalesOffset: Int = 0,
                out: MTLBuffer, outOffset: Int = 0) {
        precondition(oOffset % 8 == 0, "grouped GEMV x reads need 8-aligned oOffset")
        precondition(woBWeightsOffset % 4 == 0, "FP8 GEMV needs 4-aligned weights")
        encodeGrouped(commandBuffer: cb,
                      o: o, oOffset: oOffset,
                      woA: woA, woAOffset: woAOffset)
        encodeUp(commandBuffer: cb,
                 woBWeights: woBWeights, woBWeightsOffset: woBWeightsOffset,
                 woBScales: woBScales, woBScalesOffset: woBScalesOffset,
                 out: out, outOffset: outOffset)
    }

    /// Stage 1 exposed for tests: `o` -> internal low-rank scratch.
    func encodeGrouped(commandBuffer cb: MTLCommandBuffer,
                       o: MTLBuffer, oOffset: Int = 0,
                       woA: MTLBuffer, woAOffset: Int = 0) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(groupedPSO)
        enc.setBuffer(woA, offset: woAOffset, index: 0)
        enc.setBuffer(o, offset: oOffset, index: 1)
        enc.setBuffer(lowRank, offset: 0, index: 2)
        var m = UInt32(Self.groups * Self.loraRank)
        var n = UInt32(Self.groupDim)
        var rpg = UInt32(Self.loraRank)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&rpg, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(m) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Stage 2 exposed for tests: internal low-rank scratch -> `out`.
    func encodeUp(commandBuffer cb: MTLCommandBuffer,
                  woBWeights: MTLBuffer, woBWeightsOffset: Int = 0,
                  woBScales: MTLBuffer, woBScalesOffset: Int = 0,
                  out: MTLBuffer, outOffset: Int = 0) {
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(fp8PSO)
        enc.setBuffer(woBWeights, offset: woBWeightsOffset, index: 0)
        enc.setBuffer(woBScales, offset: woBScalesOffset, index: 1)
        enc.setBuffer(lowRank, offset: 0, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var m = UInt32(Self.hidden)
        var n = UInt32(Self.groups * Self.loraRank)
        enc.setBytes(&m, length: 4, index: 4)
        enc.setBytes(&n, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (Int(m) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Read back the grouped-stage scratch (tests only; syncs on the CB).
    var lowRankBuffer: MTLBuffer { lowRank }
}
