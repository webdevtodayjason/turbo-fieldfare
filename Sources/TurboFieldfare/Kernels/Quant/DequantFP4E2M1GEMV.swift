import Metal

/// FP4 e2m1 + ue8m0 matrix-vector multiplication for DeepSeek V4-Flash
/// routed-expert weights (V4F-02).
///
/// Layout: `weights` is `[M, N/2]` packed e2m1 pairs (low nibble = element
/// 2i), `scales` is `[M, N/32]` ue8m0 bytes (one power-of-two scale per 32
/// elements along K). No biases. Eight SIMD groups process eight output rows
/// per threadgroup, one SIMD group per row — the
/// `dequant_int4_gemv_simd` geometry with aligned `uint` weight loads, which
/// the 4-byte sub-tensor padding contract of the V4 repack guarantees.
final class DequantFP4E2M1GEMV {
    /// FP4 microscaling group: elements per ue8m0 scale along K.
    static let groupSize = 32

    private static let rowsPerThreadgroup = 8

    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try V4ShaderLibrary().pipeline(
            device: context.device,
            module: "dequant_v4",
            subdirectory: "Metal/Quant",
            name: "dequant_fp4_e2m1_gemv_simd")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer,
                weightsOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                x: MTLBuffer,
                xOffset: Int = 0,
                y: MTLBuffer,
                yOffset: Int = 0,
                m: UInt32,
                n: UInt32) {
        precondition(n % UInt32(Self.groupSize) == 0,
                     "N must be a multiple of \(Self.groupSize)")
        // The kernel reads packed weights through an aligned `uint*`; the V4
        // repack pads every sub-tensor to 4 bytes.
        precondition(weightsOffset % 4 == 0,
                     "dequant_fp4_e2m1_gemv_simd needs a 4-aligned weightsOffset, got \(weightsOffset)")
        // x is read as half4 pairs; activation offsets must stay 8-byte aligned.
        precondition(xOffset % 8 == 0,
                     "dequant_fp4_e2m1_gemv_simd needs an 8-aligned xOffset, got \(xOffset)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(x, offset: xOffset, index: 2)
        encoder.setBuffer(y, offset: yOffset, index: 3)
        var mValue = m
        var nValue = n
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 5)

        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(m) + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Self.rowsPerThreadgroup,
                                           height: 1, depth: 1))
        encoder.endEncoding()
    }
}
