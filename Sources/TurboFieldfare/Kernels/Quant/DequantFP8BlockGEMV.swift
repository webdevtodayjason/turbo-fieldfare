import Metal

/// FP8 e4m3 + ue8m0 128x128-block matrix-vector multiplication for DeepSeek
/// V4-Flash dense weights (V4F-02): attention projections, the grouped output
/// projection, and the 129280x4096 lm_head. (The router gate is BF16 and runs
/// through `MoEV4.encodeRouterV4`; the embedding table uses `EmbedLookupFP8`.)
///
/// Layout: `weights` is `[M, N]` e4m3 bytes, `scales` is a ue8m0 byte grid of
/// `[ceil(M/128), N/128]`, row-major over the grid. No biases. Per 128-column
/// block the kernel accumulates an FP32 dot and applies one exact power-of-two
/// scale multiply. Eight SIMD groups process eight rows per threadgroup.
final class DequantFP8BlockGEMV {
    /// Block-scale tile edge; N must be a multiple of it.
    static let blockSize = 128

    private static let rowsPerThreadgroup = 8

    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try V4ShaderLibrary().pipeline(
            device: context.device,
            module: "dequant_v4",
            subdirectory: "Metal/Quant",
            name: "dequant_fp8_e4m3_gemv_simd")
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
        precondition(n % UInt32(Self.blockSize) == 0,
                     "N must be a multiple of \(Self.blockSize)")
        // uchar4 weight loads need a 4-aligned base.
        precondition(weightsOffset % 4 == 0,
                     "dequant_fp8_e4m3_gemv_simd needs a 4-aligned weightsOffset, got \(weightsOffset)")
        precondition(xOffset % 8 == 0,
                     "dequant_fp8_e4m3_gemv_simd needs an 8-aligned xOffset, got \(xOffset)")
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
