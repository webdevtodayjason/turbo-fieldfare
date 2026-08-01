import Metal

/// FP8 e4m3 embedding lookup for the untied DeepSeek V4-Flash embedding
/// table (V4F-02). Row gather + e4m3 convert + 128x128 block scale + optional
/// `outScale` (pass 1.0 to disable), mirroring `EmbedLookupInt4`.
final class EmbedLookupFP8 {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pipeline = try V4ShaderLibrary().pipeline(
            device: context.device,
            module: "dequant_v4",
            subdirectory: "Metal/Quant",
            name: "embed_lookup_fp8")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                table: MTLBuffer,
                tableOffset: Int = 0,
                scales: MTLBuffer,
                scalesOffset: Int = 0,
                out: MTLBuffer,
                outOffset: Int = 0,
                tokenID: UInt32,
                d: UInt32,
                outScale: Float = 1.0) {
        precondition(d % UInt32(DequantFP8BlockGEMV.blockSize) == 0,
                     "D must be a multiple of \(DequantFP8BlockGEMV.blockSize)")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(table, offset: tableOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(out, offset: outOffset, index: 2)
        var token = tokenID
        var dimension = d
        var scale = outScale
        encoder.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(d) + 255) / 256, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
