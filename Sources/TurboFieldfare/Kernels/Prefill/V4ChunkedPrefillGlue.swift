import Foundation
import Metal

/// Batched glue kernels for V4 chunked prefill.
///
/// This wrapper intentionally lives outside `MetalContext` registration. It first
/// tries `V4ShaderLibrary` and falls back to compiling `prefill_chunked_glue_v4`
/// directly from bundle resources so the API self-compiles before centralized
/// registration is added.
final class V4ChunkedPrefillGlue: @unchecked Sendable {
    static let streamCount = 4
    static let routerExperts = 256
    static let routerTopK = 6
    static let threadsPerGroup = 256

    private let embeddingPSO: MTLComputePipelineState
    private let topKPSO: MTLComputePipelineState
    private let hashWeightsPSO: MTLComputePipelineState
    private let addF16PSO: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.embeddingPSO = try Self.pipeline(device: device,
                                             name: "v4cg_bf16_embedding_gather_broadcast")
        self.topKPSO = try Self.pipeline(device: device,
                                        name: "v4cg_router_top6_sqrtsoftplus")
        self.hashWeightsPSO = try Self.pipeline(device: device,
                                               name: "v4cg_hash_router_weights_sqrtsoftplus")
        self.addF16PSO = try Self.pipeline(device: device,
                                           name: "v4cg_add_f16")
    }

    private static func pipeline(device: MTLDevice, name: String) throws -> MTLComputePipelineState {
        if let pso = try? V4ShaderLibrary().pipeline(device: device,
                                                     module: "prefill_chunked_glue_v4",
                                                     subdirectory: "Metal/Prefill",
                                                     name: name) {
            return pso
        }
        guard let url = Bundle.module.url(forResource: "prefill_chunked_glue_v4",
                                          withExtension: "metal",
                                          subdirectory: "Metal/Prefill") else {
            throw MetalError.missingShaderResource("prefill_chunked_glue_v4")
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

    /// Gather BF16 token embeddings and broadcast each token row into the four
    /// fp32 mHC streams: `out[row, stream, d] = embedding[tokenIDs[row], d]`.
    func encodeBF16EmbeddingGatherBroadcast(commandBuffer cb: MTLCommandBuffer,
                                            embeddings: MTLBuffer,
                                            embeddingsOffset: Int = 0,
                                            tokenIDs: MTLBuffer,
                                            tokenIDsOffset: Int = 0,
                                            out: MTLBuffer,
                                            outOffset: Int = 0,
                                            rows: Int,
                                            dim: Int) {
        precondition(rows > 0 && dim > 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(embeddingPSO)
        enc.setBuffer(embeddings, offset: embeddingsOffset, index: 0)
        enc.setBuffer(tokenIDs, offset: tokenIDsOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var r = UInt32(rows)
        var d = UInt32(dim)
        enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        dispatch1D(enc, pso: embeddingPSO, threads: rows * dim)
        enc.endEncoding()
    }

    /// Select top-6 experts from fp32 logits with `staticBias` applied only to
    /// the selection key. Weights are un-biased sqrtsoftplus scores normalized
    /// over the selected IDs and multiplied by `routeScale`.
    func encodeRouterTop6(commandBuffer cb: MTLCommandBuffer,
                          logits: MTLBuffer,
                          logitsOffset: Int = 0,
                          staticBias: MTLBuffer,
                          staticBiasOffset: Int = 0,
                          outIDs: MTLBuffer,
                          outIDsOffset: Int = 0,
                          outWeights: MTLBuffer,
                          outWeightsOffset: Int = 0,
                          rows: Int,
                          routeScale: Float) {
        precondition(rows > 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(topKPSO)
        enc.setBuffer(logits, offset: logitsOffset, index: 0)
        enc.setBuffer(staticBias, offset: staticBiasOffset, index: 1)
        enc.setBuffer(outIDs, offset: outIDsOffset, index: 2)
        enc.setBuffer(outWeights, offset: outWeightsOffset, index: 3)
        var r = UInt32(rows)
        var scale = routeScale
        enc.setBytes(&r, length: 4, index: 4)
        enc.setBytes(&scale, length: 4, index: 5)
        dispatch1D(enc, pso: topKPSO, threads: rows)
        enc.endEncoding()
    }

    /// For hash routing, gather sqrtsoftplus logits at caller-provided `[T,6]`
    /// expert IDs, normalize those six scores, and multiply by `routeScale`.
    func encodeHashRouterWeights(commandBuffer cb: MTLCommandBuffer,
                                 logits: MTLBuffer,
                                 logitsOffset: Int = 0,
                                 tid2eid: MTLBuffer,
                                 tid2eidOffset: Int = 0,
                                 outWeights: MTLBuffer,
                                 outWeightsOffset: Int = 0,
                                 rows: Int,
                                 routeScale: Float) {
        precondition(rows > 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(hashWeightsPSO)
        enc.setBuffer(logits, offset: logitsOffset, index: 0)
        enc.setBuffer(tid2eid, offset: tid2eidOffset, index: 1)
        enc.setBuffer(outWeights, offset: outWeightsOffset, index: 2)
        var r = UInt32(rows)
        var scale = routeScale
        enc.setBytes(&r, length: 4, index: 3)
        enc.setBytes(&scale, length: 4, index: 4)
        dispatch1D(enc, pso: hashWeightsPSO, threads: rows)
        enc.endEncoding()
    }

    /// Elementwise fp16 addition: `out[i] = a[i] + b[i]` for `count` elements.
    func encodeAddF16(commandBuffer cb: MTLCommandBuffer,
                      a: MTLBuffer,
                      aOffset: Int = 0,
                      b: MTLBuffer,
                      bOffset: Int = 0,
                      out: MTLBuffer,
                      outOffset: Int = 0,
                      count: Int) {
        precondition(count > 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(addF16PSO)
        enc.setBuffer(a, offset: aOffset, index: 0)
        enc.setBuffer(b, offset: bOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var n = UInt32(count)
        enc.setBytes(&n, length: 4, index: 3)
        dispatch1D(enc, pso: addF16PSO, threads: count)
        enc.endEncoding()
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState,
                            threads: Int) {
        let width = min(Self.threadsPerGroup, Int(pso.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
