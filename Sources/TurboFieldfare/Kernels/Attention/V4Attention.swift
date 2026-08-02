import Foundation
import Metal

/// Swift wrapper for the DeepSeek V4-Flash decode attention kernels in
/// `attention_v4.metal` (V4F-03, milestones 2 and 3).
///
/// Unlike `Attention` (Gemma, compiled into the shared `MetalContext`
/// library), this class historically compiled `attention_v4.metal`
/// standalone. The module is now registered in `MetalContext.shaderModules`
/// (V4F-04), so pipelines come from the shared context via
/// `V4ShaderLibrary.context(for:)`; the `init(device:...)` signature stays
/// for the committed call sites.
///
/// Decode entry points:
///
/// - `encodeWindowMQADecode`: ratio-0 layers (0/1/42). Pure sliding-window
///   MQA over the 128-slot FP16 ring + per-head sinks. This is milestone 2;
///   it is the `nSparse == 0` case of the merged kernel.
/// - `encodeCSADecode`: CSA layers. Lightning indexer scores over the
///   visible compressed groups, chunked bitonic top-512 selection, then the
///   merged sparse + window partial and the sink combine. The indexer stays
///   GPU-resident end to end — no CPU readback, cb1-compatible.
/// - `encodeHCADecode`: HCA layers. Dense gather over all visible
///   compressed entries (iota fill) + window, same merged kernel.
///
/// All three end with `v4_sink_combine` (sink in the denominator only).
final class V4Attention {
    static let headDim = 512
    static let numQHeads = 64
    static let indexHeads = 64
    static let indexHeadDim = 128
    static let window = 128
    static let indexTopK = 512
    static let topKChunk = 2048
    static let threadsPerGroup = 256

    /// Split-KV chunking: the attendable range is bounded (512 sparse + 128
    /// window for CSA; maxContext/128 + 128 dense for HCA), so 16 chunks
    /// ceiling keeps the partial scratch at 64 x 16 x 512 x 4 B = 2 MB.
    static let maxChunks = 16

    /// Attention scale applied after the QK dot (recon: 512^-0.5, no YaRN
    /// mscale adjustment).
    static let softmaxScale: Float = 1.0 / Float(512).squareRoot()

    private let device: MTLDevice
    private let context: MetalContext

    // Partial scratch (pass 1 -> pass 2, hazard-tracked in one CB).
    private let mPartial: MTLBuffer
    private let dPartial: MTLBuffer
    private let oPartial: MTLBuffer

    // Indexer scratch: scores over compressed groups, then ping-pong
    // (score, index) candidate lists for the chunked top-k.
    private let maxBlocks: Int
    private let indexScores: MTLBuffer
    private let topkScoresA: MTLBuffer
    private let topkIndexA: MTLBuffer
    private let topkScoresB: MTLBuffer
    private let topkIndexB: MTLBuffer
    /// Final gather list consumed by the sparse partial (also the HCA dense
    /// iota list; sized for the larger of the two).
    private let gatherList: MTLBuffer
    private let maxGather: Int

    init(device: MTLDevice, maxContext: Int, bundle _: Bundle = .module) throws {
        precondition(maxContext > 0)
        self.device = device
        self.context = try V4ShaderLibrary.context(for: device)

        let md = Self.numQHeads * Self.maxChunks
        guard let m = device.makeBuffer(length: md * MemoryLayout<Float>.size,
                                        options: .storageModeShared),
              let d = device.makeBuffer(length: md * MemoryLayout<Float>.size,
                                        options: .storageModeShared),
              let o = device.makeBuffer(length: md * Self.headDim * MemoryLayout<Float>.size,
                                        options: .storageModeShared) else {
            throw MetalError.missingFunction("v4 attention split-KV scratch")
        }
        self.mPartial = m; self.dPartial = d; self.oPartial = o

        let maxBlocks = max(1, maxContext / 4)      // CSA compressed capacity
        self.maxBlocks = maxBlocks
        // Ping-pong candidate lists: pass 1 emits ceil(blocks/2048) chunks of
        // 512 candidates; later passes only shrink, so size for the worst.
        let maxCandidates = max(Self.topKChunk,
                                ((maxBlocks + Self.topKChunk - 1) / Self.topKChunk) * Self.indexTopK)
        guard let scores = device.makeBuffer(length: maxBlocks * MemoryLayout<Float>.size,
                                             options: .storageModeShared),
              let sa = device.makeBuffer(length: maxCandidates * MemoryLayout<Float>.size,
                                         options: .storageModeShared),
              let ia = device.makeBuffer(length: maxCandidates * MemoryLayout<Int32>.size,
                                         options: .storageModeShared),
              let sb = device.makeBuffer(length: maxCandidates * MemoryLayout<Float>.size,
                                         options: .storageModeShared),
              let ib = device.makeBuffer(length: maxCandidates * MemoryLayout<Int32>.size,
                                         options: .storageModeShared) else {
            throw MetalError.missingFunction("v4 indexer scratch")
        }
        self.indexScores = scores
        self.topkScoresA = sa; self.topkIndexA = ia
        self.topkScoresB = sb; self.topkIndexB = ib

        let maxGather = max(Self.indexTopK, maxContext / 128)   // CSA sparse vs HCA dense
        self.maxGather = maxGather
        guard let g = device.makeBuffer(length: maxGather * MemoryLayout<Int32>.size,
                                        options: .storageModeShared) else {
            throw MetalError.missingFunction("v4 gather list")
        }
        self.gatherList = g
    }

    private func pipeline(_ name: String) throws -> MTLComputePipelineState {
        try context.pipeline(name)
    }

    // MARK: - Milestone 2: ratio-0 sliding-window MQA

    /// Pure 128-token sliding-window MQA + per-head sinks (layers 0/1/42).
    ///
    /// - `q`: [64, 512] FP16 (post per-head RMSNorm + trailing-slice RoPE).
    /// - `windowK`: the layer's FP16 window ring ([128, 512]).
    /// - `tokenCount`: tokens written so far, including the current one;
    ///   the window covers the last `min(128, tokenCount)` ring slots.
    /// - `sinks`: [64] FP32 per-head sink logits.
    /// - `out`: [64, 512] FP16.
    func encodeWindowMQADecode(commandBuffer cb: MTLCommandBuffer,
                               q: MTLBuffer, qOffset: Int = 0,
                               windowK: MTLBuffer, windowKOffset: Int = 0,
                               tokenCount: Int,
                               sinks: MTLBuffer, sinksOffset: Int = 0,
                               out: MTLBuffer, outOffset: Int = 0) {
        let nWindow = min(Self.window, tokenCount)
        precondition(nWindow > 0, "window attention requires at least one token")
        encodeMergedPartial(commandBuffer: cb,
                            q: q, qOffset: qOffset,
                            gather: gatherList, nSparse: 0,
                            compressedValues: windowK,   // unused when nSparse == 0
                            compressedScales: windowK,
                            compressedRope: windowK,
                            windowK: windowK, windowKOffset: windowKOffset,
                            nWindow: nWindow)
        encodeSinkCombine(commandBuffer: cb,
                          sinks: sinks, sinksOffset: sinksOffset,
                          out: out, outOffset: outOffset)
    }

    // MARK: - Milestone 3: CSA sparse decode

    /// CSA decode: indexer scores over the visible compressed groups,
    /// top-512 selection, merged sparse + window attention with sinks.
    ///
    /// - `indexQ`: [64, 128] FP16 indexer queries (from the shared q_lora,
    ///   RoPE'd; Hadamard + FP4 sim applied write-side per reference).
    /// - `indexKV`: indexer compressed cache ([capacity, 128] FP16).
    /// - `indexWeights`: [64] FP32 per-head weights (pre-scaled).
    /// - `nVisible`: `cache.visibleGroupCount(layer:windowStart:)` — every
    ///   completed compressed group, including intentional window overlap.
    /// - Compressed split buffers + window ring from the cache manager.
    /// - Returns the number of selected sparse entries (for tests).
    @discardableResult
    func encodeCSADecode(commandBuffer cb: MTLCommandBuffer,
                         q: MTLBuffer, qOffset: Int = 0,
                         indexQ: MTLBuffer, indexQOffset: Int = 0,
                         indexKV: MTLBuffer, indexKVOffset: Int = 0,
                         indexWeights: MTLBuffer, indexWeightsOffset: Int = 0,
                         nVisible: Int,
                         compressedValues: MTLBuffer,
                         compressedScales: MTLBuffer,
                         compressedRope: MTLBuffer,
                         windowK: MTLBuffer, windowKOffset: Int = 0,
                         tokenCount: Int,
                         sinks: MTLBuffer, sinksOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0) -> Int {
        precondition(nVisible <= maxBlocks, "nVisible exceeds indexer scratch")
        let nWindow = min(Self.window, tokenCount)
        precondition(nWindow > 0, "CSA decode requires at least one token")
        let nSparse: Int
        if nVisible > 0 {
            encodeIndexerScores(commandBuffer: cb,
                                indexQ: indexQ, indexQOffset: indexQOffset,
                                indexKV: indexKV, indexKVOffset: indexKVOffset,
                                indexWeights: indexWeights,
                                indexWeightsOffset: indexWeightsOffset,
                                nBlocks: nVisible)
            nSparse = encodeTopK(commandBuffer: cb, n: nVisible)
        } else {
            nSparse = 0
        }
        encodeMergedPartial(commandBuffer: cb,
                            q: q, qOffset: qOffset,
                            gather: gatherList, nSparse: nSparse,
                            compressedValues: compressedValues,
                            compressedScales: compressedScales,
                            compressedRope: compressedRope,
                            windowK: windowK, windowKOffset: windowKOffset,
                            nWindow: nWindow)
        encodeSinkCombine(commandBuffer: cb,
                          sinks: sinks, sinksOffset: sinksOffset,
                          out: out, outOffset: outOffset)
        return nSparse
    }

    /// HCA decode: dense over all visible compressed entries (no indexer).
    @discardableResult
    func encodeHCADecode(commandBuffer cb: MTLCommandBuffer,
                         q: MTLBuffer, qOffset: Int = 0,
                         nVisible: Int,
                         compressedValues: MTLBuffer,
                         compressedScales: MTLBuffer,
                         compressedRope: MTLBuffer,
                         windowK: MTLBuffer, windowKOffset: Int = 0,
                         tokenCount: Int,
                         sinks: MTLBuffer, sinksOffset: Int = 0,
                         out: MTLBuffer, outOffset: Int = 0) -> Int {
        precondition(nVisible <= maxGather, "nVisible exceeds gather scratch")
        let nWindow = min(Self.window, tokenCount)
        precondition(nWindow > 0, "HCA decode requires at least one token")
        if nVisible > 0 {
            do {
                let pso = try pipeline("v4_iota")
                guard let enc = cb.makeComputeCommandEncoder() else { return 0 }
                enc.setComputePipelineState(pso)
                enc.setBuffer(gatherList, offset: 0, index: 0)
                var n = UInt32(nVisible)
                enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 1)
                let w = min(256, Int(pso.maxTotalThreadsPerThreadgroup))
                enc.dispatchThreads(MTLSize(width: nVisible, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
                enc.endEncoding()
            } catch {
                preconditionFailure("v4_iota pipeline failed: \(error)")
            }
        }
        encodeMergedPartial(commandBuffer: cb,
                            q: q, qOffset: qOffset,
                            gather: gatherList, nSparse: nVisible,
                            compressedValues: compressedValues,
                            compressedScales: compressedScales,
                            compressedRope: compressedRope,
                            windowK: windowK, windowKOffset: windowKOffset,
                            nWindow: nWindow)
        encodeSinkCombine(commandBuffer: cb,
                          sinks: sinks, sinksOffset: sinksOffset,
                          out: out, outOffset: outOffset)
        return nVisible
    }

    // MARK: - CSA compressor

    /// Flush one CSA compressed entry (decode group completion, or per-group
    /// in prefill). `prev`/`cur` are the overlapped 4-token groups'
    /// 1024-dim wkv/wgate projections (FP32). Writes the split FP8/FP16
    /// entry via the slot offsets from `CompressedKVCacheManager`.
    func encodeCSACompressGroup(commandBuffer cb: MTLCommandBuffer,
                                prevKV: MTLBuffer, curKV: MTLBuffer,
                                prevGate: MTLBuffer, curGate: MTLBuffer,
                                ape: MTLBuffer, apeOffset: Int = 0,
                                gamma: MTLBuffer, gammaOffset: Int = 0,
                                outValues: MTLBuffer, valuesOffset: Int,
                                outScales: MTLBuffer, scalesOffset: Int,
                                outRope: MTLBuffer, ropeOffset: Int,
                                ropePosition: UInt32,
                                ropeTheta: Float = 160_000,
                                yarnFactor: Float = 16,
                                originalSeqLen: Float = 65_536,
                                betaFast: Float = 32,
                                betaSlow: Float = 1,
                                useYarn: Bool = true,
                                normEps: Float = 1e-6) {
        do {
            let pso = try pipeline("v4_csa_compress_group")
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(prevKV, offset: 0, index: 0)
            enc.setBuffer(curKV, offset: 0, index: 1)
            enc.setBuffer(prevGate, offset: 0, index: 2)
            enc.setBuffer(curGate, offset: 0, index: 3)
            enc.setBuffer(ape, offset: apeOffset, index: 4)
            enc.setBuffer(gamma, offset: gammaOffset, index: 5)
            enc.setBuffer(outValues, offset: valuesOffset, index: 6)
            enc.setBuffer(outScales, offset: scalesOffset, index: 7)
            enc.setBuffer(outRope, offset: ropeOffset, index: 8)
            var pos = ropePosition
            var theta = ropeTheta
            var factor = yarnFactor
            var orig = originalSeqLen
            var bf = betaFast
            var bs = betaSlow
            var uy = UInt32(useYarn ? 1 : 0)
            var eps = normEps
            enc.setBytes(&pos, length: 4, index: 9)
            enc.setBytes(&theta, length: 4, index: 10)
            enc.setBytes(&factor, length: 4, index: 11)
            enc.setBytes(&orig, length: 4, index: 12)
            enc.setBytes(&bf, length: 4, index: 13)
            enc.setBytes(&bs, length: 4, index: 14)
            enc.setBytes(&uy, length: 4, index: 15)
            enc.setBytes(&eps, length: 4, index: 16)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
            enc.endEncoding()
        } catch {
            preconditionFailure("v4_csa_compress_group pipeline failed: \(error)")
        }
    }

    // MARK: - Internals

    private var lastNumChunks = 0

    /// Pass 1: merged [sparse gather | window ring] partial. With
    /// `nSparse == 0` this is the pure ratio-0 window path.
    private func encodeMergedPartial(commandBuffer cb: MTLCommandBuffer,
                                     q: MTLBuffer, qOffset: Int,
                                     gather: MTLBuffer, nSparse: Int,
                                     compressedValues: MTLBuffer,
                                     compressedScales: MTLBuffer,
                                     compressedRope: MTLBuffer,
                                     windowK: MTLBuffer, windowKOffset: Int,
                                     nWindow: Int) {
        let total = nSparse + nWindow
        let numChunks = max(1, min(Self.maxChunks, (total + 63) / 64))
        let chunkLen = (total + numChunks - 1) / numChunks
        lastNumChunks = numChunks
        do {
            let pso = try pipeline("v4_mqa_partial")
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(q, offset: qOffset, index: 0)
            enc.setBuffer(gather, offset: 0, index: 1)
            enc.setBuffer(compressedValues, offset: 0, index: 2)
            enc.setBuffer(compressedScales, offset: 0, index: 3)
            enc.setBuffer(compressedRope, offset: 0, index: 4)
            enc.setBuffer(windowK, offset: windowKOffset, index: 5)
            enc.setBuffer(mPartial, offset: 0, index: 6)
            enc.setBuffer(dPartial, offset: 0, index: 7)
            enc.setBuffer(oPartial, offset: 0, index: 8)
            var nq = UInt32(Self.numQHeads)
            var ns = UInt32(nSparse)
            var nw = UInt32(nWindow)
            var cl = UInt32(chunkLen)
            var nc = UInt32(numChunks)
            var sc = Self.softmaxScale
            enc.setBytes(&nq, length: 4, index: 9)
            enc.setBytes(&ns, length: 4, index: 10)
            enc.setBytes(&nw, length: 4, index: 11)
            enc.setBytes(&cl, length: 4, index: 12)
            enc.setBytes(&nc, length: 4, index: 13)
            enc.setBytes(&sc, length: 4, index: 14)
            let tg = min(Self.threadsPerGroup, Int(pso.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreadgroups(MTLSize(width: Self.numQHeads * numChunks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        } catch {
            preconditionFailure("v4_mqa_partial pipeline failed: \(error)")
        }
    }

    /// Pass 2: partial merge + per-head sink in the denominator.
    private func encodeSinkCombine(commandBuffer cb: MTLCommandBuffer,
                                   sinks: MTLBuffer, sinksOffset: Int,
                                   out: MTLBuffer, outOffset: Int) {
        do {
            let pso = try pipeline("v4_sink_combine")
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(mPartial, offset: 0, index: 0)
            enc.setBuffer(dPartial, offset: 0, index: 1)
            enc.setBuffer(oPartial, offset: 0, index: 2)
            enc.setBuffer(sinks, offset: sinksOffset, index: 3)
            enc.setBuffer(out, offset: outOffset, index: 4)
            var nc = UInt32(max(1, lastNumChunks))
            enc.setBytes(&nc, length: 4, index: 5)
            let tg = min(Self.threadsPerGroup, Int(pso.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreadgroups(MTLSize(width: Self.numQHeads, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        } catch {
            preconditionFailure("v4_sink_combine pipeline failed: \(error)")
        }
    }

    private func encodeIndexerScores(commandBuffer cb: MTLCommandBuffer,
                                     indexQ: MTLBuffer, indexQOffset: Int,
                                     indexKV: MTLBuffer, indexKVOffset: Int,
                                     indexWeights: MTLBuffer, indexWeightsOffset: Int,
                                     nBlocks: Int) {
        do {
            let pso = try pipeline("v4_indexer_score")
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pso)
            enc.setBuffer(indexQ, offset: indexQOffset, index: 0)
            enc.setBuffer(indexKV, offset: indexKVOffset, index: 1)
            enc.setBuffer(indexWeights, offset: indexWeightsOffset, index: 2)
            enc.setBuffer(indexScores, offset: 0, index: 3)
            var nh = UInt32(Self.indexHeads)
            enc.setBytes(&nh, length: 4, index: 4)
            let tg = min(Self.threadsPerGroup, Int(pso.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreadgroups(MTLSize(width: nBlocks, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            enc.endEncoding()
        } catch {
            preconditionFailure("v4_indexer_score pipeline failed: \(error)")
        }
    }

    /// Chunked bitonic top-k. Ping-pongs the candidate list until one
    /// 2048-wide chunk remains; compacts the final selection into
    /// `gatherList` (indices only) and returns the selected count.
    private func encodeTopK(commandBuffer cb: MTLCommandBuffer, n: Int) -> Int {
        let k = min(Self.indexTopK, n)
        do {
            let pso = try pipeline("v4_topk_chunk")
            var inScores = indexScores
            var inIndex = topkIndexA       // unused on the implicit first pass
            var outScores = topkScoresA
            var outIndex = topkIndexA
            var implicit = true
            var count = n
            while true {
                let chunks = (count + Self.topKChunk - 1) / Self.topKChunk
                guard let enc = cb.makeComputeCommandEncoder() else { return 0 }
                enc.setComputePipelineState(pso)
                enc.setBuffer(inScores, offset: 0, index: 0)
                enc.setBuffer(inIndex, offset: 0, index: 1)
                enc.setBuffer(outScores, offset: 0, index: 2)
                enc.setBuffer(outIndex, offset: 0, index: 3)
                var cn = UInt32(count)
                var imp = UInt32(implicit ? 1 : 0)
                enc.setBytes(&cn, length: 4, index: 4)
                enc.setBytes(&imp, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: chunks, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
                enc.endEncoding()
                // Every chunk emits a full kV4TopK-strided candidate block
                // (padding slots carry -inf scores), so the next pass sees
                // chunks * indexTopK candidates.
                let next = chunks * Self.indexTopK
                if chunks == 1 { break }
                // Next pass reads this pass's candidates.
                inScores = outScores
                inIndex = outIndex
                implicit = false
                count = next
                outScores = (outScores === topkScoresA) ? topkScoresB : topkScoresA
                outIndex = (outIndex === topkIndexA) ? topkIndexB : topkIndexA
            }
            // Compact the final chunk's indices into the gather list.
            // (A trivial copy kernel would be nicer; this encoder-side
            // blit keeps the selection GPU-resident without a sync.)
            if let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: outIndex, sourceOffset: 0,
                          to: gatherList, destinationOffset: 0,
                          size: k * MemoryLayout<Int32>.size)
                blit.endEncoding()
            }
            return k
        } catch {
            preconditionFailure("v4_topk_chunk pipeline failed: \(error)")
        }
    }

    // MARK: - Test introspection

    /// Read back indexer scores (tests only; syncs on the caller's CB).
    var indexerScoresBuffer: MTLBuffer { indexScores }
    /// Read back the final gather list (tests only).
    var gatherListBuffer: MTLBuffer { gatherList }
}
