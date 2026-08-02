import Foundation
import Metal

/// Decode forward runner for DeepSeek V4-Flash (V4F-04). Composes the
/// committed V4 kernel wrappers into the per-token layer graph:
///
///   embed broadcast -> per layer [ mHC attn boundary -> QKV epilogue ->
///   group flush (CSA/HCA) -> attention (window / CSA / HCA) -> output
///   de-rotation -> grouped o-proj -> mHC post -> mHC ffn boundary ->
///   router (or hash table) -> expert stream -> fused SwiGLU MoE + shared
///   expert -> mHC post ] -> head boundary -> final norm -> fp32 lm_head.
///
/// Command-buffer phasing mirrors the Gemma cb1/io/cb2 pattern:
///
/// - `cb1` ends at the router's top-6 readback (router layers) or
///   immediately (hash layers read the resident tid2eid table on CPU).
/// - `cb2` encodes the resident shared expert and the cache-HIT subset of
///   routed phase 1 while the async `pread` fills miss slots.
/// - `cb3` runs the miss-subset phase 1, the weighted phase-2 reduce
///   (shared-expert output as the residual add), and the mHC ffn post.
///
/// Cross-layer pipelining (V4F-06a): the per-layer `waitUntilCompleted`
/// stalls are removed. GPU-side ordering needs no scratch duplication:
/// every layer's cb3 writes `stream` and the next layer's cb1 reads it, so
/// on one command queue Metal's hazard tracking serializes layers in
/// commit order exactly like the explicit waits did. The CPU-side waits
/// that remain:
///
/// - cb1 is awaited only on router layers, and only to read back the
///   top-6 indices; hash layers (resident tid2eid table) skip the wait.
/// - CPU-uploaded staging (hit/miss slot lists, hash ids) is a ring of 2
///   by layer parity; a layer drains the previous occupant of its parity
///   (layer N-2's cb2/cb3) before rewriting, bounding GPU lag to one
///   layer without ever letting the CPU write a buffer queued work reads.
/// - The routed argument buffer is fresh per layer (`makeRoutedArgumentBuffer`);
///   the reused singleton would be re-encoded while a prior layer's
///   cb2/cb3 still references it.
///
/// Slot ownership needs no `avoidingSlots`: each layer's
/// `PreadExpertStreamer` owns a disjoint slot pool, so a later layer's
/// plan can never evict a slot an earlier layer's queued GPU work reads,
/// and the per-token head wait guarantees same-layer eviction safety
/// across tokens.
///
/// Recorded simplifications (follow-ups, not oversights):
///
/// - Decode only. Prefill is the V4F-06 work item.
/// - The compressor epilogue writes one fp32 row at offset 0 of a scratch
///   buffer per token; the runner blits rows into the group accumulators
///   (CSA: prev/cur 4-row buffers, 1024-dim; HCA: 128 rows, 512-dim). A
///   row-offset variant of `V4QKVEpilogue.encodeDecode` would remove the
///   blit.
public final class V4ForwardRunner {

    // MARK: - Dependencies

    private let model: V4Model
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let maxContext: Int
    private let cache: CompressedKVCacheManager
    private let attention: V4Attention
    private let qkv: V4QKVEpilogue
    private let rope: V4RoPE
    private let hc: V4HyperConnections
    private let hca: V4HCACompressor
    private let moe: MoEV4
    private let glue: V4DecodeGlue
    private let fp8: DequantFP8BlockGEMV
    private let fp4: DequantFP4E2M1GEMV

    private var dim: Int { model.config.hiddenSize }
    private var ffn: Int { model.config.moeIntermediateSize }
    private var streams: Int { model.config.hcMult }
    private var topK: Int { model.config.topKExperts }
    private var numExperts: Int { model.config.numExperts }
    private var eps: Float { Float(model.config.rmsNormEps) }
    private var swigluLimit: Float { Float(model.config.swigluLimit) }
    private var routeScale: Float { Float(model.config.routedScalingFactor) }

    // MARK: - Scratch (allocated once, reused across tokens and layers)

    private let stream: MTLBuffer        // [4 * dim] fp32 mHC residual stream
    private let branch: MTLBuffer        // [dim] fp32 mHC pre-gather output
    private let xNorm: MTLBuffer         // [dim] fp16 normed branch input
    private let q: MTLBuffer             // [64, 512] fp16 queries
    private let indexQ: MTLBuffer        // [64, 128] fp16 indexer queries
    private let indexW: MTLBuffer        // [64] fp32 indexer head weights
    private let attnOut: MTLBuffer       // [64, 512] fp16 attention output
    private let oProjOut: MTLBuffer      // [dim] fp16 attention projection
    private let routerIdx: MTLBuffer     // [6] int32 (CPU readback)
    private let routerWts: MTLBuffer     // [6] fp32
    private let routerLogits: MTLBuffer  // [numExperts] fp32 (hash scores)
    private let acts: MTLBuffer          // [6 * F] fp16 routed activations
    private let sharedGate: MTLBuffer    // [F] fp16
    private let sharedUp: MTLBuffer      // [F] fp16
    private let sharedAct: MTLBuffer     // [F] fp16
    private let sharedOut: MTLBuffer     // [dim] fp16 shared-expert output
    private let ffnOut: MTLBuffer        // [dim] fp16 shared + routed
    private let logits: MTLBuffer        // [vocab] fp32
    /// CPU-uploaded expert staging, ring of 2 indexed by `layer & 1`.
    /// Layer N's cb2/cb3 may still be queued when the CPU fills layer
    /// N+2's lists, so each parity owns its copies; hit and miss lists are
    /// separate because cb2 reads the hit list while the CPU writes the
    /// miss list for cb3 of the SAME layer.
    private struct UploadStaging {
        let hit: MTLBuffer   // [6] uint32 hit-subset plan positions (cb2)
        let miss: MTLBuffer  // [6] uint32 miss-subset plan positions (cb3)
        let ids: MTLBuffer   // [6] int32 hash-layer expert ids (cb2)
    }
    private let uploadStaging: [UploadStaging]
    /// The (cb2, cb3) most recently committed against each staging parity,
    /// drained before that parity's buffers are rewritten and once after
    /// the layer loop.
    private var stagingInFlight: [(MTLCommandBuffer, MTLCommandBuffer)?] = [nil, nil]
    private let compressorAccumulators: V4CompressorAccumulatorStore
    private let lowRank: MTLBuffer       // [8192] fp16 o-proj stage 1
    private let headF32: MTLBuffer       // [vocab, dim] fp32 (converted)
    private let rowScratch: MTLBuffer    // [1024] fp32 epilogue single row
    private let rowScratch2: MTLBuffer   // [1024] fp32 second series
    private var chunkedPrefillState: V4ChunkedPrefillState?
    private var chunkedPrefillDirty = false

    public init(model: V4Model, maxContext: Int) throws {
        self.model = model
        self.device = model.device
        self.maxContext = maxContext
        let context = try V4ShaderLibrary.context(for: model.device)
        self.queue = context.queue
        let cacheConfig = V4CacheConfig(compressRatios: model.config.compressRatios,
                                        headDim: model.config.headDim,
                                        ropeDim: model.config.qkRopeHeadDim,
                                        window: model.config.slidingWindow,
                                        numQHeads: model.config.numHeads)
        self.cache = try CompressedKVCacheManager(device: model.device,
                                                  config: cacheConfig,
                                                  maxContext: maxContext)
        self.attention = try V4Attention(device: model.device, maxContext: maxContext)
        self.qkv = try V4QKVEpilogue(device: model.device)
        self.rope = try V4RoPE(device: model.device)
        self.hc = try V4HyperConnections(device: model.device)
        self.hca = try V4HCACompressor(device: model.device)
        self.moe = try MoEV4(context: context)
        self.glue = try V4DecodeGlue(context: context)
        self.fp8 = try DequantFP8BlockGEMV(context: context)
        self.fp4 = try DequantFP4E2M1GEMV(context: context)

        func scratch(_ floats: Int, _ label: String) throws -> MTLBuffer {
            guard let b = model.device.makeBuffer(length: floats * 4,
                                                  options: .storageModeShared) else {
                throw MetalError.missingFunction("v4 runner scratch: \(label)")
            }
            return b
        }
        func scratch16(_ elems: Int, _ label: String) throws -> MTLBuffer {
            guard let b = model.device.makeBuffer(
                    length: elems * MemoryLayout<Float16>.size,
                    options: .storageModeShared) else {
                throw MetalError.missingFunction("v4 runner scratch: \(label)")
            }
            return b
        }
        let dim = model.config.hiddenSize
        let heads = model.config.numHeads
        let headDim = model.config.headDim
        self.stream = try scratch(model.config.hcMult * dim, "stream")
        self.branch = try scratch(dim, "branch")
        self.xNorm = try scratch16(dim, "xNorm")
        self.q = try scratch16(heads * headDim, "q")
        self.indexQ = try scratch16(heads * 128, "indexQ")
        self.indexW = try scratch(heads, "indexW")
        self.attnOut = try scratch16(heads * headDim, "attnOut")
        self.oProjOut = try scratch16(dim, "oProjOut")
        self.routerIdx = try scratch(4, "routerIdx")
        self.routerWts = try scratch(8, "routerWts")
        self.routerLogits = try scratch(model.config.numExperts, "routerLogits")
        self.acts = try scratch16(6 * model.config.moeIntermediateSize, "acts")
        self.sharedGate = try scratch16(model.config.moeIntermediateSize, "sharedGate")
        self.sharedUp = try scratch16(model.config.moeIntermediateSize, "sharedUp")
        self.sharedAct = try scratch16(model.config.moeIntermediateSize, "sharedAct")
        self.sharedOut = try scratch16(dim, "sharedOut")
        self.ffnOut = try scratch16(dim, "ffnOut")
        self.logits = try scratch(model.config.vocabSize, "logits")
        self.uploadStaging = [
            UploadStaging(hit: try scratch(4, "slotUploadHit0"),
                          miss: try scratch(4, "slotUploadMiss0"),
                          ids: try scratch(4, "idUpload0")),
            UploadStaging(hit: try scratch(4, "slotUploadHit1"),
                          miss: try scratch(4, "slotUploadMiss1"),
                          ids: try scratch(4, "idUpload1")),
        ]
        self.compressorAccumulators = try V4CompressorAccumulatorStore(
            device: model.device,
            layerKinds: cacheConfig.compressRatios.indices.map { cacheConfig.kind(layer: $0) })
        self.lowRank = try scratch16(8192, "lowRank")
        // head.weight ships BF16; v4b_gemv_f32 wants fp32. Exact one-time
        // widening (bf16 bits << 16) at init.
        let headView = try model.lmHead
        guard let headBuf = model.device.makeBuffer(
                length: model.config.vocabSize * model.config.hiddenSize * 4,
                options: .storageModeShared) else {
            throw MetalError.missingFunction("v4 head staging")
        }
        widenBF16ToF32(src: headView.buffer.contents().advanced(by: Int(headView.offset)),
                       dst: headBuf.contents(),
                       count: model.config.vocabSize * model.config.hiddenSize)
        self.headF32 = headBuf
        self.rowScratch = try scratch(1024, "rowScratch")
        self.rowScratch2 = try scratch(1024, "rowScratch2")
    }

    // MARK: - Decode step

    /// Embed `token` into the fp32 mHC stream, then run one decode token
    /// through the full stack. Returns the fp32 logits buffer (valid until
    /// the next call). `position` is the absolute 0-based token index.
    public func forward(token: UInt32, position: Int) async throws -> MTLBuffer {
        guard !chunkedPrefillDirty else {
            throw PrefillError.chunkedRunnerDirty(
                "V4 decode rejected because a failed chunked prefill left cache state dirty; reset the producer")
        }
        guard position == cache.position else {
            throw PrefillError.prefillCursorMismatch(
                "V4 decode position \(position) != cache cursor \(cache.position)")
        }
        let tokenCount = position + 1

        for layer in 0..<model.config.numLayers {
            let kind = cache.layerKind(layer)
            let ropeCfg = V4RoPE.Config.forLayer(kind)

            // ---- cb1: boundaries, attention, router ----------------------
            guard let cb1 = queue.makeCommandBuffer() else { throw MetalError.noQueue }
            if layer == 0 {
                let table = try model.embedding
                glue.encodeEmbedBroadcast(commandBuffer: cb1,
                                          table: table.buffer,
                                          tableOffset: Int(table.offset),
                                          out: stream, token: token,
                                          dim: UInt32(dim), streams: UInt32(streams))
            }

            let hcA = try (model.hcAttnFn(layer: layer),
                           model.hcAttnBase(layer: layer),
                           model.hcAttnScale(layer: layer))
            hc.encodeParams(commandBuffer: cb1, x: stream,
                            hcFn: hcA.0.buffer, hcFnOffset: Int(hcA.0.offset),
                            hcBase: hcA.1.buffer, hcBaseOffset: Int(hcA.1.offset),
                            hcScale: hcA.2.buffer, hcScaleOffset: Int(hcA.2.offset),
                            dim: dim, normEps: eps, hcEps: Float(model.config.hcEps))
            hc.encodePre(commandBuffer: cb1, x: stream, out: branch, dim: dim)
            let attnNorm = try gammaF32(model.attnNorm(layer: layer),
                                        name: "layers.\(layer).attn_norm")
            glue.encodeRMSNormF32In(commandBuffer: cb1, x: branch,
                                    w: attnNorm, wOffset: 0,
                                    out: xNorm, dim: UInt32(dim), eps: eps)

            let weights = try attentionWeights(layer: layer, kind: kind)
            let slot = cache.windowSlot(layer: layer, position: position)
            qkv.encodeDecode(commandBuffer: cb1, x: xNorm, position: position,
                             weights: weights, rope: ropeCfg, qOut: q,
                             windowSlot: .init(buffer: slot.buffer, offset: slot.offset),
                             compressorWKVOut: nil,
                             compressorWGateOut: nil,
                             indexQOut: kind == .csa ? indexQ : nil,
                             normEps: eps)
            stageCompressorRows(commandBuffer: cb1, layer: layer, kind: kind,
                                position: position)
            if kind != .passthrough,
               cache.completesGroup(layer: layer, tokenPosition: position) {
                flushGroup(commandBuffer: cb1, layer: layer, kind: kind,
                           position: position, ropeCfg: ropeCfg)
            }
            try encodeAttention(commandBuffer: cb1, layer: layer, kind: kind,
                                tokenCount: tokenCount, ropeCfg: ropeCfg)

            // Output de-rotation: complex conjugate at the QUERY position
            // (reference: apply_rotary_emb(o[..., -rd:], freqs_cis, True)).
            // The kernel's `inverse` flag already negates the sine term, so
            // the position itself stays positive — passing -position here
            // would re-rotate instead of de-rotating.
            rope.encode(commandBuffer: cb1, x: attnOut,
                        rows: model.config.numHeads, width: model.config.headDim,
                        ropeDim: model.config.qkRopeHeadDim,
                        position: Float(position), inverse: true, config: ropeCfg)
            // Grouped output projection (reference: o.view(b,s,8,4096) then
            // einsum("bsgd,grd->bsgr", o, wo_a)). wo_a is F8_E4M3
            // [8192, 4096] group-major: rows [g*1024, (g+1)*1024) belong to
            // group g and must dot with attnOut[g*4096 ..< (g+1)*4096]. The
            // FP8 GEMV kernel is a plain y = W.x over one x slice, so loop
            // the 8 groups with per-group weight/scale/x/y offsets.
            let woA = try model.woA(layer: layer)
            let woB = try model.woB(layer: layer)
            let groupRows = model.config.oLoraRank              // 1024 rows per group
            let groupDim = dim                                  // 4096 inputs per group
            let scaleRowBytes = groupDim / 128                  // ue8m0 grid row width
            for g in 0..<model.config.oGroups {
                fp8.encode(commandBuffer: cb1,
                           weights: woA.buffer,
                           weightsOffset: Int(woA.offset) + g * groupRows * groupDim,
                           scales: woA.buffer,
                           scalesOffset: Int(woA.scaleOffset) + g * (groupRows / 128) * scaleRowBytes,
                           x: attnOut,
                           xOffset: g * groupDim * MemoryLayout<Float16>.size,
                           y: lowRank,
                           yOffset: g * groupRows * MemoryLayout<Float16>.size,
                           m: UInt32(groupRows),
                           n: UInt32(groupDim))
            }
            fp8.encode(commandBuffer: cb1,
                       weights: woB.buffer, weightsOffset: Int(woB.offset),
                       scales: woB.buffer, scalesOffset: Int(woB.scaleOffset),
                       x: lowRank, y: oProjOut,
                       m: UInt32(dim),
                       n: UInt32(model.config.oGroups * model.config.oLoraRank))
            hc.encodePost(commandBuffer: cb1, residual: stream,
                          sublayer: oProjOut, out: stream, dim: dim)

            let hcF = try (model.hcFfnFn(layer: layer),
                           model.hcFfnBase(layer: layer),
                           model.hcFfnScale(layer: layer))
            hc.encodeParams(commandBuffer: cb1, x: stream,
                            hcFn: hcF.0.buffer, hcFnOffset: Int(hcF.0.offset),
                            hcBase: hcF.1.buffer, hcBaseOffset: Int(hcF.1.offset),
                            hcScale: hcF.2.buffer, hcScaleOffset: Int(hcF.2.offset),
                            dim: dim, normEps: eps, hcEps: Float(model.config.hcEps))
            hc.encodePre(commandBuffer: cb1, x: stream, out: branch, dim: dim)
            let ffnNorm = try gammaF32(model.ffnNorm(layer: layer),
                                       name: "layers.\(layer).ffn_norm")
            glue.encodeRMSNormF32In(commandBuffer: cb1, x: branch,
                                    w: ffnNorm, wOffset: 0,
                                    out: xNorm, dim: UInt32(dim), eps: eps)
            let isHash = model.config.isHashRouted(layer: layer)
            if !isHash {
                let gate = try model.routerWeight(layer: layer)
                let bias = try model.routerBias(layer: layer)
                moe.encodeRouterV4(commandBuffer: cb1,
                                   weights: gate.buffer, weightsOffset: Int(gate.offset),
                                   bias: bias.buffer, biasOffset: Int(bias.offset),
                                   hidden: xNorm,
                                   outIndices: routerIdx, outWeights: routerWts,
                                   numExperts: UInt32(numExperts), d: UInt32(dim),
                                   routeScale: routeScale)
            }
            cb1.commit()
            // The cb1 wait exists only for the router top-6 readback.
            // Hash layers take ids from the resident tid2eid table, so
            // their plan/fetch runs CPU-side with no GPU round trip; cb1
            // stays queued behind the previous layer's cb3 via hazard
            // tracking on `stream`.
            if !isHash {
                await cb1.completed()
                if let error = cb1.error { throw error }
            }

            // ---- io: expert plan + async fetch ----------------------------
            let expertIDs: [Int]
            if isHash {
                expertIDs = try model.hashExpertIDs(layer: layer, token: token)
            } else {
                let ptr = routerIdx.contents().assumingMemoryBound(to: Int32.self)
                expertIDs = (0..<topK).map { Int(ptr[$0]) }
            }
            let plan = try model.planRoutedExperts(layer: layer, experts: expertIDs)
            // Phase-1 subset indices are POSITIONS in the plan's top-K
            // blob list (0..<topK), not cache slot numbers.
            let missSet = Set(plan.misses)
            let hitPositions = (0..<topK).filter { !missSet.contains($0) }
            let blobs = try model.routedExpertBuffers(for: plan).map(\.buffer)
            let offsets = model.routedExpertV4Offsets(layer: layer)
            // Fresh argument buffer per layer: the reused singleton would
            // be CPU-re-encoded here while layer N-1's queued cb2/cb3
            // still references it (a data race once cb3 waits are
            // deferred). The command buffer retains what it encodes.
            guard let argBuffer = moe.makeRoutedArgumentBuffer(routedBlobs: blobs,
                                                               topK: UInt32(topK)) else {
                throw MetalError.missingFunction("v4 routed argument buffer")
            }

            // Drain the previous occupant of this staging parity (layer
            // N-2's cb2/cb3) before the CPU rewrites its upload buffers.
            // Bounds GPU lag to one layer; usually free, because a router
            // layer's cb1 wait already implies the older work finished.
            let parity = layer & 1
            let staging = uploadStaging[parity]
            if let inFlight = stagingInFlight[parity] {
                await inFlight.0.completed()
                await inFlight.1.completed()
                if let error = inFlight.0.error ?? inFlight.1.error { throw error }
                stagingInFlight[parity] = nil
            }

            // ---- cb2: shared expert + hit-subset phase 1 ------------------
            guard let cb2 = queue.makeCommandBuffer() else { throw MetalError.noQueue }
            try encodeSharedExpert(commandBuffer: cb2, layer: layer)
            if isHash {
                let gate = try model.routerWeight(layer: layer)
                glue.encodeBF16GEMV(commandBuffer: cb2,
                                    weights: gate.buffer, weightsOffset: Int(gate.offset),
                                    x: xNorm, out: routerLogits,
                                    m: UInt32(numExperts), d: UInt32(dim))
                uploadInts(expertIDs.map { Int32($0) }, to: staging.ids)
                glue.encodeRouterWeightsAtIndices(commandBuffer: cb2,
                                                  logits: routerLogits,
                                                  indices: staging.ids,
                                                  outWeights: routerWts,
                                                  k: UInt32(topK),
                                                  routeScale: routeScale)
            }
            if !hitPositions.isEmpty {
                uploadUInts(hitPositions.map { UInt32($0) }, to: staging.hit)
                moe.encodeRoutedPhase1SwiGLUSubset(commandBuffer: cb2,
                                                   routedArgBuffer: argBuffer,
                                                   routedBlobs: blobs,
                                                   routedOffsets: offsets,
                                                   x: xNorm, acts: acts,
                                                   activeSlots: staging.hit,
                                                   activeSlotIndices: hitPositions.map { UInt32($0) },
                                                   activeCount: UInt32(hitPositions.count),
                                                   d: UInt32(dim), f: UInt32(ffn),
                                                   topK: UInt32(topK))
            }
            cb2.commit()

            // Await the miss reads, then finish the layer.
            _ = try await model.fetchRoutedExperts(plan: plan)

            guard let cb3 = queue.makeCommandBuffer() else { throw MetalError.noQueue }
            let missPositions = plan.misses
            if !missPositions.isEmpty {
                uploadUInts(missPositions.map { UInt32($0) }, to: staging.miss)
                moe.encodeRoutedPhase1SwiGLUSubset(commandBuffer: cb3,
                                                   routedArgBuffer: argBuffer,
                                                   routedBlobs: blobs,
                                                   routedOffsets: offsets,
                                                   x: xNorm, acts: acts,
                                                   activeSlots: staging.miss,
                                                   activeSlotIndices: missPositions.map { UInt32($0) },
                                                   activeCount: UInt32(missPositions.count),
                                                   d: UInt32(dim), f: UInt32(ffn),
                                                   topK: UInt32(topK))
            }
            moe.encodeRoutedPhase2Reduce(commandBuffer: cb3,
                                         routedArgBuffer: argBuffer,
                                         routedBlobs: blobs,
                                         routedOffsets: offsets,
                                         acts: acts,
                                         routingWeights: routerWts,
                                         residual: sharedOut,
                                         y: ffnOut,
                                         d: UInt32(dim), f: UInt32(ffn),
                                         topK: UInt32(topK))
            hc.encodePost(commandBuffer: cb3, residual: stream,
                          sublayer: ffnOut, out: stream, dim: dim)
            cb3.commit()
            // No per-layer cb3 wait: the next layer's cb1 reads `stream`
            // only on the GPU, and hazard tracking on one queue orders it
            // behind this cb3. The pairing is recorded so the staging
            // parity's drain (and the post-loop drain) can bound GPU lag
            // and surface errors.
            stagingInFlight[parity] = (cb2, cb3)

        }

        // Drain the pipeline once per token: every layer's GPU work is
        // complete before the cache position advances and the head reads
        // `stream`, which also keeps same-layer slot eviction across
        // tokens safe (a token's plans never race its own queued reads).
        for parity in 0..<uploadStaging.count {
            if let inFlight = stagingInFlight[parity] {
                await inFlight.0.completed()
                await inFlight.1.completed()
                if let error = inFlight.0.error ?? inFlight.1.error { throw error }
                stagingInFlight[parity] = nil
            }
        }

        // One cache position advance per token, after every layer consumed
        // this token's rows.
        cache.advance()

        // ---- head: pre-only mHC boundary, final norm, fp32 lm_head -------
        guard let cbH = queue.makeCommandBuffer() else { throw MetalError.noQueue }
        let hcH = try (model.hcHeadFn, model.hcHeadBase, model.hcHeadScale)
        hc.encodeHeadParams(commandBuffer: cbH, x: stream,
                            hcFn: hcH.0.buffer, hcFnOffset: Int(hcH.0.offset),
                            hcBase: hcH.1.buffer, hcBaseOffset: Int(hcH.1.offset),
                            hcScale: hcH.2.buffer, hcScaleOffset: Int(hcH.2.offset),
                            dim: dim, normEps: eps, hcEps: Float(model.config.hcEps))
        hc.encodePre(commandBuffer: cbH, x: stream, out: branch, dim: dim)
        let finalNorm = try gammaF32(model.finalNorm, name: "norm")
        glue.encodeRMSNormF32In(commandBuffer: cbH, x: branch,
                                w: finalNorm, wOffset: 0,
                                out: xNorm, dim: UInt32(dim), eps: eps)
        glue.encodeGemvF32(commandBuffer: cbH,
                           weights: headF32, weightsOffset: 0,
                           x: xNorm, out: logits,
                           m: UInt32(model.config.vocabSize), n: UInt32(dim))
        cbH.commit()
        await cbH.completed()
        return logits
    }

    // MARK: - Chunked prefill

    /// Run prompt tokens through the model in token-major chunks and
    /// layer-major order. This path never replays `forward`; projections and
    /// boundaries are batched while the existing sparse-attention decode
    /// kernels are queued causally for each row.
    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               config: PrefillRuntimeConfig) async throws -> MTLBuffer {
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "V4 prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard !chunkedPrefillDirty else {
            throw PrefillError.chunkedRunnerDirty(
                "V4 chunked prefill rejected because a previous chunk failed; reset the producer")
        }
        guard startPosition == cache.position else {
            throw PrefillError.prefillCursorMismatch(
                "V4 chunked prefill cursor \(cache.position) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "V4 chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else { return logits }

        let state: V4ChunkedPrefillState
        if let existing = chunkedPrefillState, existing.maxRows >= config.chunkTokens {
            state = existing
        } else {
            let created = try V4ChunkedPrefillState(model: model,
                                                    cache: cache,
                                                    attention: attention,
                                                    compressorAccumulators: compressorAccumulators,
                                                    maxRows: config.chunkTokens)
            chunkedPrefillState = created
            state = created
        }

        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for span in spans {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(tokens: tokens[lower..<upper],
                                          startPosition: span.startPosition,
                                          state: state)
        }
        return logits
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     state: V4ChunkedPrefillState) async throws {
        let rows = tokens.count
        precondition(rows > 0 && rows <= state.maxRows)
        guard cache.position == startPosition else {
            throw PrefillError.prefillCursorMismatch(
                "V4 chunk cursor \(cache.position) != span start \(startPosition)")
        }

        state.uploadTokenIDs(tokens.map { UInt32(bitPattern: $0) })
        state.fillPositions(startPosition: startPosition, rowCount: rows)
        chunkedPrefillDirty = true
        do {
            guard let embedCB = queue.makeCommandBuffer() else { throw MetalError.noQueue }
            let embedding = try model.embedding
            state.glue.encodeBF16EmbeddingGatherBroadcast(
                commandBuffer: embedCB,
                embeddings: embedding.buffer,
                embeddingsOffset: Int(embedding.offset),
                tokenIDs: state.tokenIDs,
                out: state.stream,
                rows: rows,
                dim: dim)
            embedCB.commit()
            await embedCB.completed()
            if let error = embedCB.error { throw error }

            for layer in 0..<model.config.numLayers {
                try await executePrefillLayer(layer: layer,
                                              tokens: tokens,
                                              startPosition: startPosition,
                                              rows: rows,
                                              state: state)
            }

            // Every layer consumed the same absolute chunk range. Only now
            // may the shared cache cursor move to the next prompt span.
            for _ in 0..<rows { cache.advance() }
            try await encodePrefillHead(state: state, lastRow: rows - 1)
            chunkedPrefillDirty = false
        } catch {
            // The runner may have written window/compressed rows without a
            // matching cursor commit. A fresh runner is required after this.
            throw error
        }
    }

    private func executePrefillLayer(layer: Int,
                                     tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     rows: Int,
                                     state: V4ChunkedPrefillState) async throws {
        let kind = cache.layerKind(layer)
        let ropeConfig = V4RoPE.Config.forLayer(kind)
        guard let cb = queue.makeCommandBuffer() else { throw MetalError.noQueue }

        // Attention boundary and input norm.
        let hcAttnFn = try model.hcAttnFn(layer: layer)
        let hcAttnBase = try model.hcAttnBase(layer: layer)
        let hcAttnScale = try model.hcAttnScale(layer: layer)
        state.boundary.encodeParams(commandBuffer: cb,
                                    x: state.stream,
                                    hcFn: hcAttnFn.buffer, hcFnOffset: Int(hcAttnFn.offset),
                                    hcBase: hcAttnBase.buffer, hcBaseOffset: Int(hcAttnBase.offset),
                                    hcScale: hcAttnScale.buffer, hcScaleOffset: Int(hcAttnScale.offset),
                                    rows: rows, dim: dim,
                                    normEps: eps, hcEps: Float(model.config.hcEps))
        state.boundary.encodePre(commandBuffer: cb,
                                 x: state.stream,
                                 out: state.branch,
                                 rows: rows, dim: dim)
        let attnNorm = try gammaF32(model.attnNorm(layer: layer),
                                    name: "layers.\(layer).attn_norm")
        state.boundary.encodeRMSNorm(commandBuffer: cb,
                                     x: state.branch,
                                     gamma: attnNorm,
                                     out: state.xNorm,
                                     rows: rows, n: dim, eps: eps)

        var weights = try attentionWeights(layer: layer, kind: kind)
        if kind != .passthrough {
            let compressorWKV = try model.compressorWKV(layer: layer)
            let compressorWGate = try model.compressorWGate(layer: layer)
            weights.compressorWKV = compressorWKV.buffer
            weights.compressorWGate = compressorWGate.buffer
            weights.compressorWKVOffset = Int(compressorWKV.offset)
            weights.compressorWGateOffset = Int(compressorWGate.offset)
            weights.compressorOutDim = kind == .csa ? 1024 : 512
        }

        var qkvOutputs = V4BatchedQKVCompressorPrefillStage.Outputs(
            qOut: state.q,
            windowKVOut: state.windowKV)
        if kind != .passthrough {
            qkvOutputs.compressorWKVOut = state.compressorKV
            qkvOutputs.compressorWGateOut = state.compressorGate
        }
        if kind == .csa {
            qkvOutputs.indexQOut = state.indexQ
            let indexWeightsProjection = try model.indexerWeightsProj(layer: layer)
            state.proj.encodeBF16GEMM(commandBuffer: cb,
                                      weights: indexWeightsProjection.buffer,
                                      weightsOffset: Int(indexWeightsProjection.offset),
                                      x: state.xNorm,
                                      out: state.indexWeights,
                                      rows: rows,
                                      m: model.config.indexNHeads,
                                      n: dim,
                                      outFP16: false)
            let indexerWKV = try model.indexerCompressorWKV(layer: layer)
            let indexerWGate = try model.indexerCompressorWGate(layer: layer)
            state.proj.encodeBF16GEMM(commandBuffer: cb,
                                      weights: indexerWKV.buffer,
                                      weightsOffset: Int(indexerWKV.offset),
                                      x: state.xNorm,
                                      out: state.indexerCompressorKV,
                                      rows: rows, m: 256, n: dim,
                                      outFP16: false)
            state.proj.encodeBF16GEMM(commandBuffer: cb,
                                      weights: indexerWGate.buffer,
                                      weightsOffset: Int(indexerWGate.offset),
                                      x: state.xNorm,
                                      out: state.indexerCompressorGate,
                                      rows: rows, m: 256, n: dim,
                                      outFP16: false)
        }
        state.qkvStage.encode(commandBuffer: cb,
                              hiddenRows: state.xNorm,
                              rowCount: rows,
                              startPosition: startPosition,
                              weights: weights,
                              rope: ropeConfig,
                              outputs: qkvOutputs,
                              normEps: eps)

        let sinks = try model.attnSink(layer: layer)
        var compressorWeights: V4ChunkedPrefillExecutor.CompressorWeights?
        if kind != .passthrough {
            let ape = try model.compressorAPE(layer: layer)
            let gamma = try model.compressorNorm(layer: layer)
            if kind == .csa {
                let indexerAPE = try model.indexerCompressorAPE(layer: layer)
                let indexerGamma = try model.indexerCompressorNorm(layer: layer)
                compressorWeights = .init(ape: ape.buffer, apeOffset: Int(ape.offset),
                                          gamma: gamma.buffer, gammaOffset: Int(gamma.offset),
                                          indexerAPE: indexerAPE.buffer,
                                          indexerAPEOffset: Int(indexerAPE.offset),
                                          indexerGamma: indexerGamma.buffer,
                                          indexerGammaOffset: Int(indexerGamma.offset))
            } else {
                compressorWeights = .init(ape: ape.buffer, apeOffset: Int(ape.offset),
                                          gamma: gamma.buffer, gammaOffset: Int(gamma.offset))
            }
        }
        let executorInputs = V4ChunkedPrefillExecutor.Inputs(
            q: state.q,
            windowKV: state.windowKV,
            indexQ: kind == .csa ? state.indexQ : nil,
            indexWeights: kind == .csa ? state.indexWeights : nil,
            compressorKV: kind == .passthrough ? nil : state.compressorKV,
            compressorGate: kind == .passthrough ? nil : state.compressorGate,
            indexerCompressorKV: kind == .csa ? state.indexerCompressorKV : nil,
            indexerCompressorGate: kind == .csa ? state.indexerCompressorGate : nil,
            sinks: sinks.buffer,
            sinksOffset: Int(sinks.offset),
            output: state.attn,
            rowStrideCompressor: (kind == .csa ? 1024 : 512) * MemoryLayout<Float>.stride)
        state.executors[layer].encode(commandBuffer: cb,
                                      layer: layer,
                                      startPosition: startPosition,
                                      rowCount: rows,
                                      inputs: executorInputs,
                                      compressorWeights: compressorWeights,
                                      rope: ropeConfig,
                                      normEps: eps)

        // De-rotate each query-head output at its positive absolute position,
        // then apply the checkpoint's grouped low-rank output projection.
        state.boundary.encodeRoPE(commandBuffer: cb,
                                  x: state.attn,
                                  positions: state.repeatedHeadPositions,
                                  rows: rows * model.config.numHeads,
                                  width: model.config.headDim,
                                  ropeDim: model.config.qkRopeHeadDim,
                                  inverse: true,
                                  config: ropeConfig)
        let woA = try model.woA(layer: layer)
        let woB = try model.woB(layer: layer)
        state.proj.encodeGroupedOProjDown(commandBuffer: cb,
                                          attn: state.attn,
                                          woAWeights: woA.buffer,
                                          woAWeightsOffset: Int(woA.offset),
                                          woAScales: woA.buffer,
                                          woAScalesOffset: Int(woA.scaleOffset),
                                          rows: rows,
                                          lowRank: state.lowRank)
        state.proj.encodeOProjUp(commandBuffer: cb,
                                 lowRank: state.lowRank,
                                 woBWeights: woB.buffer,
                                 woBWeightsOffset: Int(woB.offset),
                                 woBScales: woB.buffer,
                                 woBScalesOffset: Int(woB.scaleOffset),
                                 rows: rows,
                                 out: state.oProj)
        state.boundary.encodePost(commandBuffer: cb,
                                  residual: state.stream,
                                  sublayer: state.oProj,
                                  out: state.stream,
                                  rows: rows, dim: dim)

        // FFN boundary and batched router/shared-expert work.
        let hcFfnFn = try model.hcFfnFn(layer: layer)
        let hcFfnBase = try model.hcFfnBase(layer: layer)
        let hcFfnScale = try model.hcFfnScale(layer: layer)
        state.boundary.encodeParams(commandBuffer: cb,
                                    x: state.stream,
                                    hcFn: hcFfnFn.buffer, hcFnOffset: Int(hcFfnFn.offset),
                                    hcBase: hcFfnBase.buffer, hcBaseOffset: Int(hcFfnBase.offset),
                                    hcScale: hcFfnScale.buffer, hcScaleOffset: Int(hcFfnScale.offset),
                                    rows: rows, dim: dim,
                                    normEps: eps, hcEps: Float(model.config.hcEps))
        state.boundary.encodePre(commandBuffer: cb,
                                 x: state.stream,
                                 out: state.branch,
                                 rows: rows, dim: dim)
        let ffnNorm = try gammaF32(model.ffnNorm(layer: layer),
                                   name: "layers.\(layer).ffn_norm")
        state.boundary.encodeRMSNorm(commandBuffer: cb,
                                     x: state.branch,
                                     gamma: ffnNorm,
                                     out: state.xNorm,
                                     rows: rows, n: dim, eps: eps)

        let router = try model.routerWeight(layer: layer)
        state.proj.encodeBF16GEMM(commandBuffer: cb,
                                  weights: router.buffer,
                                  weightsOffset: Int(router.offset),
                                  x: state.xNorm,
                                  out: state.routerLogits,
                                  rows: rows, m: numExperts, n: dim,
                                  outFP16: false)
        let hashRouted = model.config.isHashRouted(layer: layer)
        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        var hashExpertIDs: [UInt32] = []
        if hashRouted {
            hashExpertIDs.reserveCapacity(rows * topK)
            for token in tokenIDs {
                hashExpertIDs.append(contentsOf:
                    try model.hashExpertIDs(layer: layer, token: token).map(UInt32.init))
            }
            state.uploadTokenIDs(tokenIDs)
            let routePointer = state.routeIDs.contents().assumingMemoryBound(to: UInt32.self)
            for (index, expert) in hashExpertIDs.enumerated() { routePointer[index] = expert }
            state.glue.encodeHashRouterWeights(commandBuffer: cb,
                                                logits: state.routerLogits,
                                                tid2eid: state.routeIDs,
                                                outWeights: state.routeWeights,
                                                rows: rows,
                                                routeScale: routeScale)
        } else {
            let bias = try model.routerBias(layer: layer)
            state.glue.encodeRouterTop6(commandBuffer: cb,
                                        logits: state.routerLogits,
                                        staticBias: bias.buffer,
                                        staticBiasOffset: Int(bias.offset),
                                        outIDs: state.routeIDs,
                                        outWeights: state.routeWeights,
                                        rows: rows,
                                        routeScale: routeScale)
        }

        try encodePrefillSharedExpert(commandBuffer: cb,
                                      layer: layer,
                                      rows: rows,
                                      state: state)
        cb.commit()
        await cb.completed()
        if let error = cb.error { throw error }

        let expertIDs: [UInt32]
        if hashRouted {
            expertIDs = hashExpertIDs
        } else {
            let pointer = state.routeIDs.contents().assumingMemoryBound(to: UInt32.self)
            expertIDs = (0..<(rows * topK)).map { pointer[$0] }
        }
        var pairs: [PrefillTokenExpertPair] = []
        pairs.reserveCapacity(rows * topK)
        for row in 0..<rows {
            for rank in 0..<topK {
                pairs.append(PrefillTokenExpertPair(
                    token: UInt32(row),
                    expert: expertIDs[row * topK + rank],
                    rank: UInt32(rank),
                    weight: 0))
            }
        }
        let routed = try await state.routedMoE.encode(
            provider: model,
            hidden: state.xNorm,
            pairs: pairs,
            routeWeights: state.routeWeights,
            layer: layer,
            queryCount: rows,
            numExperts: numExperts,
            d: dim,
            routedIntermediate: ffn,
            hiddenStrideElements: dim)

        guard let postCB = queue.makeCommandBuffer() else { throw MetalError.noQueue }
        state.glue.encodeAddF16(commandBuffer: postCB,
                                a: routed.output,
                                b: state.sharedOut,
                                out: state.routedPlusShared,
                                count: rows * dim)
        state.boundary.encodePost(commandBuffer: postCB,
                                  residual: state.stream,
                                  sublayer: state.routedPlusShared,
                                  out: state.stream,
                                  rows: rows, dim: dim)
        postCB.commit()
        await postCB.completed()
        if let error = postCB.error { throw error }
    }

    private func encodePrefillSharedExpert(commandBuffer cb: MTLCommandBuffer,
                                           layer: Int,
                                           rows: Int,
                                           state: V4ChunkedPrefillState) throws {
        let w1 = try model.sharedExpertW1(layer: layer)
        let w3 = try model.sharedExpertW3(layer: layer)
        let w2 = try model.sharedExpertW2(layer: layer)
        state.proj.encodeFP8GEMM(commandBuffer: cb,
                                 weights: w1.buffer, weightsOffset: Int(w1.offset),
                                 scales: w1.buffer, scalesOffset: Int(w1.scaleOffset),
                                 x: state.xNorm,
                                 out: state.sharedGate,
                                 rows: rows, m: ffn, n: dim,
                                 outFP16: true)
        state.proj.encodeFP8GEMM(commandBuffer: cb,
                                 weights: w3.buffer, weightsOffset: Int(w3.offset),
                                 scales: w3.buffer, scalesOffset: Int(w3.scaleOffset),
                                 x: state.xNorm,
                                 out: state.sharedUp,
                                 rows: rows, m: ffn, n: dim,
                                 outFP16: true)
        glue.encodeSwiGLUAct(commandBuffer: cb,
                             gate: state.sharedGate,
                             up: state.sharedUp,
                             act: state.sharedAct,
                             n: UInt32(rows * ffn),
                             limit: swigluLimit)
        state.proj.encodeFP8GEMM(commandBuffer: cb,
                                 weights: w2.buffer, weightsOffset: Int(w2.offset),
                                 scales: w2.buffer, scalesOffset: Int(w2.scaleOffset),
                                 x: state.sharedAct,
                                 out: state.sharedOut,
                                 rows: rows, m: dim, n: ffn,
                                 outFP16: true)
    }

    private func encodePrefillHead(state: V4ChunkedPrefillState,
                                   lastRow: Int) async throws {
        guard let cb = queue.makeCommandBuffer() else { throw MetalError.noQueue }
        let streamOffset = lastRow * streams * dim * MemoryLayout<Float>.stride
        let hcHeadFn = try model.hcHeadFn
        let hcHeadBase = try model.hcHeadBase
        let hcHeadScale = try model.hcHeadScale
        hc.encodeHeadParams(commandBuffer: cb,
                            x: state.stream, xOffset: streamOffset,
                            hcFn: hcHeadFn.buffer, hcFnOffset: Int(hcHeadFn.offset),
                            hcBase: hcHeadBase.buffer, hcBaseOffset: Int(hcHeadBase.offset),
                            hcScale: hcHeadScale.buffer, hcScaleOffset: Int(hcHeadScale.offset),
                            dim: dim, normEps: eps,
                            hcEps: Float(model.config.hcEps))
        hc.encodePre(commandBuffer: cb,
                     x: state.stream, xOffset: streamOffset,
                     out: branch, dim: dim)
        let finalNorm = try gammaF32(model.finalNorm, name: "norm")
        glue.encodeRMSNormF32In(commandBuffer: cb,
                                x: branch,
                                w: finalNorm,
                                out: xNorm,
                                dim: UInt32(dim), eps: eps)
        glue.encodeGemvF32(commandBuffer: cb,
                           weights: headF32,
                           x: xNorm,
                           out: logits,
                           m: UInt32(model.config.vocabSize),
                           n: UInt32(dim))
        cb.commit()
        await cb.completed()
        if let error = cb.error { throw error }
    }

    // MARK: - Per-layer assembly

    private func attentionWeights(layer: Int, kind: V4LayerKind) throws
        -> V4QKVEpilogue.Weights {
        let wqA = try model.wqA(layer: layer)
        let wqB = try model.wqB(layer: layer)
        let wkv = try model.wkv(layer: layer)
        let qNorm = try model.attnQNorm(layer: layer)
        let kvNorm = try model.attnKVNorm(layer: layer)
        let qNormF32 = try gammaF32(qNorm, name: "layers.\(layer).attn.q_norm")
        let kvNormF32 = try gammaF32(kvNorm, name: "layers.\(layer).attn.kv_norm")
        var w = V4QKVEpilogue.Weights(
            wqA: (wqA.buffer, wqA.buffer),
            wqB: (wqB.buffer, wqB.buffer),
            qNormGamma: qNormF32,
            windowWKV: (wkv.buffer, wkv.buffer),
            kvNormGamma: kvNormF32,
            compressorWKV: nil, compressorWGate: nil, indexerWqB: nil)
        w.wqACodesOffset = Int(wqA.offset)
        w.wqAScalesOffset = Int(wqA.scaleOffset)
        w.wqBCodesOffset = Int(wqB.offset)
        w.wqBScalesOffset = Int(wqB.scaleOffset)
        w.qNormGammaOffset = 0
        w.windowWKVCodesOffset = Int(wkv.offset)
        w.windowWKVScalesOffset = Int(wkv.scaleOffset)
        w.kvNormGammaOffset = 0
        if kind == .csa {
            let iqb = try model.indexerWQB(layer: layer)
            w.indexerWqB = (iqb.buffer, iqb.buffer)
            w.indexerWqBCodesOffset = Int(iqb.offset)
            w.indexerWqBScalesOffset = Int(iqb.scaleOffset)
        }
        return w
    }

    /// Stage this token's compressor projection rows into the group
    /// accumulators. The epilogue writes one row at offset 0 of the
    /// scratch buffers; rows land by position within the current group.
    private func stageCompressorRows(commandBuffer cb: MTLCommandBuffer,
                                     layer: Int, kind: V4LayerKind, position: Int) {
        guard kind != .passthrough else { return }
        let ratio = model.config.compressRatios[layer]
        let rowInGroup = position % ratio
        // Compressor weights ship BF16; projections land fp32 (one row).
        let outDim = UInt32(kind == .csa ? 1024 : 512)
        if let cwkv = try? model.compressorWKV(layer: layer),
           let cwg = try? model.compressorWGate(layer: layer) {
            glue.encodeBF16GEMV(commandBuffer: cb,
                                weights: cwkv.buffer, weightsOffset: Int(cwkv.offset),
                                x: xNorm, out: rowScratch,
                                m: outDim, d: UInt32(dim))
            glue.encodeBF16GEMV(commandBuffer: cb,
                                weights: cwg.buffer, weightsOffset: Int(cwg.offset),
                                x: xNorm, out: rowScratch2,
                                m: outDim, d: UInt32(dim))
        }
        if kind == .csa {
            let state = compressorAccumulators.csa(layer: layer)
            blitRow(commandBuffer: cb, from: rowScratch, to: state.curKV,
                    rowBytes: 1024 * 4, row: rowInGroup)
            blitRow(commandBuffer: cb, from: rowScratch2, to: state.curGate,
                    rowBytes: 1024 * 4, row: rowInGroup)
            // Indexer series: 256-dim rows (two 128-dim overlapped series).
            if let ikv = try? model.indexerCompressorWKV(layer: layer),
               let ig = try? model.indexerCompressorWGate(layer: layer) {
                glue.encodeBF16GEMV(commandBuffer: cb,
                                    weights: ikv.buffer, weightsOffset: Int(ikv.offset),
                                    x: xNorm, out: rowScratch,
                                    m: 256, d: UInt32(dim))
                glue.encodeBF16GEMV(commandBuffer: cb,
                                    weights: ig.buffer, weightsOffset: Int(ig.offset),
                                    x: xNorm, out: rowScratch2,
                                    m: 256, d: UInt32(dim))
                blitRow(commandBuffer: cb, from: rowScratch, to: state.idxCurKV,
                        rowBytes: 256 * 4, row: rowInGroup)
                blitRow(commandBuffer: cb, from: rowScratch2, to: state.idxCurGate,
                        rowBytes: 256 * 4, row: rowInGroup)
            }
        } else {
            let state = compressorAccumulators.hca(layer: layer)
            blitRow(commandBuffer: cb, from: rowScratch, to: state.ringKV,
                    rowBytes: 512 * 4, row: rowInGroup)
            blitRow(commandBuffer: cb, from: rowScratch2, to: state.ringGate,
                    rowBytes: 512 * 4, row: rowInGroup)
        }
    }

    private func blitRow(commandBuffer cb: MTLCommandBuffer,
                         from src: MTLBuffer, to dst: MTLBuffer,
                         rowBytes: Int, row: Int) {
        guard let blit = cb.makeBlitCommandEncoder() else { return }
        blit.copy(from: src, sourceOffset: 0,
                  to: dst, destinationOffset: row * rowBytes,
                  size: rowBytes)
        blit.endEncoding()
    }

    /// Flush the group completing at `position` into the cache's compressed
    /// slots (and, for CSA, the indexer slot), then roll the current group
    /// accumulators into the previous-group slots for the overlap.
    private func flushGroup(commandBuffer cb: MTLCommandBuffer,
                            layer: Int, kind: V4LayerKind, position: Int,
                            ropeCfg: V4RoPE.Config) {
        let group = cache.groupIndex(layer: layer, tokenPosition: position)
        let slot = cache.compressedSlot(layer: layer, group: group)
        let ropePos = UInt32(cache.ropePosition(layer: layer, group: group))
        switch kind {
        case .csa:
            let state = compressorAccumulators.csa(layer: layer)
            attention.encodeCSACompressGroup(
                commandBuffer: cb,
                prevKV: state.prevKV, curKV: state.curKV,
                prevGate: state.prevGate, curGate: state.curGate,
                ape: (try? model.compressorAPE(layer: layer))?.buffer ?? rowScratch,
                gamma: (try? model.compressorNorm(layer: layer))?.buffer ?? rowScratch,
                outValues: slot.values.buffer, valuesOffset: slot.values.offset,
                outScales: slot.scales.buffer, scalesOffset: slot.scales.offset,
                outRope: slot.rope.buffer, ropeOffset: slot.rope.offset,
                ropePosition: ropePos,
                ropeTheta: Float(model.config.compressRopeTheta),
                yarnFactor: Float(model.config.yarnFactor),
                originalSeqLen: Float(model.config.yarnOriginalMaxPositions),
                betaFast: Float(model.config.yarnBetaFast),
                betaSlow: Float(model.config.yarnBetaSlow),
                useYarn: true, normEps: eps)
            let idxSlot = cache.indexerSlot(layer: layer, group: group)
            glue.encodeIndexerCompressGroup(
                commandBuffer: cb,
                prevKV: state.idxPrevKV, curKV: state.idxCurKV,
                prevGate: state.idxPrevGate, curGate: state.idxCurGate,
                ape: (try? model.indexerCompressorAPE(layer: layer))?.buffer ?? rowScratch,
                gamma: (try? model.indexerCompressorNorm(layer: layer))?.buffer ?? rowScratch,
                out: idxSlot.buffer, outOffset: idxSlot.offset,
                ropePosition: ropePos, rope: ropeCfg, normEps: eps)
            // Roll: the current group becomes the previous group for the
            // next entry's overlapped pooling window.
            compressorAccumulators.rollCSA(commandBuffer: cb, layer: layer)
        case .hca:
            let state = compressorAccumulators.hca(layer: layer)
            hca.encodeGroup(commandBuffer: cb,
                            kv: state.ringKV, gate: state.ringGate,
                            ape: (try? model.compressorAPE(layer: layer))?.buffer ?? rowScratch,
                            gamma: (try? model.compressorNorm(layer: layer))?.buffer ?? rowScratch,
                            outValues: slot.values.buffer, valuesOffset: slot.values.offset,
                            outScales: slot.scales.buffer, scalesOffset: slot.scales.offset,
                            outRope: slot.rope.buffer, ropeOffset: slot.rope.offset,
                            ropePosition: ropePos, rope: ropeCfg, normEps: eps)
        case .passthrough:
            break
        }
    }

    private func encodeAttention(commandBuffer cb: MTLCommandBuffer,
                                 layer: Int, kind: V4LayerKind,
                                 tokenCount: Int,
                                 ropeCfg: V4RoPE.Config) throws {
        let sinks = try model.attnSink(layer: layer)
        let window = cache.windowBuffer(layer: layer)
        let windowStart = max(0, tokenCount - model.config.slidingWindow)
        switch kind {
        case .passthrough:
            attention.encodeWindowMQADecode(commandBuffer: cb, q: q,
                                            windowK: window, tokenCount: tokenCount,
                                            sinks: sinks.buffer,
                                            sinksOffset: Int(sinks.offset),
                                            out: attnOut)
        case .csa:
            // Indexer per-head weights: h . W^w (BF16 [indexNHeads, dim]).
            let wproj = try model.indexerWeightsProj(layer: layer)
            glue.encodeBF16GEMV(commandBuffer: cb,
                                weights: wproj.buffer, weightsOffset: Int(wproj.offset),
                                x: xNorm, out: indexW,
                                m: UInt32(model.config.indexNHeads), d: UInt32(dim))
            let base = cache.compressedSlot(layer: layer, group: 0)
            attention.encodeCSADecode(commandBuffer: cb, q: q,
                                      indexQ: indexQ,
                                      indexKV: cache.indexerBuffer(layer: layer),
                                      indexWeights: indexW,
                                      nVisible: cache.visibleGroupCount(
                                          layer: layer, windowStart: windowStart),
                                      compressedValues: base.values.buffer,
                                      compressedScales: base.scales.buffer,
                                      compressedRope: base.rope.buffer,
                                      windowK: window, tokenCount: tokenCount,
                                      sinks: sinks.buffer, sinksOffset: Int(sinks.offset),
                                      out: attnOut)
        case .hca:
            let base = cache.compressedSlot(layer: layer, group: 0)
            attention.encodeHCADecode(commandBuffer: cb, q: q,
                                      nVisible: cache.visibleGroupCount(
                                          layer: layer, windowStart: windowStart),
                                      compressedValues: base.values.buffer,
                                      compressedScales: base.scales.buffer,
                                      compressedRope: base.rope.buffer,
                                      windowK: window, tokenCount: tokenCount,
                                      sinks: sinks.buffer, sinksOffset: Int(sinks.offset),
                                      out: attnOut)
        }
    }

    /// Resident shared expert: FP8 w1/w3 -> clamped SwiGLU -> FP8 w2.
    private func encodeSharedExpert(commandBuffer cb: MTLCommandBuffer,
                                    layer: Int) throws {
        let w1 = try model.sharedExpertW1(layer: layer)
        let w3 = try model.sharedExpertW3(layer: layer)
        let w2 = try model.sharedExpertW2(layer: layer)
        fp8.encode(commandBuffer: cb,
                   weights: w1.buffer, weightsOffset: Int(w1.offset),
                   scales: w1.buffer, scalesOffset: Int(w1.scaleOffset),
                   x: xNorm, y: sharedGate,
                   m: UInt32(ffn), n: UInt32(dim))
        fp8.encode(commandBuffer: cb,
                   weights: w3.buffer, weightsOffset: Int(w3.offset),
                   scales: w3.buffer, scalesOffset: Int(w3.scaleOffset),
                   x: xNorm, y: sharedUp,
                   m: UInt32(ffn), n: UInt32(dim))
        glue.encodeSwiGLUAct(commandBuffer: cb, gate: sharedGate, up: sharedUp,
                             act: sharedAct, n: UInt32(ffn), limit: swigluLimit)
        fp8.encode(commandBuffer: cb,
                   weights: w2.buffer, weightsOffset: Int(w2.offset),
                   scales: w2.buffer, scalesOffset: Int(w2.scaleOffset),
                   x: sharedAct, y: sharedOut,
                   m: UInt32(dim), n: UInt32(ffn))
    }

    // MARK: - Small utilities

    /// Norm gammas ship BF16; the RMSNorm kernels want fp32 gammas.
    /// Exact one-time widening per tensor, cached.
    private var stagedGammas: [String: MTLBuffer] = [:]

    private func gammaF32(_ view: TensorView, name: String) throws -> MTLBuffer {
        if let staged = stagedGammas[name] { return staged }
        let count = Int(view.length) / 2
        guard let staged = device.makeBuffer(length: count * 4,
                                             options: .storageModeShared) else {
            throw MetalError.missingFunction("v4 gamma staging: \(name)")
        }
        widenBF16ToF32(src: view.buffer.contents().advanced(by: Int(view.offset)),
                       dst: staged.contents(), count: count)
        stagedGammas[name] = staged
        return staged
    }

    private func uploadInts(_ values: [Int32], to buffer: MTLBuffer) {
        let ptr = buffer.contents().assumingMemoryBound(to: Int32.self)
        for (i, v) in values.enumerated() { ptr[i] = v }
    }

    private func uploadUInts(_ values: [UInt32], to buffer: MTLBuffer) {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for (i, v) in values.enumerated() { ptr[i] = v }
    }
}

/// Exact BF16 -> FP32 widening (bf16 is the top 16 bits of fp32).
private func widenBF16ToF32(src: UnsafeMutableRawPointer,
                            dst: UnsafeMutableRawPointer,
                            count: Int) {
    let s = src.assumingMemoryBound(to: UInt16.self)
    let d = dst.assumingMemoryBound(to: UInt32.self)
    for i in 0..<count {
        d[i] = UInt32(s[i]) << 16
    }
}
