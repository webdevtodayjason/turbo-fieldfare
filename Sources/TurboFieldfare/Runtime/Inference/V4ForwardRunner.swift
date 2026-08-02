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
/// Recorded v1 simplifications (follow-ups, not oversights):
///
/// - Layers are serialized: cb3 completes before the next layer's plan,
///   so a later layer's cache plan can never evict a slot that queued GPU
///   work still owns. Cross-layer pipelining with `avoidingSlots` is the
///   V4F-04 follow-up that recovers the Gemma pipeline depth.
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
    private let slotUpload: MTLBuffer    // [6] uint32 active-slot list
    private let idUpload: MTLBuffer      // [6] int32 hash-layer expert ids
    private let logits: MTLBuffer        // [vocab] fp32
    // Compressor group accumulators (fp32; kernels read from offset 0).
    private let csaPrevKV: MTLBuffer     // [4, 1024] fp32
    private let csaCurKV: MTLBuffer      // [4, 1024] fp32
    private let csaPrevGate: MTLBuffer   // [4, 1024] fp32
    private let csaCurGate: MTLBuffer    // [4, 1024] fp32
    private let idxPrevKV: MTLBuffer     // [4, 256] fp32 (indexer series)
    private let idxCurKV: MTLBuffer      // [4, 256] fp32
    private let idxPrevGate: MTLBuffer   // [4, 256] fp32
    private let idxCurGate: MTLBuffer    // [4, 256] fp32
    private let lowRank: MTLBuffer       // [8192] fp16 o-proj stage 1
    private let headF32: MTLBuffer       // [vocab, dim] fp32 (converted)
    private let hcaRingKV: MTLBuffer     // [128, 512] fp32
    private let hcaRingGate: MTLBuffer   // [128, 512] fp32
    private let rowScratch: MTLBuffer    // [1024] fp32 epilogue single row
    private let rowScratch2: MTLBuffer   // [1024] fp32 second series

    /// Temporary debug tracing (TURBO_V4_DEBUG=1): per-layer activation
    /// stats for the first tokens. Remove with the debug session.
    private let debugTrace = ProcessInfo.processInfo.environment["TURBO_V4_DEBUG"] == "1"

    public init(model: V4Model, maxContext: Int) throws {
        self.model = model
        self.device = model.device
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
        self.slotUpload = try scratch(4, "slotUpload")
        self.idUpload = try scratch(4, "idUpload")
        self.logits = try scratch(model.config.vocabSize, "logits")
        self.csaPrevKV = try scratch(4 * 1024, "csaPrevKV")
        self.csaCurKV = try scratch(4 * 1024, "csaCurKV")
        self.csaPrevGate = try scratch(4 * 1024, "csaPrevGate")
        self.csaCurGate = try scratch(4 * 1024, "csaCurGate")
        self.idxPrevKV = try scratch(4 * 256, "idxPrevKV")
        self.idxCurKV = try scratch(4 * 256, "idxCurKV")
        self.idxPrevGate = try scratch(4 * 256, "idxPrevGate")
        self.idxCurGate = try scratch(4 * 256, "idxCurGate")
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
        self.hcaRingKV = try scratch(128 * 512, "hcaRingKV")
        self.hcaRingGate = try scratch(128 * 512, "hcaRingGate")
        self.rowScratch = try scratch(1024, "rowScratch")
        self.rowScratch2 = try scratch(1024, "rowScratch2")
    }

    // MARK: - Decode step

    /// Embed `token` into the fp32 mHC stream, then run one decode token
    /// through the full stack. Returns the fp32 logits buffer (valid until
    /// the next call). `position` is the absolute 0-based token index.
    public func forward(token: UInt32, position: Int) async throws -> MTLBuffer {
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
                if debugTrace {
                    // CPU-side rms of the resident embed row for this token.
                    let row = table.buffer.contents()
                        .advanced(by: Int(table.offset) + Int(token) * dim * 2)
                        .assumingMemoryBound(to: UInt16.self)
                    var acc = Float(0)
                    for i in 0..<dim {
                        let b = UInt32(row[i]) << 16
                        let v = Float(bitPattern: b)
                        acc += v * v
                    }
                    print(String(format: "V4DBG pos=%d embed token=%d rowRms=%.5f",
                                 position, token, (acc / Float(dim)).squareRoot()))
                }
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

            rope.encode(commandBuffer: cb1, x: attnOut,
                        rows: model.config.numHeads, width: model.config.headDim,
                        ropeDim: model.config.qkRopeHeadDim,
                        position: -Float(position), inverse: true, config: ropeCfg)
            // Grouped output projection as two FP8 GEMVs (wo_a is
            // F8_E4M3 [8192, 4096] group-major; wo_b is [4096, 8192]).
            let woA = try model.woA(layer: layer)
            let woB = try model.woB(layer: layer)
            fp8.encode(commandBuffer: cb1,
                       weights: woA.buffer, weightsOffset: Int(woA.offset),
                       scales: woA.buffer, scalesOffset: Int(woA.scaleOffset),
                       x: attnOut, y: lowRank,
                       m: UInt32(model.config.oGroups * model.config.oLoraRank),
                       n: UInt32(dim))
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
            await cb1.completed()

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
            let argBuffer = moe.makeReusedRoutedArgumentBuffer(routedBlobs: blobs,
                                                               topK: UInt32(topK))

            // ---- cb2: shared expert + hit-subset phase 1 ------------------
            guard let cb2 = queue.makeCommandBuffer() else { throw MetalError.noQueue }
            try encodeSharedExpert(commandBuffer: cb2, layer: layer)
            if isHash {
                let gate = try model.routerWeight(layer: layer)
                glue.encodeBF16GEMV(commandBuffer: cb2,
                                    weights: gate.buffer, weightsOffset: Int(gate.offset),
                                    x: xNorm, out: routerLogits,
                                    m: UInt32(numExperts), d: UInt32(dim))
                uploadInts(expertIDs.map { Int32($0) }, to: idUpload)
                glue.encodeRouterWeightsAtIndices(commandBuffer: cb2,
                                                  logits: routerLogits,
                                                  indices: idUpload,
                                                  outWeights: routerWts,
                                                  k: UInt32(topK),
                                                  routeScale: routeScale)
            }
            if !hitPositions.isEmpty {
                uploadUInts(hitPositions.map { UInt32($0) }, to: slotUpload)
                moe.encodeRoutedPhase1SwiGLUSubset(commandBuffer: cb2,
                                                   routedArgBuffer: argBuffer,
                                                   routedBlobs: blobs,
                                                   routedOffsets: offsets,
                                                   x: xNorm, acts: acts,
                                                   activeSlots: slotUpload,
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
                uploadUInts(missPositions.map { UInt32($0) }, to: slotUpload)
                moe.encodeRoutedPhase1SwiGLUSubset(commandBuffer: cb3,
                                                   routedArgBuffer: argBuffer,
                                                   routedBlobs: blobs,
                                                   routedOffsets: offsets,
                                                   x: xNorm, acts: acts,
                                                   activeSlots: slotUpload,
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
            await cb3.completed()

            if debugTrace {
                func stats16(_ b: MTLBuffer, _ n: Int) -> (Float, Float, Float) {
                    let p = b.contents().assumingMemoryBound(to: Float16.self)
                    var s = Float(0), lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
                    for i in 0..<n { let v = Float(p[i]); s += v * v; lo = min(lo, v); hi = max(hi, v) }
                    return ((s / Float(n)).squareRoot(), lo, hi)
                }
                func stats32(_ b: MTLBuffer, _ n: Int) -> (Float, Float, Float) {
                    let p = b.contents().assumingMemoryBound(to: Float.self)
                    var s = Float(0), lo = Float.greatestFiniteMagnitude, hi = -Float.greatestFiniteMagnitude
                    for i in 0..<n { let v = p[i]; s += v * v; lo = min(lo, v); hi = max(hi, v) }
                    return ((s / Float(n)).squareRoot(), lo, hi)
                }
                let xs = stats32(stream, 4 * dim)
                let at = stats16(attnOut, 64 * 512)
                let ff = stats16(ffnOut, dim)
                let xn = stats16(xNorm, dim)
                let ac = stats16(acts, 6 * ffn)
                let so = stats16(sharedOut, dim)
                let wp = routerWts.contents().assumingMemoryBound(to: Float.self)
                // CPU cross-check: dequant slot-0 gate from the LIVE blob
                // and project the LIVE xNorm, comparing scale vs kernel.
                if layer == 0 {
                    let E2M1: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6,
                                         0, -0.5, -1, -1.5, -2, -3, -4, -6]
                    let xp = xNorm.contents().assumingMemoryBound(to: Float16.self)
                    let F = 2048, D = 4096, PACK = D / 2
                    var cpuPerSlot = ""
                    for sIdx in 0..<6 {
                        let codes = blobs[sIdx].contents().assumingMemoryBound(to: UInt8.self)
                        let scales = blobs[sIdx].contents().advanced(by: Int(offsets.gateSOff))
                            .assumingMemoryBound(to: UInt8.self)
                        let upCodes = blobs[sIdx].contents().advanced(by: Int(offsets.upWOff))
                            .assumingMemoryBound(to: UInt8.self)
                        let upScales = blobs[sIdx].contents().advanced(by: Int(offsets.upSOff))
                            .assumingMemoryBound(to: UInt8.self)
                        var accAct = Float(0); var cnt = 0
                        for row in stride(from: 0, to: F, by: 64) {
                            var g = Float(0); var u = Float(0)
                            for col in 0..<PACK {
                                let byte = codes[row * PACK + col]
                                let sc = scales[row * (D / 32) + col / 16]
                                let scale = powf(2, Float(Int(sc) - 127))
                                g += E2M1[Int(byte & 0x0F)] * scale * Float(xp[col * 2])
                                g += E2M1[Int(byte >> 4)] * scale * Float(xp[col * 2 + 1])
                                let ub = upCodes[row * PACK + col]
                                let us = upScales[row * (D / 32) + col / 16]
                                let usc = powf(2, Float(Int(us) - 127))
                                u += E2M1[Int(ub & 0x0F)] * usc * Float(xp[col * 2])
                                u += E2M1[Int(ub >> 4)] * usc * Float(xp[col * 2 + 1])
                            }
                            let gc = min(g, Float(10))
                            let act = (gc / (1 + exp(-gc))) * max(-10, min(10, u))
                            accAct += act * act; cnt += 1
                        }
                        cpuPerSlot += String(format: " s%d=%.3f", sIdx,
                                             (accAct / Float(cnt)).squareRoot())
                    }
                    print("V4DBG L00 CPU per-slot acts rms:" + cpuPerSlot)
                    // Dump raw inputs/outputs for offline analysis.
                    let dumpDir = "/Users/sem/code/turbo-fieldfare/scratch/v4f-recon"
                    try? Data(bytes: xNorm.contents(), count: dim * 2)
                        .write(to: URL(fileURLWithPath: dumpDir + "/dbg-xNorm.bin"))
                    try? Data(bytes: acts.contents(), count: 6 * ffn * 2)
                        .write(to: URL(fileURLWithPath: dumpDir + "/dbg-acts.bin"))
                    try? Data(bytes: blobs[1].contents(), count: blobs[1].length)
                        .write(to: URL(fileURLWithPath: dumpDir + "/dbg-blob1.bin"))
                    try? Data(bytes: blobs[0].contents(), count: blobs[0].length)
                        .write(to: URL(fileURLWithPath: dumpDir + "/dbg-blob0.bin"))
                    // Blob identity probe: first 16 bytes of each slot's gate
                    // region, plus a CPU recomputation of slot-1 gate rms
                    // using ALL rows (not sampled).
                    for sIdx in 0..<3 {
                        let b = blobs[sIdx].contents().assumingMemoryBound(to: UInt8.self)
                        var hex = ""
                        for i in 0..<16 { hex += String(format: "%02x", b[i]) }
                        print(String(format: "V4DBG L00 blob%d first16: %@  expertIDs=%d", sIdx, hex, expertIDs[sIdx]))
                    }
                    do {
                        let codes = blobs[1].contents().assumingMemoryBound(to: UInt8.self)
                        let scales = blobs[1].contents().advanced(by: Int(offsets.gateSOff))
                            .assumingMemoryBound(to: UInt8.self)
                        let xp = xNorm.contents().assumingMemoryBound(to: Float16.self)
                        var acc = Float(0); var cnt = 0
                        let E2M1f: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6,
                                              0, -0.5, -1, -1.5, -2, -3, -4, -6]
                        for row in 0..<2048 {
                            var g = Float(0)
                            for col in 0..<2048 {
                                let byte = codes[row * 2048 + col]
                                let sc = scales[row * 128 + col / 16]
                                let sc2 = powf(2, Float(Int(sc) - 127))
                                g += E2M1f[Int(byte & 0x0F)] * sc2 * Float(xp[col * 2])
                                g += E2M1f[Int(byte >> 4)] * sc2 * Float(xp[col * 2 + 1])
                            }
                            acc += g * g; cnt += 1
                        }
                        print(String(format: "V4DBG L00 blob1 cpu gate rms(all rows)=%.4f",
                                     (acc / Float(cnt)).squareRoot()))
                    }
                    // Retry probe: re-run the identical phase1 subset in a
                    // fresh command buffer and re-measure. If correct, the
                    // inputs were fine and cb3's execution was the problem.
                    // Bisect: copy live blob contents into fresh makeBuffer
                    // blobs and rebind. If results turn correct, the
                    // bytesNoCopy slot buffers are the problem; if still
                    // wrong, xNorm is.
                    var freshBlobs: [MTLBuffer] = []
                    for b in blobs {
                        let nb = device.makeBuffer(length: b.length, options: .storageModeShared)!
                        nb.contents().copyMemory(from: b.contents(), byteCount: b.length)
                        freshBlobs.append(nb)
                    }
                    let freshArg = moe.makeReusedRoutedArgumentBuffer(routedBlobs: freshBlobs, topK: 6)
                    if let cbF = queue.makeCommandBuffer() {
                        moe.encodeRoutedPhase1SwiGLU(commandBuffer: cbF,
                            routedArgBuffer: freshArg, routedBlobs: freshBlobs,
                            routedOffsets: offsets, x: xNorm, acts: acts,
                            d: UInt32(dim), f: UInt32(ffn), topK: UInt32(topK))
                        cbF.commit()
                        await cbF.completed()
                        var fresh = ""
                        for sIdx in 0..<6 {
                            let ptr = acts.contents().advanced(by: sIdx * ffn * 2)
                            let hp = ptr.assumingMemoryBound(to: Float16.self)
                            var ss = Float(0)
                            for i in 0..<ffn { let v = Float(hp[i]); ss += v * v }
                            fresh += String(format: " s%d=%.3f", sIdx, (ss / Float(ffn)).squareRoot())
                        }
                        print("V4DBG L00 FRESH-BLOB kernel per-slot acts rms:" + fresh)
                    }
                    if let cbR = queue.makeCommandBuffer() {
                        let missPos = (0..<6).map { UInt32($0) }
                        uploadUInts(missPos, to: slotUpload)
                        moe.encodeRoutedPhase1SwiGLU(commandBuffer: cbR,
                            routedArgBuffer: argBuffer, routedBlobs: blobs,
                            routedOffsets: offsets, x: xNorm, acts: acts,
                            d: UInt32(dim), f: UInt32(ffn),
                            topK: UInt32(topK))
                        cbR.commit()
                        await cbR.completed()
                        let gateAfter = stats16(sharedOut, ffn)
                        print(String(format: "V4DBG L00 standalone gate rms(blob1)=%.4f (cpu gate rms was ~0.33)", gateAfter.0))
                        // Row-level split: kernel gate rows vs CPU rows.
                        let kout = sharedOut.contents().assumingMemoryBound(to: Float16.self)
                        var kvals = ""
                        for r in 0..<4 { kvals += String(format: " r%d=%.4f", r, Float(kout[r])) }
                        print("V4DBG L00 kernel gate rows:" + kvals)
                        do {
                            let codes = blobs[1].contents().assumingMemoryBound(to: UInt8.self)
                            let scales = blobs[1].contents().advanced(by: Int(offsets.gateSOff))
                                .assumingMemoryBound(to: UInt8.self)
                            let xp = xNorm.contents().assumingMemoryBound(to: Float16.self)
                            let E2M1f: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6,
                                                  0, -0.5, -1, -1.5, -2, -3, -4, -6]
                            var cvals = ""
                            for row in 0..<4 {
                                var g = Float(0)
                                for col in 0..<2048 {
                                    let byte = codes[row * 2048 + col]
                                    let sc = scales[row * 128 + col / 16]
                                    let sc2 = powf(2, Float(Int(sc) - 127))
                                    g += E2M1f[Int(byte & 0x0F)] * sc2 * Float(xp[col * 2])
                                    g += E2M1f[Int(byte >> 4)] * sc2 * Float(xp[col * 2 + 1])
                                }
                                cvals += String(format: " r%d=%.4f", row, g)
                            }
                            print("V4DBG L00 cpu    gate rows:" + cvals)
                        }
                        var retry = ""
                        for sIdx in 0..<6 {
                            let ptr = acts.contents().advanced(by: sIdx * ffn * 2)
                            let hp = ptr.assumingMemoryBound(to: Float16.self)
                            var ss = Float(0)
                            for i in 0..<ffn { let v = Float(hp[i]); ss += v * v }
                            retry += String(format: " s%d=%.3f", sIdx,
                                            (ss / Float(ffn)).squareRoot())
                        }
                        print("V4DBG L00 RETRY kernel per-slot acts rms:" + retry)
                        let gateBefore = stats16(sharedOut, ffn)
                        print(String(format: "V4DBG L00 standalone-gate-before rms=%.4f", gateBefore.0))
                        // Split probe: standalone FP4 GEMV for slot 1 gate.
                        let gateOut = rowScratch   // >= 2048 fp32? rowScratch is fp32 [1024]; use logits instead
                        _ = gateOut
                        fp4.encode(commandBuffer: cbR, weights: blobs[1],
                                   weightsOffset: Int(offsets.gateWOff),
                                   scales: blobs[1], scalesOffset: Int(offsets.gateSOff),
                                   x: xNorm, y: sharedOut, m: 2048, n: 4096)
                    }
                    let codes = blobs[0].contents().assumingMemoryBound(to: UInt8.self)
                    let scales = blobs[0].contents().advanced(by: Int(offsets.gateSOff))
                        .assumingMemoryBound(to: UInt8.self)
                    var acc = Float(0); var count = 0
                    for row in stride(from: 0, to: F, by: 64) {
                        var g = Float(0)
                        for col in 0..<PACK {
                            let byte = codes[row * PACK + col]
                            let sc = scales[row * (D / 32) + col / 16]
                            let scale = powf(2, Float(Int(sc) - 127))
                            g += E2M1[Int(byte & 0x0F)] * scale * Float(xp[col * 2])
                            g += E2M1[Int(byte >> 4)] * scale * Float(xp[col * 2 + 1])
                        }
                        acc += g * g; count += 1
                    }
                    var accUp = Float(0); var accAct = Float(0)
                    for row in stride(from: 0, to: F, by: 64) {
                        var g = Float(0); var u = Float(0)
                        let upCodes = blobs[0].contents().advanced(by: Int(offsets.upWOff))
                            .assumingMemoryBound(to: UInt8.self)
                        let upScales = blobs[0].contents().advanced(by: Int(offsets.upSOff))
                            .assumingMemoryBound(to: UInt8.self)
                        for col in 0..<PACK {
                            let byte = codes[row * PACK + col]
                            let sc = scales[row * (D / 32) + col / 16]
                            let scale = powf(2, Float(Int(sc) - 127))
                            g += E2M1[Int(byte & 0x0F)] * scale * Float(xp[col * 2])
                            g += E2M1[Int(byte >> 4)] * scale * Float(xp[col * 2 + 1])
                            let ub = upCodes[row * PACK + col]
                            let us = upScales[row * (D / 32) + col / 16]
                            let usc = powf(2, Float(Int(us) - 127))
                            u += E2M1[Int(ub & 0x0F)] * usc * Float(xp[col * 2])
                            u += E2M1[Int(ub >> 4)] * usc * Float(xp[col * 2 + 1])
                        }
                        let gc = min(g, Float(10))
                        let act = (gc / (1 + exp(-gc))) * max(-10, min(10, u))
                        accUp += u * u; accAct += act * act
                    }
                    var perSlot = ""
                    for sIdx in 0..<6 {
                        let ptr = acts.contents().advanced(by: sIdx * ffn * 2)
                        var ss = Float(0)
                        let hp = ptr.assumingMemoryBound(to: Float16.self)
                        for i in 0..<ffn { let v = Float(hp[i]); ss += v * v }
                        perSlot += String(format: " s%d=%.3f", sIdx, (ss / Float(ffn)).squareRoot())
                    }
                    let actsSlot0 = stats16(acts, ffn)
                    print("V4DBG L00 per-slot acts rms:" + perSlot)
                    print(String(format: "V4DBG L00 cpuGate rms=%.4f cpuUp rms=%.4f cpuAct rms=%.4f kernelActs(slot0) rms=%.4f xRms=%.4f",
                                 (acc / Float(count)).squareRoot(),
                                 (accUp / Float(count)).squareRoot(),
                                 (accAct / Float(count)).squareRoot(),
                                 actsSlot0.0, stats16(xNorm, dim).0))
                    print(String(format: "V4DBG L00 offsets g=%d gs=%d u=%d us=%d d=%d ds=%d blobLen=%d",
                                 offsets.gateWOff, offsets.gateSOff, offsets.upWOff,
                                 offsets.upSOff, offsets.downWOff, offsets.downSOff,
                                 blobs[0].length))
                }
                print(String(format: "V4DBG pos=%d L%02d stream rms=%.4f [%.3f,%.3f] xNorm rms=%.4f attn rms=%.4f [%.2f,%.2f] ffn rms=%.4f [%.2f,%.2f] acts rms=%.4f shared rms=%.4f w=[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]",
                             position, layer, xs.0, xs.1, xs.2, xn.0, at.0, at.1, at.2, ff.0, ff.1, ff.2,
                             ac.0, so.0, wp[0], wp[1], wp[2], wp[3], wp[4], wp[5]))
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
            blitRow(commandBuffer: cb, from: rowScratch, to: csaCurKV,
                    rowBytes: 1024 * 4, row: rowInGroup)
            blitRow(commandBuffer: cb, from: rowScratch2, to: csaCurGate,
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
                blitRow(commandBuffer: cb, from: rowScratch, to: idxCurKV,
                        rowBytes: 256 * 4, row: rowInGroup)
                blitRow(commandBuffer: cb, from: rowScratch2, to: idxCurGate,
                        rowBytes: 256 * 4, row: rowInGroup)
            }
        } else {
            blitRow(commandBuffer: cb, from: rowScratch, to: hcaRingKV,
                    rowBytes: 512 * 4, row: rowInGroup)
            blitRow(commandBuffer: cb, from: rowScratch2, to: hcaRingGate,
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

    private func blitWhole(commandBuffer cb: MTLCommandBuffer,
                           from src: MTLBuffer, to dst: MTLBuffer, bytes: Int) {
        guard let blit = cb.makeBlitCommandEncoder() else { return }
        blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0,
                  size: bytes)
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
            attention.encodeCSACompressGroup(
                commandBuffer: cb,
                prevKV: csaPrevKV, curKV: csaCurKV,
                prevGate: csaPrevGate, curGate: csaCurGate,
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
                prevKV: idxPrevKV, curKV: idxCurKV,
                prevGate: idxPrevGate, curGate: idxCurGate,
                ape: (try? model.indexerCompressorAPE(layer: layer))?.buffer ?? rowScratch,
                gamma: (try? model.indexerCompressorNorm(layer: layer))?.buffer ?? rowScratch,
                out: idxSlot.buffer, outOffset: idxSlot.offset,
                ropePosition: ropePos, rope: ropeCfg, normEps: eps)
            // Roll: the current group becomes the previous group for the
            // next entry's overlapped pooling window.
            blitWhole(commandBuffer: cb, from: csaCurKV, to: csaPrevKV, bytes: 4 * 1024 * 4)
            blitWhole(commandBuffer: cb, from: csaCurGate, to: csaPrevGate, bytes: 4 * 1024 * 4)
            blitWhole(commandBuffer: cb, from: idxCurKV, to: idxPrevKV, bytes: 4 * 128 * 4)
            blitWhole(commandBuffer: cb, from: idxCurGate, to: idxPrevGate, bytes: 4 * 128 * 4)
        case .hca:
            hca.encodeGroup(commandBuffer: cb,
                            kv: hcaRingKV, gate: hcaRingGate,
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
