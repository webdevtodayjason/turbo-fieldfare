import Foundation
import Metal
import Darwin

/// Loaded DeepSeek V4-family `.gturbo/` model (modelFamily
/// "deepseek-v4-flash", versionMinor 1). Sibling of `Model` for the V4
/// schema: shares the resident buffer/index, packed-experts layout, SHA-256
/// verification, and `PreadExpertStreamer` infrastructure, but reads the
/// DeepSeek-keyed arch dict via `ManifestReader.loadV4` and exposes the
/// flat V4 tensor names (`layers.N.attn.wq_a.weight`, ...). The Gemma
/// `Model` type is untouched.
///
/// Resident dtype conventions (recon V4F-reference-notes §7): dense
/// projections FP8 e4m3 + ue8m0 block scales (TensorView scale offsets);
/// router gate + indexer weights_proj BF16; norm gammas, compressor
/// wkv/wgate, hc_* params, attn_sink, ape, lm_head FP32; tid2eid I64;
/// embed BF16.
public struct V4Model {
    public let device: MTLDevice
    public let config: V4ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: V4Manifest
    let directoryURL: URL

    /// Lazy state, same reference-box pattern as `Model`.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue

    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(device: MTLDevice,
         config: V4ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffer: ResidentBuffer,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: V4Manifest,
         directoryURL: URL) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "turbo-fieldfare.v4-expert-streamers")
    }

    // MARK: - Resident accessors (flat DeepSeek tensor names)

    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let residentFileOffset = residentIndex.header.indexSize
        let relativeOffset = entry.fileOffset - residentFileOffset
        let scaleRel: UInt64 = entry.scaleSize > 0
            ? entry.scaleOffset - residentFileOffset : 0
        return TensorView(
            buffer: residentBuffer.buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset: 0, biasLength: 0,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    /// Embedding table, BF16 [vocab, hidden]. Untied from `head`.
    public var embedding: TensorView {
        get throws { try resident(name: "embed.weight") }
    }
    /// LM head, FP32 [vocab, hidden].
    public var lmHead: TensorView {
        get throws { try resident(name: "head.weight") }
    }
    /// Final RMSNorm gamma, FP32 [hidden].
    public var finalNorm: TensorView {
        get throws { try resident(name: "norm.weight") }
    }

    // Attention projections (FP8 e4m3 + ue8m0 block scales).
    public func wqA(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.wq_a.weight")
    }
    public func wqB(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.wq_b.weight")
    }
    public func wkv(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.wkv.weight")
    }
    public func woA(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.wo_a.weight")
    }
    public func woB(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.wo_b.weight")
    }
    /// Per-head norms and sinks (FP32).
    public func attnQNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.q_norm.weight")
    }
    public func attnKVNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.kv_norm.weight")
    }
    public func attnSink(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.attn_sink")
    }
    /// Sublayer input norms (FP32).
    public func attnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn_norm.weight")
    }
    public func ffnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).ffn_norm.weight")
    }

    // Compressor (CSA/HCA layers; BF16 or FP32 per the checkpoint).
    public func compressorWKV(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.compressor.wkv.weight")
    }
    public func compressorWGate(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.compressor.wgate.weight")
    }
    /// Compressor RMSNorm gamma (FP32; consumed by the compress kernels).
    public func compressorNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.compressor.norm.weight")
    }
    /// Absolute positional embedding (FP32 [ratio, 2*headDim]).
    public func compressorAPE(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.compressor.ape")
    }

    // Lightning indexer (CSA layers only).
    public func indexerWQB(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.indexer.wq_b.weight")
    }
    public func indexerWeightsProj(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.indexer.weights_proj.weight")
    }
    public func indexerCompressorWKV(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.indexer.compressor.wkv.weight")
    }
    public func indexerCompressorWGate(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.indexer.compressor.wgate.weight")
    }
    public func indexerCompressorNorm(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.indexer.compressor.norm.weight")
    }
    public func indexerCompressorAPE(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).attn.indexer.compressor.ape")
    }

    // FFN: router, hash table, shared expert.
    /// Router gate weight, BF16 [numExperts, hidden].
    public func routerWeight(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).ffn.gate.weight")
    }
    /// Static noaux_tc correction bias, F32 [numExperts] (router layers only).
    public func routerBias(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).ffn.gate.bias")
    }
    /// Fixed hash-routing table, I64 [vocab, topK] (hash layers only).
    public func hashTable(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).ffn.gate.tid2eid")
    }
    public func sharedExpertW1(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).ffn.shared_experts.w1.weight")
    }
    public func sharedExpertW2(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).ffn.shared_experts.w2.weight")
    }
    public func sharedExpertW3(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).ffn.shared_experts.w3.weight")
    }

    // mHC boundary parameters (FP32).
    public func hcAttnFn(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).hc_attn_fn")
    }
    public func hcAttnBase(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).hc_attn_base")
    }
    public func hcAttnScale(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).hc_attn_scale")
    }
    public func hcFfnFn(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).hc_ffn_fn")
    }
    public func hcFfnBase(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).hc_ffn_base")
    }
    public func hcFfnScale(layer L: Int) throws -> TensorView {
        try resident(name: "layers.\(L).hc_ffn_scale")
    }
    public var hcHeadFn: TensorView {
        get throws { try resident(name: "hc_head_fn") }
    }
    public var hcHeadBase: TensorView {
        get throws { try resident(name: "hc_head_base") }
    }
    public var hcHeadScale: TensorView {
        get throws { try resident(name: "hc_head_scale") }
    }

    // MARK: - Hash routing

    /// The fixed top-K expert ids for `token` on a hash-routed layer: row
    /// `token` of the resident I64 `tid2eid` table. Cheap CPU read (48
    /// bytes at production shape); hash layers need no router kernel.
    public func hashExpertIDs(layer L: Int, token: UInt32) throws -> [Int] {
        precondition(config.isHashRouted(layer: L),
                     "hashExpertIDs on a router layer \(L)")
        let table = try hashTable(layer: L)
        let topK = config.topKExperts
        let row = Int(token) * topK
        precondition(table.dtype == 5, "tid2eid must be I64 (dtype 5)")
        let ptr = table.buffer.contents()
            .advanced(by: Int(table.offset) + row * MemoryLayout<Int64>.stride)
            .assumingMemoryBound(to: Int64.self)
        var ids = [Int](repeating: 0, count: topK)
        for i in 0..<topK {
            ids[i] = max(0, min(Int(ptr[i]), config.numExperts - 1))
        }
        return ids
    }

    // MARK: - Routed experts (lazy streaming)

    /// Per-expert sub-tensor offsets for the fused MoE kernels, from the
    /// packed layout (6 regions: gate/up/down weights + ue8m0 scales).
    public func routedExpertV4Offsets(layer: Int) -> V4ExpertOffsets {
        let expert = packedExpertsLayout.expert(layer: layer, expert: 0)
        func offset(_ role: String) -> UInt32 {
            UInt32(expert.subTensors[role]?.offset ?? 0)
        }
        return V4ExpertOffsets(
            gateWOff: offset("gate"),
            gateSOff: offset("gate_scales"),
            upWOff: offset("up"),
            upSOff: offset("up_scales"),
            downWOff: offset("down"),
            downSOff: offset("down_scales"))
    }

    public func planRoutedExperts(layer: Int,
                                  experts: [Int],
                                  avoidingSlots: Set<Int> = []) throws -> RoutedExpertFetchPlan {
        try ensureLayerOpened(layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[layer]! }
        let validSlots = Set(avoidingSlots.filter { $0 >= 0 && $0 < streamer.slotCount })
        return RoutedExpertFetchPlan(
            layer: layer,
            cachePlan: streamer.planExpertsCached(experts: experts,
                                                  avoidingSlots: validSlots))
    }

    public func adviseRoutedExperts(plan: RoutedExpertFetchPlan) throws -> ExpertIOAdviceResult {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.adviseExpertCachePlanMisses(plan.cachePlan)
    }

    /// Slot buffers for a plan without issuing I/O: hit slots already hold
    /// valid data; miss slots are filled by `fetchRoutedExperts`. Used to
    /// encode the hit-subset phase 1 before the pread completes.
    public func routedExpertBuffers(for plan: RoutedExpertFetchPlan) throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return streamer.expertCachePlanBuffers(plan.cachePlan).enumerated().map { index, entry in
            TensorView(
                buffer: entry.buffer,
                offset: entry.offset,
                length: entry.size,
                scaleOffset: 0, scaleLength: 0,
                biasOffset: 0, biasLength: 0,
                shape: (UInt32(plan.layer), UInt32(plan.experts[index]), 0, 0),
                dtype: 0)
        }
    }

    public func fetchRoutedExperts(plan: RoutedExpertFetchPlan) async throws -> [TensorView] {
        try ensureLayerOpened(plan.layer)
        let streamer = streamersQueue.sync { streamersBox.streamers[plan.layer]! }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let buffers = try streamer.executeExpertCachePlan(plan.cachePlan)
                    continuation.resume(returning: buffers.enumerated().map { index, entry in
                        TensorView(
                            buffer: entry.buffer,
                            offset: entry.offset,
                            length: entry.size,
                            scaleOffset: 0, scaleLength: 0,
                            biasOffset: 0, biasLength: 0,
                            shape: (UInt32(plan.layer), UInt32(plan.experts[index]), 0, 0),
                            dtype: 0)
                    })
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook, mirroring `Model`.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            try? model.openLayerLocked(L)
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ModelError.missingFile(name: "packed_experts/\(basename)")
        }
        if !streamersBox.layerVerified[L] {
            let manifestRel = "packed_experts/\(basename)"
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(at: url, named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                try Self.verifyTrustedReceiptFileSize(url: url,
                                                      relativePath: manifestRel,
                                                      expectedSize: entry.size)
            }
            streamersBox.layerVerified[L] = true
        }
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer)
            * packedExpertsLayout.expertStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: slotCount,
            cachePolicy: expertCachePolicy)
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }
}

extension V4Model {

    /// Open a V4-family `.gturbo/` directory. Eagerly verifies SHA-256 of
    /// `model_weights.bin` and `packed_experts/layout.json`; layer files
    /// are verified lazily on first stream open, matching `Model.load`.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: V4ArchConfig,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> V4Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }

        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let family = try ManifestReader.probeModelFamily(directoryURL: directoryURL)
        guard family == ManifestReader.deepSeekV4FlashFamily else {
            throw ModelError.unsupportedModelFamily(found: family ?? "<absent>")
        }

        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = try Sha256Verifier.hashFile(at: manifestURL)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart
        let manifestSize = try Self.fileSize(at: manifestURL,
                                             relativePath: "manifest.json")
        let receipt: VerifiedInstallReceipt?
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let loadedReceipt = try VerifiedInstallReceiptReader.load(directoryURL: directoryURL)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                loadedReceipt,
                directoryURL: directoryURL,
                manifestSha256: manifestSha)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
            receipt = loadedReceipt
        } else {
            receipt = nil
        }

        let manifest = try ManifestReader.loadV4(directoryURL: directoryURL,
                                                 expecting: expecting)
        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(receipt,
                                                      directoryURL: directoryURL,
                                                      manifest: manifest,
                                                      manifestSha256: manifestSha,
                                                      manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        let layoutURL  = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent("layout.json")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try Sha256Verifier.verifyFile(at: weightsURL, named: "model_weights.bin",
                                      expectedHex: weightsEntry.sha256)
        try Sha256Verifier.verifyFile(at: layoutURL, named: "packed_experts/layout.json",
                                      expectedHex: layoutEntry.sha256)
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        let residentIndex = try ResidentIndexReader.load(fileURL: weightsURL)

        // The resident index must account for the complete weights file.
        let attrs = try FileManager.default.attributesOfItem(atPath: weightsURL.path)
        if let fileSize = attrs[.size] as? UInt64 {
            let expected = residentIndex.header.indexSize + residentIndex.header.residentSize
            if fileSize != expected {
                throw ModelError.indexCorrupt(detail: """
                    model_weights.bin size \(fileSize) != indexSize \
                    \(residentIndex.header.indexSize) + residentSize \
                    \(residentIndex.header.residentSize) = \(expected)
                    """)
            }
        }

        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device)

        let layout = try PackedExpertsLayoutReader.load(directoryURL: directoryURL)
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(directoryURL: directoryURL,
                                                  manifest: manifest,
                                                  layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        return V4Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL)
    }

    private static func verifyTrustedReceiptFileSize(url: URL,
                                                     relativePath: String,
                                                     expectedSize: UInt64) throws {
        let actualSize = try fileSize(at: url, relativePath: relativePath)
        guard actualSize == expectedSize else {
            throw ModelError.trustedReceiptInvalid(
                detail: "\(relativePath) size \(actualSize) != \(expectedSize)")
        }
    }

    private static func fileSize(at url: URL, relativePath: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let sizeValue = attrs[.size] else {
            throw ModelError.trustedReceiptInvalid(detail: "missing size for \(relativePath)")
        }
        if let number = sizeValue as? NSNumber { return number.uint64Value }
        if let value = sizeValue as? UInt64 { return value }
        if let value = sizeValue as? Int { return UInt64(value) }
        throw ModelError.trustedReceiptInvalid(detail: "invalid size for \(relativePath)")
    }

    private static func validateTrustedReceiptLayerLayout(directoryURL: URL,
                                                          manifest: V4Manifest,
                                                          layout: PackedExpertsLayout) throws {
        let pageSize = UInt64(getpagesize())
        guard layout.expertStride % pageSize == 0 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "expertStride \(layout.expertStride) is not page-aligned")
        }
        guard layout.numLayers == manifest.numLayers,
              layout.expertsPerLayer == manifest.expertsPerLayer,
              layout.expertStride == manifest.expertStride else {
            throw ModelError.trustedReceiptInvalid(detail: "layout does not match manifest dimensions")
        }
        for layer in layout.layers {
            guard layer.layer >= 0 && layer.layer < manifest.numLayers else {
                throw ModelError.trustedReceiptInvalid(detail: "layout layer out of range")
            }
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(detail: "manifest missing \(relativePath)")
            }
            let expectedSize = UInt64(layout.expertsPerLayer) * layout.expertStride
            guard manifestEntry.size == expectedSize else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedSize)")
            }
            let url = directoryURL
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(layer.file)
            let actualSize = try fileSize(at: url, relativePath: relativePath)
            guard actualSize == expectedSize else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(expectedSize)")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw ModelError.trustedReceiptInvalid(detail: "\(relativePath) expert count mismatch")
            }
            for expert in layer.experts {
                guard expert.size == layout.expertStride else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) size mismatch")
                }
                guard expert.offset % pageSize == 0 else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) offset is not page-aligned")
                }
                guard expert.offset <= actualSize,
                      expert.size <= actualSize - expert.offset else {
                    throw ModelError.trustedReceiptInvalid(
                        detail: "\(relativePath) expert \(expert.expert) range exceeds file size")
                }
            }
        }
    }
}
