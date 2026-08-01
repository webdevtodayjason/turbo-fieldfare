import Foundation

/// On-disk page alignment unit for `.gturbo` files. Fixed at 16 KB regardless
/// of host page size — the format is the contract, not the kernel.
enum Layout {
    static let pageBytes: UInt64 = 16_384
}

// MARK: - Plan data types

struct ResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    /// File offset where the (packed) weight bytes start.
    let fileOffset: UInt64
    /// Size in bytes of the weight bytes.
    let sizeBytes: UInt64
    /// Offset where BF16 scales start (0 if none).
    let scaleOffset: UInt64
    let scaleSize: UInt64
    /// Offset where BF16 biases start (0 if none).
    let biasOffset: UInt64
    let biasSize: UInt64
    /// Quantization spec (nil for unquantized scalars/norms).
    let quantSpec: QuantSpec?

    /// Source tensors that supply this entry's bytes.
    let sourceWeight: SourceTensor
    let sourceScales: SourceTensor?
    let sourceBiases: SourceTensor?
}

struct ResidentFilePlan: Sendable {
    let path: String
    let entries: [ResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]   // per-entry offsets into the table
    let indexSize: UInt64              // header + entries + table + padding
    let residentSize: UInt64           // tensor payload region
    var totalSize: UInt64 { indexSize + residentSize }
}

struct PerExpertTensorSlice: Sendable {
    let role: String                   // "gate" | "up" | "down"
    let component: String              // "weights" | "scales" | "biases"
    /// dtype byte: 0=U32, 1=BF16, 2=FP16, 3=FP32, 4=I8, 5=I64,
    /// 6=F8_E4M3, 7=F8_E8M0 (mirrors `SourceTensor.Dtype` raw values).
    let dtype: UInt8
    let logicalShape: [UInt64]         // per-expert logical shape
    let offsetInExpertBlob: UInt64     // within each expert blob
    let sizeInExpertBlob: UInt64
    /// Stacked rank-3 source: stride per expert within `sourceTensor`.
    let sourceOffsetPerExpert: UInt64
    let sourceTensor: SourceTensor
    /// Per-expert 2D sources (DeepSeek V4): one source tensor per expert, in
    /// expert order. When non-nil, overrides the stacked-source stride pair
    /// (experts may live in different shards at unrelated offsets).
    let perExpertSources: [SourceTensor]?
    let bitsForWeights: Int?           // 4 for routed expert weight; nil for scales/biases

    /// Source tensor supplying expert `e`'s bytes for this slice.
    func source(forExpert expert: Int) -> SourceTensor {
        if let perExpertSources { return perExpertSources[expert] }
        return SourceTensor(name: sourceTensor.name,
                            shardPath: sourceTensor.shardPath,
                            dtype: sourceTensor.dtype,
                            shape: sourceTensor.shape,
                            absoluteOffset: sourceTensor.absoluteOffset
                                + UInt64(expert) * sourceOffsetPerExpert,
                            sizeBytes: sizeInExpertBlob)
    }
}

struct LayerFilePlan: Sendable {
    let layerIndex: Int
    let path: String
    let expertsPerLayer: Int
    let expertStride: UInt64
    let subTensors: [PerExpertTensorSlice]  // 9 entries: gate/up/down × {weights, scales, biases}
    var fileSize: UInt64 { UInt64(expertsPerLayer) * expertStride }

    func physicalRank(for logicalExpert: Int) -> Int {
        logicalExpert
    }

    init(layerIndex: Int,
                path: String,
                expertsPerLayer: Int,
                expertStride: UInt64,
                subTensors: [PerExpertTensorSlice]) {
        self.layerIndex = layerIndex
        self.path = path
        self.expertsPerLayer = expertsPerLayer
        self.expertStride = expertStride
        self.subTensors = subTensors
    }
}

struct RepackPlan: Sendable {
    let arch: PlanArch
    var family: ModelFamily { arch.family }
    let baseMode: String                  // "affine" | "fp8-blockwise-e4m3"
    let baseGroupSize: Int                // 64 (MLX) | 128 (DeepSeek FP8 block rows)
    let bitsOverrideCount: Int
    /// DeepSeek FP8/FP4 quant facts; nil for MLX-family plans.
    let deepseekQuant: DeepSeekQuantDescriptor?
    let resident: ResidentFilePlan
    let layers: [LayerFilePlan]
    let matchedModelID: String?
    let excludedMultimodalTensorNames: [String]
    /// MTP-module tensors deliberately not packed (V4 family), recorded in
    /// the install audit like multimodal drops.
    let droppedMTPTensorNames: [String]
}

// MARK: - Planner

enum RepackPlanner {

    /// Classify a tensor name. Routed-expert tensors split off the LM bucket.
    enum Bucket: Equatable {
        case lmResident
        case routedExpert(role: String, layer: Int)   // role = "gate"|"up"|"down"
        case excludedMultimodal
        case unknown
    }

    static func classify(_ name: String, numLayers: Int) -> Bucket {
        if name.hasPrefix("language_model.") {
            // Routed expert?
            if let role = routedExpertRole(in: name),
               let layer = layerIndex(in: name),
               layer >= 0 && layer < numLayers {
                return .routedExpert(role: role, layer: layer)
            }
            return .lmResident
        }
        if isMultimodalTensorName(name) {
            return .excludedMultimodal
        }
        return .unknown
    }

    private static func routedExpertRole(in name: String) -> String? {
        guard name.contains(".experts.switch_glu.") else { return nil }
        if name.contains(".gate_proj.") { return "gate" }
        if name.contains(".up_proj.")   { return "up" }
        if name.contains(".down_proj.") { return "down" }
        return nil
    }

    private static func layerIndex(in name: String) -> Int? {
        // matches "...layers.<N>...."
        guard let r = name.range(of: ".layers.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Build the plan from parsed shard headers + source metadata.
    /// - throws: classification + companion + override count failures.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: ArchInfo,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String) throws -> RepackPlan {
        try plan(meta: meta, arch: .gemma(arch),
                 shardHeaders: shardHeaders, outputDir: outputDir)
    }

    /// Family-forking entry point used by the remote installer.
    static func plan(meta: IndexLoader.SourceMetadata,
                            arch: PlanArch,
                            shardHeaders: [Safetensors.Header],
                            outputDir: String) throws -> RepackPlan {
        switch arch {
        case .gemma(let archInfo):
            return try planGemma(meta: meta, arch: archInfo,
                                 shardHeaders: shardHeaders, outputDir: outputDir)
        case .deepseekV4(let archInfo):
            return try planV4(meta: meta, arch: archInfo,
                              shardHeaders: shardHeaders, outputDir: outputDir)
        }
    }

    private static func planGemma(meta: IndexLoader.SourceMetadata,
                                  arch: ArchInfo,
                                  shardHeaders: [Safetensors.Header],
                                  outputDir: String) throws -> RepackPlan {
        // Companion tensors may live in different shards, so resolve them
        // through one global registry.
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        // Source allowlisting owns exact fingerprint validation. Preserve the
        // declared override count for the output manifest audit.
        let bitsOverrideCount = meta.bitsOverrides.count

        var lmResidentBases: [String] = []
        var excludedMultimodalNames: [String] = []
        var routedByLayerAndRole: [Int: [String: String]] = [:]
        for (name, _) in registry {
            if isMultimodalTensorName(name) {
                excludedMultimodalNames.append(name)
            }
            if name.hasSuffix(".scales") || name.hasSuffix(".biases") { continue }
            let b = classify(name, numLayers: arch.numLayers)
            switch b {
            case .lmResident:                   lmResidentBases.append(name)
            case .routedExpert(let role, let layer):
                var byRole = routedByLayerAndRole[layer] ?? [:]
                if byRole[role] != nil {
                    throw RepackError.configurationInvalid(detail:
                        "two routed-expert tensors for layer \(layer) role \(role)")
                }
                byRole[role] = name
                routedByLayerAndRole[layer] = byRole
            case .excludedMultimodal:           continue
            case .unknown:                      throw RepackError.unknownTensorPrefix(name: name)
            }
        }

        // Sort deterministically. The LM order follows a fixed template.
        lmResidentBases.sort(by: lmResidentOrdering())
        excludedMultimodalNames.sort()

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            baseNames: lmResidentBases,
                                            registry: registry, meta: meta)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let bundle = routedByLayerAndRole[layer] ?? [:]
            // Synthetic snapshots may legitimately have no routed experts.
            guard let gName = bundle["gate"], let uName = bundle["up"], let dName = bundle["down"] else {
                if bundle.isEmpty {
                    layerPlans.append(LayerFilePlan(layerIndex: layer,
                                                    path: (layersDir as NSString).appendingPathComponent("layer_\(String(format: "%02d", layer)).bin"),
                                                    expertsPerLayer: 0,
                                                    expertStride: 0,
                                                    subTensors: []))
                    continue
                }
                throw RepackError.configurationInvalid(detail:
                    "layer \(layer) routed-expert bundle incomplete: \(bundle)")
            }
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            let lp = try planLayerFile(path: path, layer: layer,
                                       gateName: gName, upName: uName, downName: dName,
                                       registry: registry, meta: meta, arch: arch)
            layerPlans.append(lp)
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)

        return RepackPlan(arch: .gemma(arch),
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: bitsOverrideCount,
                          deepseekQuant: nil,
                          resident: resident,
                          layers: layerPlans,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: excludedMultimodalNames,
                          droppedMTPTensorNames: [])
    }

    private static func isMultimodalTensorName(_ name: String) -> Bool {
        name.hasPrefix("vision_tower.") ||
            name.hasPrefix("embed_vision.") ||
            name.hasPrefix("audio_tower.")
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         baseNames: [String],
                                         registry: [String: SourceTensor],
                                         meta: IndexLoader.SourceMetadata) throws
                                        -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for n in baseNames {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        // Index size includes the fixed header, fixed-width entries, and the
        // string table, padded to a 16 KB page boundary.
        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for name in baseNames {
            guard let weight = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            let dtype = ietnyDtype(weight.dtype)
            let isQuantizedPacked = (weight.dtype == .u32) && name.hasSuffix(".weight")

            if isQuantizedPacked {
                let base = String(name.dropLast(".weight".count))
                guard let scales = registry[base + ".scales"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard let biases = registry[base + ".biases"] else {
                    throw RepackError.missingBiasesCompanion(name: name)
                }
                if scales.dtype != .bf16 || biases.dtype != .bf16 {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected BF16 scales/biases, got \(scales.dtype)/\(biases.dtype)")
                }
                let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
                let logical = logicalShape(forPackedSource: weight.shape, bits: spec.bits)

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = scales.sizeBytes
                let bOff = sOff + sSize
                let bSize = biases.sizeBytes
                fileCursor = bOff + bSize

                entries.append(ResidentEntry(
                    name: name, dtype: 0,
                    logicalShape4: padTo4(logical),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: bOff, biasSize: bSize,
                    quantSpec: spec,
                    sourceWeight: weight, sourceScales: scales, sourceBiases: biases))
            } else {
                // Unquantized (BF16 norm / scalar) — no companions.
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: dtype,
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize

        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    // MARK: - Layer planning

    private static func planLayerFile(path: String, layer: Int,
                                      gateName: String, upName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      meta: IndexLoader.SourceMetadata,
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, name: String)] = [
            ("gate", gateName), ("up", upName), ("down", downName)
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)
        var blobCursor: UInt64 = 0

        for (role, name) in roles {
            guard let w = registry[name] else { throw RepackError.missingTensor(name: name) }
            if w.dtype != .u32 || w.shape.count != 3 || Int(w.shape[0]) != expertCount {
                throw RepackError.shapeMismatch(name: name,
                    detail: "expected U32 rank-3 with leading \(expertCount), got \(w.dtype) \(w.shape)")
            }
            let base = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
            guard let s = registry[base + ".scales"] else { throw RepackError.missingScalesCompanion(name: name) }
            guard let b = registry[base + ".biases"] else { throw RepackError.missingBiasesCompanion(name: name) }
            if s.dtype != .bf16 || b.dtype != .bf16 {
                throw RepackError.dtypeMismatch(name: name,
                    detail: "expected BF16 scales/biases, got \(s.dtype)/\(b.dtype)")
            }

            let perExpertWeightSize = w.sizeBytes / UInt64(expertCount)
            let perExpertScaleSize  = s.sizeBytes / UInt64(expertCount)
            let perExpertBiasSize   = b.sizeBytes / UInt64(expertCount)
            if perExpertWeightSize * UInt64(expertCount) != w.sizeBytes ||
               perExpertScaleSize  * UInt64(expertCount) != s.sizeBytes ||
               perExpertBiasSize   * UInt64(expertCount) != b.sizeBytes {
                throw RepackError.shapeMismatch(name: name,
                    detail: "source bytes not evenly divisible by \(expertCount) experts")
            }

            let spec = IndexLoader.quantSpec(forTensor: name, meta: meta)
            let perExpertSourceShape = Array(w.shape.dropFirst())
            let logicalPerExpert = logicalShape(forPackedSource: perExpertSourceShape, bits: spec.bits)
            let scalesLogical = Array(s.shape.dropFirst())
            let biasesLogical = Array(b.shape.dropFirst())

            let wSlice = PerExpertTensorSlice(
                role: role, component: "weights", dtype: 0,
                logicalShape: logicalPerExpert,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertWeightSize,
                sourceOffsetPerExpert: perExpertWeightSize, sourceTensor: w,
                perExpertSources: nil,
                bitsForWeights: spec.bits)
            blobCursor += perExpertWeightSize
            let sSlice = PerExpertTensorSlice(
                role: role, component: "scales", dtype: 1,
                logicalShape: scalesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertScaleSize,
                sourceOffsetPerExpert: perExpertScaleSize, sourceTensor: s,
                perExpertSources: nil,
                bitsForWeights: nil)
            blobCursor += perExpertScaleSize
            let bSlice = PerExpertTensorSlice(
                role: role, component: "biases", dtype: 1,
                logicalShape: biasesLogical,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: perExpertBiasSize,
                sourceOffsetPerExpert: perExpertBiasSize, sourceTensor: b,
                perExpertSources: nil,
                bitsForWeights: nil)
            blobCursor += perExpertBiasSize

            subs.append(wSlice); subs.append(sSlice); subs.append(bSlice)
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - DeepSeek V4 planning

    /// V4-Flash plan: per-expert 2D FP4 source tensors grouped by
    /// (layer, role, expert), FP8 resident pairing (`.weight` + `.scale`),
    /// and explicit MTP-module drops.
    private static func planV4(meta: IndexLoader.SourceMetadata,
                               arch: ArchInfoV4,
                               shardHeaders: [Safetensors.Header],
                               outputDir: String) throws -> RepackPlan {
        guard meta.family == .deepseekV4Flash, let quant = meta.deepseekQuant else {
            throw RepackError.configurationInvalid(
                detail: "V4 plan requires DeepSeek quantization_config metadata")
        }

        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        var residentNames: [String] = []
        var droppedMTP: [String] = []
        var pendingScaleCompanions: [String] = []
        for (name, _) in registry {
            if name.hasPrefix("mtp.") {
                droppedMTP.append(name)
                continue
            }
            if let parsed = parseV4ExpertName(name) {
                guard parsed.layer >= 0, parsed.layer < arch.numLayers,
                      parsed.expert >= 0, parsed.expert < arch.numExperts else {
                    throw RepackError.unknownTensorPrefix(name: name)
                }
                continue  // consumed by the layer planner
            }
            if name.contains(".ffn.experts.") {
                // Expert-shaped but unparsable: never let it fall through to
                // the resident file silently.
                throw RepackError.unknownTensorPrefix(name: name)
            }
            if name.hasSuffix(".scale") {
                pendingScaleCompanions.append(name)  // consumed by FP8 pairing
                continue
            }
            guard isV4ResidentName(name, numLayers: arch.numLayers) else {
                throw RepackError.unknownTensorPrefix(name: name)
            }
            residentNames.append(name)
        }
        droppedMTP.sort()
        residentNames.sort(by: v4ResidentOrdering())

        var consumedScales = Set<String>()
        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planV4ResidentFile(path: residentPath,
                                              baseNames: residentNames,
                                              registry: registry,
                                              quant: quant,
                                              consumedScales: &consumedScales)
        for orphan in pendingScaleCompanions where !consumedScales.contains(orphan) {
            throw RepackError.configurationInvalid(
                detail: "orphan .scale companion \(orphan): no quantized weight pairs with it")
        }

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            layerPlans.append(try planV4LayerFile(path: path, layer: layer,
                                                  registry: registry, arch: arch))
        }

        let matched = SourceFingerprint.modelID(forIndexSha256: meta.indexSha256Hex)
        return RepackPlan(arch: .deepseekV4(arch),
                          baseMode: meta.baseMode,
                          baseGroupSize: meta.baseGroupSize,
                          bitsOverrideCount: meta.bitsOverrides.count,
                          deepseekQuant: quant,
                          resident: resident,
                          layers: layerPlans,
                          matchedModelID: matched,
                          excludedMultimodalTensorNames: [],
                          droppedMTPTensorNames: droppedMTP)
    }

    /// Parses `layers.<L>.ffn.experts.<E>.w<R>.<weight|scale>` into
    /// (layer, expert, role, component). Roles follow the reference SwiGLU:
    /// w1 = gate, w2 = down, w3 = up.
    private static func parseV4ExpertName(_ name: String)
        -> (layer: Int, expert: Int, role: String, component: String)? {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 7,
              parts[0] == "layers",
              parts[2] == "ffn",
              parts[3] == "experts",
              let layer = Int(parts[1]),
              let expert = Int(parts[4]) else { return nil }
        let role: String
        switch parts[5] {
        case "w1": role = "gate"
        case "w2": role = "down"
        case "w3": role = "up"
        default: return nil
        }
        let component: String
        switch parts[6] {
        case "weight": component = "weights"
        case "scale":  component = "scales"
        default: return nil
        }
        return (layer, expert, role, component)
    }

    private static func isV4ResidentName(_ name: String, numLayers: Int) -> Bool {
        switch name {
        case "embed.weight", "head.weight", "norm.weight",
             "hc_head_fn", "hc_head_base", "hc_head_scale":
            return true
        default:
            break
        }
        if let layer = v4LayerIndex(in: name),
           layer >= 0, layer < numLayers {
            return true
        }
        return false
    }

    /// Layer index for flat V4 names (`layers.<N>...`). The Gemma helper
    /// `layerIndex(in:)` anchors on `.layers.` mid-string; V4 names carry the
    /// prefix at the start.
    private static func v4LayerIndex(in name: String) -> Int? {
        guard name.hasPrefix("layers.") else { return nil }
        let tail = name.dropFirst("layers.".count)
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// V4 resident file: FP8 e4m3 weights pair with their `.scale` ue8m0
    /// block-scale companion (no biases); everything else (BF16 / F32 / I64)
    /// is copied unquantized.
    private static func planV4ResidentFile(path: String,
                                           baseNames: [String],
                                           registry: [String: SourceTensor],
                                           quant: DeepSeekQuantDescriptor,
                                           consumedScales: inout Set<String>) throws
        -> ResidentFilePlan {
        let entryCount = baseNames.count

        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(entryCount)
        for n in baseNames {
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: n.utf8)
        }

        let rawIdx = UInt64(GTurboBinary.indexHeaderBytes
            + entryCount * GTurboBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        var fileCursor = indexSize
        var entries: [ResidentEntry] = []
        entries.reserveCapacity(entryCount)

        for name in baseNames {
            guard let weight = registry[name] else {
                throw RepackError.missingTensor(name: name)
            }
            if weight.dtype == .f8e4m3 {
                guard name.hasSuffix(".weight") else {
                    throw RepackError.configurationInvalid(
                        detail: "F8_E4M3 tensor without .weight suffix: \(name)")
                }
                let base = String(name.dropLast(".weight".count))
                guard let scales = registry[base + ".scale"] else {
                    throw RepackError.missingScalesCompanion(name: name)
                }
                guard scales.dtype == .f8e8m0 else {
                    throw RepackError.dtypeMismatch(name: name,
                        detail: "expected F8_E8M0 scale companion, got \(scales.dtype)")
                }
                // One ue8m0 scale per weightBlockRows x weightBlockCols tile.
                let blockRows = UInt64(quant.weightBlockRows)
                let blockCols = UInt64(quant.weightBlockCols)
                guard weight.shape.count == 2, scales.shape.count == 2,
                      scales.shape[0] == ceilDiv(weight.shape[0], blockRows),
                      scales.shape[1] == ceilDiv(weight.shape[1], blockCols) else {
                    throw RepackError.shapeMismatch(name: name,
                        detail: "scale grid \(scales.shape) does not cover weight \(weight.shape) "
                            + "in \(blockRows)x\(blockCols) blocks")
                }
                consumedScales.insert(base + ".scale")

                let wOff = fileCursor
                let wSize = weight.sizeBytes
                let sOff = wOff + wSize
                let sSize = scales.sizeBytes
                fileCursor = sOff + sSize

                entries.append(ResidentEntry(
                    name: name, dtype: ietnyDtype(.f8e4m3),
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: wOff, sizeBytes: wSize,
                    scaleOffset: sOff, scaleSize: sSize,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: .fp8BlockwiseE4M3(blockRows: quant.weightBlockRows,
                                                 blockCols: quant.weightBlockCols,
                                                 scaleFmt: quant.scaleFmt),
                    sourceWeight: weight, sourceScales: scales, sourceBiases: nil))
            } else {
                let off = fileCursor
                let size = weight.sizeBytes
                fileCursor = off + size

                entries.append(ResidentEntry(
                    name: name, dtype: ietnyDtype(weight.dtype),
                    logicalShape4: padTo4(weight.shape),
                    fileOffset: off, sizeBytes: size,
                    scaleOffset: 0, scaleSize: 0,
                    biasOffset: 0, biasSize: 0,
                    quantSpec: nil,
                    sourceWeight: weight, sourceScales: nil, sourceBiases: nil))
            }
        }

        let residentSize = fileCursor - indexSize
        return ResidentFilePlan(path: path,
                                entries: entries,
                                stringTable: stringTable,
                                stringTableOffsets: offsets,
                                indexSize: indexSize,
                                residentSize: residentSize)
    }

    /// V4 layer file: per role (gate=w1, up=w3, down=w2), one FP4 weights
    /// slice (I8 container, two e2m1 per byte) plus one ue8m0 scales slice,
    /// no biases: 6 slices per expert blob, page-aligned fixed stride.
    private static func planV4LayerFile(path: String, layer: Int,
                                        registry: [String: SourceTensor],
                                        arch: ArchInfoV4) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let roles: [(role: String, key: String)] = [
            ("gate", "w1"), ("up", "w3"), ("down", "w2")
        ]
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(6)
        var blobCursor: UInt64 = 0

        let hidden = UInt64(arch.hiddenSize)
        let moe = UInt64(arch.moeIntermediateSize)
        let groupK = UInt64(DeepSeekQuantDescriptor.fp4ExpertScaleGroupK)
        let perByte = UInt64(DeepSeekQuantDescriptor.fp4ElementsPerByte)

        for (role, key) in roles {
            var weights: [SourceTensor] = []
            var scales: [SourceTensor] = []
            weights.reserveCapacity(expertCount)
            scales.reserveCapacity(expertCount)
            for expert in 0..<expertCount {
                let base = "layers.\(layer).ffn.experts.\(expert).\(key)"
                guard let w = registry[base + ".weight"] else {
                    throw RepackError.missingTensor(name: base + ".weight")
                }
                guard let s = registry[base + ".scale"] else {
                    throw RepackError.missingScalesCompanion(name: base + ".weight")
                }
                weights.append(w)
                scales.append(s)
            }

            let w0 = weights[0]
            let s0 = scales[0]
            guard w0.dtype == .i8 else {
                throw RepackError.dtypeMismatch(name: w0.name,
                    detail: "expected I8 FP4 container, got \(w0.dtype)")
            }
            guard s0.dtype == .f8e8m0 else {
                throw RepackError.dtypeMismatch(name: s0.name,
                    detail: "expected F8_E8M0 expert scale, got \(s0.dtype)")
            }
            // FP4 storage: [out, in/2] bytes; one scale per groupK along K.
            guard w0.shape.count == 2, s0.shape.count == 2,
                  s0.shape[0] == w0.shape[0],
                  s0.shape[1] * groupK == w0.shape[1] * perByte else {
                throw RepackError.shapeMismatch(name: w0.name,
                    detail: "FP4 weight \(w0.shape) / scale \(s0.shape) do not match "
                        + "the per-\(groupK) e8m0 layout")
            }
            let expectedWeightShape: [UInt64] = role == "down"
                ? [hidden, moe / perByte]
                : [moe, hidden / perByte]
            guard w0.shape == expectedWeightShape else {
                throw RepackError.shapeMismatch(name: w0.name,
                    detail: "expected \(expectedWeightShape) from arch geometry, got \(w0.shape)")
            }
            // Fixed stride requires uniform per-expert byte sizes.
            for expert in 1..<expertCount {
                let w = weights[expert]
                let s = scales[expert]
                guard w.dtype == w0.dtype, w.shape == w0.shape, w.sizeBytes == w0.sizeBytes,
                      s.dtype == s0.dtype, s.shape == s0.shape, s.sizeBytes == s0.sizeBytes else {
                    throw RepackError.shapeMismatch(name: w.name,
                        detail: "non-uniform expert tensor geometry in layer \(layer) role \(role)")
                }
            }

            blobCursor = align4(blobCursor)
            let logicalWeight: [UInt64] = [w0.shape[0], w0.shape[1] * perByte]
            subs.append(PerExpertTensorSlice(
                role: role, component: "weights", dtype: ietnyDtype(.i8),
                logicalShape: logicalWeight,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: w0.sizeBytes,
                sourceOffsetPerExpert: 0, sourceTensor: w0,
                perExpertSources: weights,
                bitsForWeights: 4))
            blobCursor += w0.sizeBytes

            blobCursor = align4(blobCursor)
            subs.append(PerExpertTensorSlice(
                role: role, component: "scales", dtype: ietnyDtype(.f8e8m0),
                logicalShape: s0.shape,
                offsetInExpertBlob: blobCursor, sizeInExpertBlob: s0.sizeBytes,
                sourceOffsetPerExpert: 0, sourceTensor: s0,
                perExpertSources: scales,
                bitsForWeights: nil))
            blobCursor += s0.sizeBytes
        }

        let expertStride = roundUpToPage(blobCursor)
        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    /// Stable order for the V4 resident tensor list: embedding first, then
    /// per-layer groups in layer order, then the trunk-level tensors.
    private static func v4ResidentOrdering() -> (String, String) -> Bool {
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "embed.weight" { return (0, 0, 0, n) }
            if let li = v4LayerIndex(in: n) {
                return (1, li, v4SlotRank(in: n), n)
            }
            switch n {
            case "norm.weight":   return (2, 0, 0, n)
            case "hc_head_fn":    return (2, 0, 1, n)
            case "hc_head_base":  return (2, 0, 2, n)
            case "hc_head_scale": return (2, 0, 3, n)
            case "head.weight":   return (2, 0, 4, n)
            default:              return (2, 1, 0, n)
            }
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order for V4 tensor names. `.scale` companions are
    /// paired into their weight's entry and never ranked independently.
    private static func v4SlotRank(in n: String) -> Int {
        if n.hasSuffix(".attn.wq_a.weight") { return 0 }
        if n.hasSuffix(".attn.wq_b.weight") { return 1 }
        if n.hasSuffix(".attn.wkv.weight") { return 2 }
        if n.hasSuffix(".attn.wo_a.weight") { return 3 }
        if n.hasSuffix(".attn.wo_b.weight") { return 4 }
        if n.hasSuffix(".attn.q_norm.weight") { return 5 }
        if n.hasSuffix(".attn.kv_norm.weight") { return 6 }
        if n.hasSuffix(".attn.attn_sink") { return 7 }
        if n.hasSuffix(".attn.compressor.wkv.weight") { return 8 }
        if n.hasSuffix(".attn.compressor.wgate.weight") { return 9 }
        if n.hasSuffix(".attn.compressor.norm.weight") { return 10 }
        if n.hasSuffix(".attn.compressor.ape") { return 11 }
        if n.hasSuffix(".attn.indexer.wq_b.weight") { return 12 }
        if n.hasSuffix(".attn.indexer.weights_proj.weight") { return 13 }
        if n.hasSuffix(".attn.indexer.compressor.wkv.weight") { return 14 }
        if n.hasSuffix(".attn.indexer.compressor.wgate.weight") { return 15 }
        if n.hasSuffix(".attn.indexer.compressor.norm.weight") { return 16 }
        if n.hasSuffix(".attn.indexer.compressor.ape") { return 17 }
        if n.hasSuffix(".attn_norm.weight") { return 18 }
        if n.hasSuffix(".ffn.gate.weight") { return 19 }
        if n.hasSuffix(".ffn.gate.bias") { return 20 }
        if n.hasSuffix(".ffn.gate.tid2eid") { return 21 }
        if n.hasSuffix(".ffn.shared_experts.w1.weight") { return 22 }
        if n.hasSuffix(".ffn.shared_experts.w2.weight") { return 23 }
        if n.hasSuffix(".ffn.shared_experts.w3.weight") { return 24 }
        if n.hasSuffix(".ffn_norm.weight") { return 25 }
        if n.hasSuffix(".hc_attn_fn") { return 26 }
        if n.hasSuffix(".hc_attn_base") { return 27 }
        if n.hasSuffix(".hc_attn_scale") { return 28 }
        if n.hasSuffix(".hc_ffn_fn") { return 29 }
        if n.hasSuffix(".hc_ffn_base") { return 30 }
        if n.hasSuffix(".hc_ffn_scale") { return 31 }
        return 100
    }

    // MARK: - Helpers

    private static func ceilDiv(_ a: UInt64, _ b: UInt64) -> UInt64 {
        (a + b - 1) / b
    }

    private static func align4(_ v: UInt64) -> UInt64 {
        (v + 3) & ~UInt64(3)
    }

    private static func ietnyDtype(_ d: SourceTensor.Dtype) -> UInt8 {
        switch d {
        case .u32: 0
        case .bf16: 1
        case .fp16: 2
        case .fp32: 3
        case .i8: 4
        case .i64: 5
        case .f8e4m3: 6
        case .f8e8m0: 7
        }
    }

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = Layout.pageBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Logical shape of a packed quantized tensor whose source is `[D0,..,Dn-1, Dn/factor]`.
    private static func logicalShape(forPackedSource source: [UInt64], bits: Int) -> [UInt64] {
        let factor = UInt64(32 / bits)
        guard !source.isEmpty else { return source }
        var out = source
        out[out.count - 1] = source[source.count - 1] * factor
        return out
    }

    /// Stable order for the resident LM tensor list. Embedding first, then
    /// per-layer groups in layer index order, then the final norm.
    private static func lmResidentOrdering() -> (String, String) -> Bool {
        // Compute a sort key per name; we order by (group rank, layer, slot rank, name).
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "language_model.model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "language_model.model.norm.weight"          { return (3, 0, 0, n) }
            if let li = layerIndex(in: n) {
                let slot = slotRank(in: n)
                return (1, li, slot, n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order. Mirrors the per-layer description in the
    /// architecture doc.
    private static func slotRank(in n: String) -> Int {
        if n.contains(".self_attn.q_proj.weight") { return 0 }
        if n.contains(".self_attn.k_proj.weight") { return 1 }
        if n.contains(".self_attn.v_proj.weight") { return 2 }
        if n.contains(".self_attn.o_proj.weight") { return 3 }
        if n.contains(".self_attn.q_norm.weight") { return 4 }
        if n.contains(".self_attn.k_norm.weight") { return 5 }
        if n.contains(".router.proj.weight")      { return 6 }
        if n.contains(".router.scale")            { return 7 }
        if n.contains(".router.per_expert_scale") { return 8 }
        if n.contains(".mlp.gate_proj.weight")    { return 9 }
        if n.contains(".mlp.up_proj.weight")      { return 10 }
        if n.contains(".mlp.down_proj.weight")    { return 11 }
        if n.hasSuffix(".input_layernorm.weight") { return 12 }
        if n.hasSuffix(".post_attention_layernorm.weight") { return 13 }
        if n.hasSuffix(".pre_feedforward_layernorm.weight") { return 14 }
        if n.hasSuffix(".pre_feedforward_layernorm_2.weight") { return 15 }
        if n.hasSuffix(".post_feedforward_layernorm.weight") { return 16 }
        if n.hasSuffix(".post_feedforward_layernorm_1.weight") { return 17 }
        if n.hasSuffix(".post_feedforward_layernorm_2.weight") { return 18 }
        if n.hasSuffix(".layer_scalar")           { return 19 }
        return 100
    }
}
