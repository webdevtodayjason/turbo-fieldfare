import Foundation

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = [
        "streamingPresent", "turboQuantKV", "aneSharedExpert"
    ]

    /// Required file entries (relative to `model.gturbo/`). Layer files
    /// `packed_experts/layer_<L>.bin` for L in 0..<numLayers are checked
    /// after decode against `numLayers` (with the zero-padded "layer_%02d"
    /// naming the writer produces; falling back to plain "layer_<L>" when
    /// only the unpadded form is present, for toy synthetics).
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: Manifest
        do {
            manifest = try JSONDecoder().decode(Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting,
                     directoryURL: directoryURL)
        return manifest
    }

    private static func metadataFileSize(_ url: URL,
                                         fileName: String) throws -> UInt64 {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        guard let number = attrs[.size] as? NSNumber else {
            throw ModelError.indexCorrupt(detail: "\(fileName): file size unavailable")
        }
        return number.uint64Value
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig,
                         directoryURL: URL) throws {
        guard m.magic == "GTURBO" else { throw ModelError.notAGTurboDirectory }
        guard m.versionMajor == 1 else {
            throw ModelError.unsupportedVersion(major: m.versionMajor, minor: m.versionMinor)
        }
        for key in m.flags.keys {
            if !knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
        }
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant)
        } else if expected.numLayers == ArchConfig.gemma4_26B_A4B.numLayers,
                  expected.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        let pageSize = UInt64(getpagesize())
        guard m.expertStride % pageSize == 0 else {
            throw ModelError.expertStrideNotPageAligned(stride: m.expertStride,
                                                        pageSize: Int(pageSize))
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
        for L in 0..<m.numLayers {
            let padded = String(format: "packed_experts/layer_%02d.bin", L)
            let plain  = "packed_experts/layer_\(L).bin"
            if m.files[padded] == nil && m.files[plain] == nil {
                throw ModelError.missingFile(name: padded)
            }
        }
    }

    private static func validateQuant(_ quant: ManifestQuant) throws {
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, [4]),
            ("attention", quant.attention, [4]),
            ("router", quant.router, [8]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
            ("routedExpert", quant.routedExpert, [4]),
        ]
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == Quantization.groupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)
    }
}

// MARK: - DeepSeek V4 family (modelFamily == "deepseek-v4-flash")
//
// The V4 repacker (V4F-01) writes a DeepSeek-keyed arch dict, per-scheme
// quant slots, `modelFamily: "deepseek-v4-flash"`, and versionMinor 1. The
// Gemma decode path above is untouched: `ManifestReader.load` neither emits
// nor accepts this schema, and a V4 manifest fails the Gemma `Decodable`
// exactly as before. V4 installs load through `loadV4`.

/// DeepSeek-keyed arch dict (mirrors the flat `config.json` schema).
public struct ManifestArchV4: Decodable, Equatable, Sendable {
    public let numLayers: Int
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numExperts: Int
    public let numSharedExperts: Int
    public let topKExperts: Int
    public let moeIntermediateSize: Int
    public let numHashLayers: Int
    public let numMTPLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let qLoraRank: Int
    public let qkRopeHeadDim: Int
    public let oGroups: Int
    public let oLoraRank: Int
    public let indexNHeads: Int
    public let indexHeadDim: Int
    public let indexTopk: Int
    public let slidingWindow: Int
    public let ropeTheta: Double
    public let compressRopeTheta: Double
    public let compressRatios: [Int]
    public let routedScalingFactor: Double
    public let swigluLimit: Double
    public let normTopkProb: Bool
    public let scoringFunc: String
    public let topkMethod: String
    public let hiddenActivation: String
    public let hcMult: Int
    public let hcEps: Double
    public let hcSinkhornIters: Int
    public let rmsNormEps: Double
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int
    public let yarnFactor: Double
    public let yarnOriginalMaxPositions: Int
    public let yarnBetaFast: Double
    public let yarnBetaSlow: Double

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case vocabSize = "vocab_size"
        case numExperts = "n_routed_experts"
        case numSharedExperts = "n_shared_experts"
        case topKExperts = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHashLayers = "num_hash_layers"
        case numMTPLayers = "num_nextn_predict_layers"
        case numHeads = "num_attention_heads"
        case numKVHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case compressRopeTheta = "compress_rope_theta"
        case compressRatios = "compress_ratios"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluLimit = "swiglu_limit"
        case normTopkProb = "norm_topk_prob"
        case scoringFunc = "scoring_func"
        case topkMethod = "topk_method"
        case hiddenActivation = "hidden_act"
        case hcMult = "hc_mult"
        case hcEps = "hc_eps"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case yarnFactor = "yarn_factor"
        case yarnOriginalMaxPositions = "yarn_original_max_position_embeddings"
        case yarnBetaFast = "yarn_beta_fast"
        case yarnBetaSlow = "yarn_beta_slow"
    }
}

/// V4 quant slot. FP8 slots carry the 2-D block geometry; the FP4 slot
/// carries the along-K scale group; BF16 slots are unquantized.
public struct ManifestQuantSlotV4: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
    public let blockRows: Int?
    public let blockCols: Int?
}

public struct ManifestQuantV4: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlotV4
    public let attention: ManifestQuantSlotV4
    public let router: ManifestQuantSlotV4
    public let sharedExpert: ManifestQuantSlotV4
    public let routedExpert: ManifestQuantSlotV4
}

public struct V4Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelFamily: String
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArchV4
    public let quant: ManifestQuantV4?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

extension ManifestReader {

    /// Family string written by the V4 repacker (`ModelFamily.deepseekV4Flash`
    /// in the repack target).
    public static let deepSeekV4FlashFamily = "deepseek-v4-flash"

    /// Read only the family discriminator from `manifest.json`. Gemma-family
    /// manifests carry no `modelFamily` key and probe as `nil`.
    public static func probeModelFamily(directoryURL: URL,
                                        maxBytes: UInt64 = defaultMaxBytes) throws -> String? {
        struct Probe: Decodable {
            let modelFamily: String?
        }
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        do {
            return try JSONDecoder().decode(Probe.self, from: data).modelFamily
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
    }

    /// Load and validate a DeepSeek V4-family install against the expected
    /// architecture. Field-for-field gate, mirroring the Gemma path.
    public static func loadV4(directoryURL: URL,
                              expecting: V4ArchConfig,
                              maxBytes: UInt64 = defaultMaxBytes) throws -> V4Manifest {
        let manifestURL = directoryURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let size = try metadataFileSize(manifestURL, fileName: "manifest.json")
        guard size <= maxBytes else {
            throw ModelError.indexCorrupt(
                detail: "manifest.json size \(size) exceeds metadata cap \(maxBytes)")
        }
        let data = try Data(contentsOf: manifestURL)
        let manifest: V4Manifest
        do {
            manifest = try JSONDecoder().decode(V4Manifest.self, from: data)
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }
        try validateV4(manifest, against: expecting)
        return manifest
    }

    static func validateV4(_ m: V4Manifest, against e: V4ArchConfig) throws {
        guard m.magic == "GTURBO" else { throw ModelError.notAGTurboDirectory }
        guard m.versionMajor == 1, m.versionMinor >= 1 else {
            throw ModelError.unsupportedVersion(major: m.versionMajor, minor: m.versionMinor)
        }
        guard m.modelFamily == deepSeekV4FlashFamily else {
            throw ModelError.unsupportedModelFamily(found: m.modelFamily)
        }
        for key in m.flags.keys {
            if !knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
        }
        if m.flags["turboQuantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed TurboQuant KV runtime support")
        }
        try validateArchV4(m.arch, expected: e)
        if let quant = m.quant {
            try validateQuantV4(quant)
        } else {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the V4 architecture")
        }
        let pageSize = UInt64(getpagesize())
        guard m.expertStride % pageSize == 0 else {
            throw ModelError.expertStrideNotPageAligned(stride: m.expertStride,
                                                        pageSize: Int(pageSize))
        }
        guard m.numLayers == e.numLayers,
              m.expertsPerLayer == e.numExperts else {
            throw ModelError.archMismatch(field: "numLayers/expertsPerLayer",
                                          expected: "\(e.numLayers)/\(e.numExperts)",
                                          actual: "\(m.numLayers)/\(m.expertsPerLayer)")
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
        for L in 0..<m.numLayers {
            let padded = String(format: "packed_experts/layer_%02d.bin", L)
            let plain  = "packed_experts/layer_\(L).bin"
            if m.files[padded] == nil && m.files[plain] == nil {
                throw ModelError.missingFile(name: padded)
            }
        }
    }

    private static func validateQuantV4(_ quant: ManifestQuantV4) throws {
        func checkBF16(_ name: String, _ slot: ManifestQuantSlotV4) throws {
            guard slot.weightBits == 16,
                  slot.scheme.lowercased() == "none",
                  slot.scaleType.lowercased() == "none",
                  slot.biasType.lowercased() == "none" else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
        func checkFP8(_ name: String, _ slot: ManifestQuantSlotV4) throws {
            guard slot.weightBits == 8,
                  slot.scheme.lowercased() == "fp8-blockwise-e4m3",
                  slot.scaleType.lowercased() == "f8_e8m0",
                  slot.biasType.lowercased() == "none",
                  slot.blockRows == V4Quantization.fp8BlockSize,
                  slot.blockCols == V4Quantization.fp8BlockSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
        try checkBF16("embedding", quant.embedding)
        try checkFP8("attention", quant.attention)
        try checkBF16("router", quant.router)
        try checkFP8("sharedExpert", quant.sharedExpert)
        let routed = quant.routedExpert
        guard routed.weightBits == 4,
              routed.scheme.lowercased() == "fp4-e2m1-e8m0",
              routed.scaleType.lowercased() == "f8_e8m0",
              routed.biasType.lowercased() == "none",
              routed.groupSize == V4Quantization.fp4GroupSize else {
            throw ModelError.indexCorrupt(detail: "unsupported quantization for routedExpert")
        }
    }

    private static func validateArchV4(_ a: ManifestArchV4,
                                       expected e: V4ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("num_hidden_layers", a.numLayers, e.numLayers)
        try check("hidden_size", a.hiddenSize, e.hiddenSize)
        try check("vocab_size", a.vocabSize, e.vocabSize)
        try check("n_routed_experts", a.numExperts, e.numExperts)
        try check("n_shared_experts", a.numSharedExperts, e.numSharedExperts)
        try check("num_experts_per_tok", a.topKExperts, e.topKExperts)
        try check("moe_intermediate_size", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("num_hash_layers", a.numHashLayers, e.numHashLayers)
        try check("num_nextn_predict_layers", a.numMTPLayers, e.numMTPLayers)
        try check("num_attention_heads", a.numHeads, e.numHeads)
        try check("num_key_value_heads", a.numKVHeads, e.numKVHeads)
        try check("head_dim", a.headDim, e.headDim)
        try check("q_lora_rank", a.qLoraRank, e.qLoraRank)
        try check("qk_rope_head_dim", a.qkRopeHeadDim, e.qkRopeHeadDim)
        try check("o_groups", a.oGroups, e.oGroups)
        try check("o_lora_rank", a.oLoraRank, e.oLoraRank)
        try check("index_n_heads", a.indexNHeads, e.indexNHeads)
        try check("index_head_dim", a.indexHeadDim, e.indexHeadDim)
        try check("index_topk", a.indexTopk, e.indexTopk)
        try check("sliding_window", a.slidingWindow, e.slidingWindow)
        try check("rope_theta", a.ropeTheta, e.ropeTheta)
        try check("compress_rope_theta", a.compressRopeTheta, e.compressRopeTheta)
        try check("compress_ratios", a.compressRatios.description, e.compressRatios.description)
        try check("routed_scaling_factor", a.routedScalingFactor, e.routedScalingFactor)
        try check("swiglu_limit", a.swigluLimit, e.swigluLimit)
        try check("norm_topk_prob", a.normTopkProb, e.normTopkProb)
        try check("scoring_func", a.scoringFunc, e.scoringFunc)
        try check("topk_method", a.topkMethod, e.topkMethod)
        try check("hidden_act", a.hiddenActivation, e.hiddenActivation)
        try check("hc_mult", a.hcMult, e.hcMult)
        try check("hc_eps", a.hcEps, e.hcEps)
        try check("hc_sinkhorn_iters", a.hcSinkhornIters, e.hcSinkhornIters)
        try check("rms_norm_eps", a.rmsNormEps, e.rmsNormEps)
        try check("tie_word_embeddings", a.tieWordEmbeddings, e.tieWordEmbeddings)
        try check("max_position_embeddings", a.maxPositionEmbeddings, e.maxPositionEmbeddings)
        try check("yarn_factor", a.yarnFactor, e.yarnFactor)
        try check("yarn_original_max_position_embeddings",
                  a.yarnOriginalMaxPositions, e.yarnOriginalMaxPositions)
        try check("yarn_beta_fast", a.yarnBetaFast, e.yarnBetaFast)
        try check("yarn_beta_slow", a.yarnBetaSlow, e.yarnBetaSlow)
    }
}
