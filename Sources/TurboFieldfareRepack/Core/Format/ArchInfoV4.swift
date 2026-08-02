import Foundation

/// Architecture facts for DeepSeek V4-Flash, read from the flat
/// `config.json` schema (no `text_config`). Mirrored into
/// `manifest.json -> arch` for V4-family installs; the runtime loader
/// cross-checks the fields it understands (V4F-03/04 wire the rest).
struct ArchInfoV4: Sendable, Equatable {
    let numLayers: Int                 // num_hidden_layers = 43
    let hiddenSize: Int                // hidden_size = 4096
    let vocabSize: Int                 // vocab_size = 129280
    let numExperts: Int                // n_routed_experts = 256
    let numSharedExperts: Int          // n_shared_experts = 1
    let topKExperts: Int               // num_experts_per_tok = 6
    let moeIntermediateSize: Int       // moe_intermediate_size = 2048
    let numHashLayers: Int             // num_hash_layers = 3 (layers 0-2 hash-routed)
    let numMTPLayers: Int              // num_nextn_predict_layers = 1 (dropped at repack)
    let numHeads: Int                  // num_attention_heads = 64
    let numKVHeads: Int                // num_key_value_heads = 1 (MQA, shared K==V)
    let headDim: Int                   // head_dim = 512
    let qLoraRank: Int                 // q_lora_rank = 1024
    let qkRopeHeadDim: Int             // qk_rope_head_dim = 64 (partial RoPE width)
    let oGroups: Int                   // o_groups = 8
    let oLoraRank: Int                 // o_lora_rank = 1024
    let indexNHeads: Int               // index_n_heads = 64
    let indexHeadDim: Int              // index_head_dim = 128
    let indexTopk: Int                 // index_topk = 512
    let slidingWindow: Int             // sliding_window = 128
    let ropeTheta: Double              // rope_theta = 10000 (ratio-0 layers)
    let compressRopeTheta: Double      // compress_rope_theta = 160000
    /// Per-layer compress ratios (0 = pure sliding window, 4 = CSA, 128 = HCA).
    let compressRatios: [Int]
    let routedScalingFactor: Double    // routed_scaling_factor = 1.5
    let swigluLimit: Double            // swiglu_limit = 10.0
    let normTopkProb: Bool             // norm_topk_prob = true
    let scoringFunc: String            // scoring_func = "sqrtsoftplus"
    let topkMethod: String             // topk_method = "noaux_tc"
    let hiddenActivation: String       // hidden_act = "silu"
    let hcMult: Int                    // hc_mult = 4 (mHC stream count)
    let hcEps: Double                  // hc_eps = 1e-6
    let hcSinkhornIters: Int           // hc_sinkhorn_iters = 20
    let rmsNormEps: Double             // rms_norm_eps = 1e-6
    let tieWordEmbeddings: Bool        // tie_word_embeddings = false
    let maxPositionEmbeddings: Int     // max_position_embeddings = 1048576
    // YaRN scaling (rope_scaling): factor 16 over 64k original context.
    let yarnFactor: Double
    let yarnOriginalMaxPositions: Int
    let yarnBetaFast: Double
    let yarnBetaSlow: Double

    static func load(configPath: String) throws -> ArchInfoV4 {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        func i(_ k: String) throws -> Int {
            guard let n = (root[k] as? Int) ?? (root[k] as? NSNumber)?.intValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func d(_ k: String) throws -> Double {
            guard let n = (root[k] as? Double) ?? (root[k] as? NSNumber)?.doubleValue else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func s(_ k: String) throws -> String {
            guard let v = root[k] as? String else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return v
        }
        guard let ratios = root["compress_ratios"] as? [Int] ??
                (root["compress_ratios"] as? [NSNumber])?.map(\.intValue) else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "missing compress_ratios")
        }
        let numLayers = try i("num_hidden_layers")
        let numMTPLayers = (root["num_nextn_predict_layers"] as? Int)
            ?? (root["num_nextn_predict_layers"] as? NSNumber)?.intValue ?? 0
        // The published config carries one compress_ratio per transformer
        // layer PLUS one per MTP module (trailing 0 on V4-Flash). MTP
        // tensors are dropped at repack, so keep only the main-layer
        // entries; reject any other count.
        guard ratios.count == numLayers || ratios.count == numLayers + numMTPLayers else {
            throw RepackError.configJsonInvalid(
                path: configPath,
                detail: "compress_ratios count \(ratios.count) != num_hidden_layers \(numLayers) (+ num_nextn_predict_layers \(numMTPLayers))")
        }
        let layerRatios = Array(ratios.prefix(numLayers))
        let rope = (root["rope_scaling"] as? [String: Any]) ?? [:]
        func yarnD(_ k: String, _ fallback: Double) -> Double {
            (rope[k] as? Double) ?? (rope[k] as? NSNumber)?.doubleValue ?? fallback
        }
        func yarnI(_ k: String, _ fallback: Int) -> Int {
            (rope[k] as? Int) ?? (rope[k] as? NSNumber)?.intValue ?? fallback
        }
        return ArchInfoV4(
            numLayers: numLayers,
            hiddenSize: try i("hidden_size"),
            vocabSize: try i("vocab_size"),
            numExperts: try i("n_routed_experts"),
            numSharedExperts: try i("n_shared_experts"),
            topKExperts: try i("num_experts_per_tok"),
            moeIntermediateSize: try i("moe_intermediate_size"),
            numHashLayers: try i("num_hash_layers"),
            numMTPLayers: numMTPLayers,
            numHeads: try i("num_attention_heads"),
            numKVHeads: try i("num_key_value_heads"),
            headDim: try i("head_dim"),
            qLoraRank: try i("q_lora_rank"),
            qkRopeHeadDim: try i("qk_rope_head_dim"),
            oGroups: try i("o_groups"),
            oLoraRank: try i("o_lora_rank"),
            indexNHeads: try i("index_n_heads"),
            indexHeadDim: try i("index_head_dim"),
            indexTopk: try i("index_topk"),
            slidingWindow: try i("sliding_window"),
            ropeTheta: try d("rope_theta"),
            compressRopeTheta: try d("compress_rope_theta"),
            compressRatios: layerRatios,
            routedScalingFactor: try d("routed_scaling_factor"),
            swigluLimit: try d("swiglu_limit"),
            normTopkProb: (root["norm_topk_prob"] as? Bool) ?? true,
            scoringFunc: try s("scoring_func"),
            topkMethod: try s("topk_method"),
            hiddenActivation: (root["hidden_act"] as? String) ?? "silu",
            hcMult: try i("hc_mult"),
            hcEps: try d("hc_eps"),
            hcSinkhornIters: try i("hc_sinkhorn_iters"),
            rmsNormEps: try d("rms_norm_eps"),
            tieWordEmbeddings: (root["tie_word_embeddings"] as? Bool) ?? false,
            maxPositionEmbeddings: try i("max_position_embeddings"),
            yarnFactor: yarnD("factor", 16),
            yarnOriginalMaxPositions: yarnI("original_max_position_embeddings", 65536),
            yarnBetaFast: yarnD("beta_fast", 32),
            yarnBetaSlow: yarnD("beta_slow", 1))
    }
}

/// Family-tagged architecture context handed to the planner.
enum PlanArch: Sendable, Equatable {
    case gemma(ArchInfo)
    case deepseekV4(ArchInfoV4)

    var family: ModelFamily {
        switch self {
        case .gemma: return .gemma4
        case .deepseekV4: return .deepseekV4Flash
        }
    }

    var numLayers: Int {
        switch self {
        case .gemma(let a): return a.numLayers
        case .deepseekV4(let a): return a.numLayers
        }
    }

    var numExperts: Int {
        switch self {
        case .gemma(let a): return a.numExperts
        case .deepseekV4(let a): return a.numExperts
        }
    }

    /// Loads the arch info matching the metadata family from a config.json.
    static func load(configPath: String, family: ModelFamily) throws -> PlanArch {
        switch family {
        case .gemma4:
            return .gemma(try ArchInfo.load(configPath: configPath))
        case .deepseekV4Flash:
            return .deepseekV4(try ArchInfoV4.load(configPath: configPath))
        }
    }
}
