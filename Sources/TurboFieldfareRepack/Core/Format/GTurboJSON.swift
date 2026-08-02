import Foundation

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum GTurboJSON {

    static let magic = "GTURBO"
    static let versionMajor = 1
    static let versionMinor = 0
    /// V4-family manifests carry a DeepSeek arch dict and per-scheme quant
    /// slots; minor 1 signals the extended schema. Gemma manifests stay at
    /// minor 0 byte-for-byte.
    static let versionMinorV4 = 1

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let archDict: [String: Any]
        let quantDict: [String: Any]
        let versionMinor: Int
        switch plan.arch {
        case .gemma(let arch):
            versionMinor = GTurboJSON.versionMinor
            archDict = [
                "hiddenSize": arch.hiddenSize,
                "ffnIntermediate": arch.intermediateSize,
                "moeIntermediateSize": arch.moeIntermediateSize,
                "numHeads": arch.numHeads,
                "numKVHeads": arch.numKVHeads,
                "numFullKVHeads": arch.numFullKVHeads,
                "headDim": arch.headDim,
                "fullHeadDim": arch.fullHeadDim,
                "vocabSize": arch.vocabSize,
                "slidingWindow": arch.slidingWindow,
                "finalLogitSoftcap": arch.finalLogitSoftcap,
                "ropeTheta": arch.ropeTheta,
                "fullRopeTheta": arch.fullRopeTheta,
                "partialRotaryFactor": arch.partialRotaryFactor,
                "numLayers": arch.numLayers,
                "numExperts": arch.numExperts,
                "topKExperts": arch.topKExperts,
                "tieWordEmbeddings": arch.tieWordEmbeddings,
                "attentionKEqV": arch.attentionKEqV,
                "hiddenActivation": arch.hiddenActivation,
                "fullAttentionLayerMask": arch.fullAttentionLayerMask.map { Int($0) }
            ]
            let quantBits = [
                "embedding": bitWidths.embedding,
                "attention": bitWidths.attention,
                "router": bitWidths.router,
                "sharedExpert": bitWidths.sharedExpert,
                "routedExpert": bitWidths.routedExpert,
            ]
            var slots: [String: Any] = [:]
            for (slot, bits) in quantBits {
                slots[slot] = [
                    "weightBits": bits,
                    "scheme": plan.baseMode,
                    "scaleType": "BF16",
                    "biasType": "BF16",
                    "groupSize": plan.baseGroupSize
                ]
            }
            quantDict = slots
        case .deepseekV4(let arch):
            versionMinor = GTurboJSON.versionMinorV4
            guard let quant = plan.deepseekQuant else {
                throw RepackError.configurationInvalid(
                    detail: "V4 manifest requires the DeepSeek quant descriptor")
            }
            archDict = v4ArchDict(arch)
            quantDict = v4QuantDict(quant)
        }

        var filesDict: [String: Any] = [:]
        for (path, info) in files {
            filesDict[path] = ["size": info.size, "sha256": info.sha256]
        }

        var manifest: [String: Any] = [
            "magic": GTurboJSON.magic,
            "versionMajor": GTurboJSON.versionMajor,
            "versionMinor": versionMinor,
            "flags": [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false
            ],
            "modelID": modelID,
            "sourceSnapshotHash": sourceSnapshotHash,
            "arch": archDict,
            "quant": quantDict,
            "files": filesDict,
            "expertsPerLayer": expertsPerLayer,
            "numLayers": numLayers,
            "expertStride": expertStride,
            "bitWidthOverridesHonored": plan.bitsOverrideCount
        ]
        if plan.family != .gemma4 {
            manifest["modelFamily"] = plan.family.rawValue
        }
        return try JSONSerialization.data(withJSONObject: manifest,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }

    /// DeepSeek arch dict. Key names mirror the flat `config.json` schema so
    /// the runtime reader can cross-check against the source config directly.
    private static func v4ArchDict(_ arch: ArchInfoV4) -> [String: Any] {
        [
            "num_hidden_layers": arch.numLayers,
            "hidden_size": arch.hiddenSize,
            "vocab_size": arch.vocabSize,
            "n_routed_experts": arch.numExperts,
            "n_shared_experts": arch.numSharedExperts,
            "num_experts_per_tok": arch.topKExperts,
            "moe_intermediate_size": arch.moeIntermediateSize,
            "num_hash_layers": arch.numHashLayers,
            "num_nextn_predict_layers": arch.numMTPLayers,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": arch.numKVHeads,
            "head_dim": arch.headDim,
            "q_lora_rank": arch.qLoraRank,
            "qk_rope_head_dim": arch.qkRopeHeadDim,
            "o_groups": arch.oGroups,
            "o_lora_rank": arch.oLoraRank,
            "index_n_heads": arch.indexNHeads,
            "index_head_dim": arch.indexHeadDim,
            "index_topk": arch.indexTopk,
            "sliding_window": arch.slidingWindow,
            "rope_theta": arch.ropeTheta,
            "compress_rope_theta": arch.compressRopeTheta,
            "compress_ratios": arch.compressRatios,
            "routed_scaling_factor": arch.routedScalingFactor,
            "swiglu_limit": arch.swigluLimit,
            "norm_topk_prob": arch.normTopkProb,
            "scoring_func": arch.scoringFunc,
            "topk_method": arch.topkMethod,
            "hidden_act": arch.hiddenActivation,
            "hc_mult": arch.hcMult,
            "hc_eps": arch.hcEps,
            "hc_sinkhorn_iters": arch.hcSinkhornIters,
            "rms_norm_eps": arch.rmsNormEps,
            "tie_word_embeddings": arch.tieWordEmbeddings,
            "max_position_embeddings": arch.maxPositionEmbeddings,
            "yarn_factor": arch.yarnFactor,
            "yarn_original_max_position_embeddings": arch.yarnOriginalMaxPositions,
            "yarn_beta_fast": arch.yarnBetaFast,
            "yarn_beta_slow": arch.yarnBetaSlow,
        ]
    }

    /// V4 quant slots. FP8 slots carry the 2-D block geometry; the FP4 slot
    /// carries the along-K scale group; unquantized slots (embedding, router)
    /// are BF16 on disk with no companions.
    private static func v4QuantDict(_ quant: DeepSeekQuantDescriptor) -> [String: Any] {
        let fp8Slot: [String: Any] = [
            "weightBits": 8,
            "scheme": DeepSeekQuantDescriptor.fp8SchemeName,
            "scaleType": "F8_E8M0",
            "biasType": "none",
            "groupSize": 0,
            "blockRows": quant.weightBlockRows,
            "blockCols": quant.weightBlockCols,
        ]
        let bf16Slot: [String: Any] = [
            "weightBits": 16,
            "scheme": "none",
            "scaleType": "none",
            "biasType": "none",
            "groupSize": 0,
        ]
        return [
            "embedding": bf16Slot,
            "attention": fp8Slot,
            "router": bf16Slot,
            "sharedExpert": fp8Slot,
            "routedExpert": [
                "weightBits": 4,
                "scheme": DeepSeekQuantDescriptor.fp4SchemeName,
                "scaleType": "F8_E8M0",
                "biasType": "none",
                "groupSize": quant.expertScaleGroupK,
            ],
        ]
    }

    /// Index/layout dtype byte -> safetensors-style name.
    static func dtypeName(_ code: UInt8) -> String {
        switch code {
        case 0: return "U32"
        case 1: return "BF16"
        case 2: return "F16"
        case 3: return "F32"
        case 4: return "I8"
        case 5: return "I64"
        case 6: return "F8_E4M3"
        case 7: return "F8_E8M0"
        default: return "UNKNOWN(\(code))"
        }
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        let arch = plan.arch
        var layersArr: [[String: Any]] = []
        layersArr.reserveCapacity(plan.layers.count)
        for lp in plan.layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [[String: Any]] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let base = UInt64(e) * lp.expertStride
                var tensors: [String: Any] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    var t: [String: Any] = [
                        "offset": slice.offsetInExpertBlob,
                        "size":   slice.sizeInExpertBlob,
                        "dtype":  dtypeName(slice.dtype),
                        "shape":  slice.logicalShape.map { Int($0) }
                    ]
                    if let bits = slice.bitsForWeights { t["bits"] = bits }
                    tensors[key] = t
                }
                let expertEntry: [String: Any] = [
                    "expert": e,
                    "offset": base,
                    "size":   lp.expertStride,
                    "tensors": tensors
                ]
                experts.append(expertEntry)
            }
            layersArr.append([
                "layer": lp.layerIndex,
                "file":  layerFile,
                "experts": experts
            ])
        }
        let obj: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": arch.numLayers,
            "expertsPerLayer": plan.layers.first?.expertsPerLayer ?? 0,
            "layers": layersArr
        ]
        return try JSONSerialization.data(withJSONObject: obj,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
