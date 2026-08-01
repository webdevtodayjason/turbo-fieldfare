import Foundation

/// Synthesises a tiny DeepSeek-V4-family safetensors snapshot (flat config,
/// `quantization_config` FP8 blockwise + FP4 per-expert tensors) inside a
/// temporary directory. Mirrors `SyntheticSnapshot` for the V4 planner fork:
/// two shards so per-expert sources provably resolve across shard files, a
/// hash-routed layer 0 (tid2eid, no router bias), a CSA layer 1 (compressor +
/// indexer, router bias), and an MTP module that must be dropped.
enum SyntheticV4Snapshot {

    struct Snapshot {
        let shardPaths: [String]
        /// name -> (shard filename, payload bytes) for digest round-trips.
        let tensors: [String: (shard: String, bytes: [UInt8])]
        /// Names under `mtp.*` that the repacker must drop.
        let mtpTensorNames: [String]
        /// Resident tensor names expected in `model_weights.bin` order group
        /// (excludes `.scale` companions and expert tensors).
        let residentNames: [String]
    }

    struct Arch {
        let hidden: Int = 128
        let moeIntermediate: Int = 64
        let vocab: Int = 256
        let numLayers: Int = 2
        let numExperts: Int = 4
        let topK: Int = 2
        let numHeads: Int = 2
        let headDim: Int = 64
        let qLoraRank: Int = 32
        let qkRopeHeadDim: Int = 16
        let oGroups: Int = 2
        let oLoraRank: Int = 32
        let indexNHeads: Int = 2
        let indexHeadDim: Int = 16
        let indexTopk: Int = 8
        let slidingWindow: Int = 16
        let numHashLayers: Int = 1
        let numMTPLayers: Int = 1
        // layer 0 = pure sliding window (hash-routed), layer 1 = CSA.
        let compressRatios: [Int] = [0, 4]
    }

    static func build(at dir: String,
                      seed: UInt64 = 0xD33F_5EED_F4A9_DEAD) throws -> Snapshot {
        try? FileManager.default.removeItem(atPath: dir)
        try FileManager.default.createDirectory(atPath: dir,
                                                withIntermediateDirectories: true)

        let arch = Arch()
        var rng = SplitMix64(seed: seed)
        var tensors: [(name: String, dtype: String, shape: [Int], bytes: [UInt8])] = []
        var residentNames: [String] = []
        var mtpNames: [String] = []

        func add(_ name: String, _ dtype: String, _ shape: [Int], resident: Bool) {
            let elemBytes: Int
            switch dtype {
            case "F32", "U32": elemBytes = 4
            case "BF16", "F16": elemBytes = 2
            case "I64": elemBytes = 8
            default: elemBytes = 1  // I8, F8_E4M3, F8_E8M0
            }
            let count = shape.reduce(1, *) * elemBytes
            var bytes = [UInt8](repeating: 0, count: count)
            for i in 0..<count { bytes[i] = UInt8(rng.next() & 0xFF) }
            tensors.append((name, dtype, shape, bytes))
            if resident { residentNames.append(name) }
        }

        /// FP8 e4m3 weight + ue8m0 128x128 block-scale companion.
        func addFP8(_ base: String, _ shape: [Int]) {
            add(base + ".weight", "F8_E4M3", shape, resident: true)
            let grid = [max(1, (shape[0] + 127) / 128), max(1, (shape[1] + 127) / 128)]
            add(base + ".scale", "F8_E8M0", grid, resident: false)
        }

        /// One routed-expert role: I8 FP4 container + ue8m0 per-32 scales.
        func addExpert(layer: Int, expert: Int, role: String, out: Int, logicalIn: Int) {
            let base = "layers.\(layer).ffn.experts.\(expert).\(role)"
            add(base + ".weight", "I8", [out, logicalIn / 2], resident: false)
            add(base + ".scale", "F8_E8M0", [out, logicalIn / 32], resident: false)
        }

        // -- Trunk-level resident tensors.
        add("embed.weight", "BF16", [arch.vocab, arch.hidden], resident: true)

        for layer in 0..<arch.numLayers {
            let p = "layers.\(layer)"
            let ratio = arch.compressRatios[layer]
            let hashRouted = layer < arch.numHashLayers

            addFP8("\(p).attn.wq_a", [arch.qLoraRank, arch.hidden])
            // Deliberately > 128 rows so the block-scale grid is non-trivial.
            addFP8("\(p).attn.wq_b", [arch.numHeads * arch.headDim + 128, arch.qLoraRank])
            addFP8("\(p).attn.wkv", [arch.headDim, arch.hidden])
            addFP8("\(p).attn.wo_a", [arch.oGroups * arch.oLoraRank,
                                      arch.numHeads * arch.headDim / arch.oGroups])
            addFP8("\(p).attn.wo_b", [arch.hidden, arch.oGroups * arch.oLoraRank])
            add("\(p).attn.q_norm.weight", "BF16", [arch.qLoraRank], resident: true)
            add("\(p).attn.kv_norm.weight", "BF16", [arch.headDim], resident: true)
            add("\(p).attn.attn_sink", "F32", [arch.numHeads], resident: true)
            if ratio != 0 {
                add("\(p).attn.compressor.wkv.weight", "BF16",
                    [2 * arch.headDim, arch.hidden], resident: true)
                add("\(p).attn.compressor.wgate.weight", "BF16",
                    [2 * arch.headDim, arch.hidden], resident: true)
                add("\(p).attn.compressor.norm.weight", "BF16", [arch.headDim], resident: true)
                add("\(p).attn.compressor.ape", "F32", [ratio, 2 * arch.headDim], resident: true)
            }
            if ratio == 4 {
                addFP8("\(p).attn.indexer.wq_b",
                       [arch.indexNHeads * arch.indexHeadDim, arch.qLoraRank])
                add("\(p).attn.indexer.weights_proj.weight", "BF16",
                    [arch.indexNHeads, arch.hidden], resident: true)
                add("\(p).attn.indexer.compressor.wkv.weight", "BF16",
                    [2 * arch.indexHeadDim, arch.hidden], resident: true)
                add("\(p).attn.indexer.compressor.wgate.weight", "BF16",
                    [2 * arch.indexHeadDim, arch.hidden], resident: true)
                add("\(p).attn.indexer.compressor.norm.weight", "BF16",
                    [arch.indexHeadDim], resident: true)
                add("\(p).attn.indexer.compressor.ape", "F32",
                    [ratio, 2 * arch.indexHeadDim], resident: true)
            }
            add("\(p).attn_norm.weight", "BF16", [arch.hidden], resident: true)

            add("\(p).ffn.gate.weight", "BF16", [arch.numExperts, arch.hidden], resident: true)
            if hashRouted {
                add("\(p).ffn.gate.tid2eid", "I64", [arch.vocab, arch.topK], resident: true)
            } else {
                add("\(p).ffn.gate.bias", "F32", [arch.numExperts], resident: true)
            }
            addFP8("\(p).ffn.shared_experts.w1", [arch.moeIntermediate, arch.hidden])
            addFP8("\(p).ffn.shared_experts.w2", [arch.hidden, arch.moeIntermediate])
            addFP8("\(p).ffn.shared_experts.w3", [arch.moeIntermediate, arch.hidden])
            add("\(p).ffn_norm.weight", "BF16", [arch.hidden], resident: true)

            add("\(p).hc_attn_fn", "F32", [24, 4 * arch.hidden], resident: true)
            add("\(p).hc_attn_base", "F32", [24], resident: true)
            add("\(p).hc_attn_scale", "F32", [3], resident: true)
            add("\(p).hc_ffn_fn", "F32", [24, 4 * arch.hidden], resident: true)
            add("\(p).hc_ffn_base", "F32", [24], resident: true)
            add("\(p).hc_ffn_scale", "F32", [3], resident: true)

            for expert in 0..<arch.numExperts {
                addExpert(layer: layer, expert: expert, role: "w1",
                          out: arch.moeIntermediate, logicalIn: arch.hidden)
                addExpert(layer: layer, expert: expert, role: "w2",
                          out: arch.hidden, logicalIn: arch.moeIntermediate)
                addExpert(layer: layer, expert: expert, role: "w3",
                          out: arch.moeIntermediate, logicalIn: arch.hidden)
            }
        }

        add("norm.weight", "BF16", [arch.hidden], resident: true)
        add("hc_head_fn", "F32", [4, 4 * arch.hidden], resident: true)
        add("hc_head_base", "F32", [4], resident: true)
        add("hc_head_scale", "F32", [1], resident: true)
        add("head.weight", "BF16", [arch.vocab, arch.hidden], resident: true)

        // -- MTP module: dropped wholesale, recorded in the audit. Includes an
        // expert-shaped name to prove MTP experts never reach a layer file.
        var mtpTensors: [(String, String, [Int])] = [
            ("mtp.0.e_proj.weight", "F8_E4M3", [arch.hidden, arch.hidden]),
            ("mtp.0.e_proj.scale", "F8_E8M0", [1, 1]),
            ("mtp.0.h_proj.weight", "F8_E4M3", [arch.hidden, arch.hidden]),
            ("mtp.0.h_proj.scale", "F8_E8M0", [1, 1]),
            ("mtp.0.enorm.weight", "BF16", [arch.hidden]),
            ("mtp.0.hnorm.weight", "BF16", [arch.hidden]),
            ("mtp.0.norm.weight", "BF16", [arch.hidden]),
            ("mtp.0.attn.wq_a.weight", "F8_E4M3", [arch.qLoraRank, arch.hidden]),
            ("mtp.0.attn.wq_a.scale", "F8_E8M0", [1, 1]),
            ("mtp.0.ffn.experts.0.w1.weight", "I8", [arch.moeIntermediate, arch.hidden / 2]),
            ("mtp.0.ffn.experts.0.w1.scale", "F8_E8M0", [arch.moeIntermediate, arch.hidden / 32]),
            ("mtp.0.ffn.gate.weight", "BF16", [arch.numExperts, arch.hidden]),
            ("mtp.0.hc_head_fn", "F32", [4, 4 * arch.hidden]),
        ]
        for (name, dtype, shape) in mtpTensors {
            add(name, dtype, shape, resident: false)
            mtpNames.append(name)
        }
        mtpNames.sort()

        // -- Encode two shards: layer-0 experts + everything else in shard 1,
        // layer-1 experts in shard 2, so per-expert sources cross shards.
        let shard1Name = "model-00001-of-00002.safetensors"
        let shard2Name = "model-00002-of-00002.safetensors"
        var shard1: [(String, String, [Int], [UInt8])] = []
        var shard2: [(String, String, [Int], [UInt8])] = []
        var inventory: [String: (shard: String, bytes: [UInt8])] = [:]
        for t in tensors {
            let isLayer1Expert = t.name.hasPrefix("layers.1.ffn.experts.")
            if isLayer1Expert {
                shard2.append((t.name, t.dtype, t.shape, t.bytes))
                inventory[t.name] = (shard2Name, t.bytes)
            } else {
                shard1.append((t.name, t.dtype, t.shape, t.bytes))
                inventory[t.name] = (shard1Name, t.bytes)
            }
        }
        let shard1Path = (dir as NSString).appendingPathComponent(shard1Name)
        let shard2Path = (dir as NSString).appendingPathComponent(shard2Name)
        try writeV4Shard(path: shard1Path, tensors: shard1)
        try writeV4Shard(path: shard2Path, tensors: shard2)

        // -- config.json: flat DeepSeek schema.
        let config: [String: Any] = [
            "architectures": ["DeepseekV4ForCausalLM"],
            "model_type": "deepseek_v4",
            "expert_dtype": "fp4",
            "quantization_config": [
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "ue8m0",
                "weight_block_size": [128, 128],
            ],
            "num_hidden_layers": arch.numLayers,
            "hidden_size": arch.hidden,
            "vocab_size": arch.vocab,
            "n_routed_experts": arch.numExperts,
            "n_shared_experts": 1,
            "num_experts_per_tok": arch.topK,
            "moe_intermediate_size": arch.moeIntermediate,
            "num_hash_layers": arch.numHashLayers,
            "num_nextn_predict_layers": arch.numMTPLayers,
            "num_attention_heads": arch.numHeads,
            "num_key_value_heads": 1,
            "head_dim": arch.headDim,
            "q_lora_rank": arch.qLoraRank,
            "qk_rope_head_dim": arch.qkRopeHeadDim,
            "o_groups": arch.oGroups,
            "o_lora_rank": arch.oLoraRank,
            "index_n_heads": arch.indexNHeads,
            "index_head_dim": arch.indexHeadDim,
            "index_topk": arch.indexTopk,
            "sliding_window": arch.slidingWindow,
            "rope_theta": 10000.0,
            "compress_rope_theta": 160000.0,
            "compress_ratios": arch.compressRatios,
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "norm_topk_prob": true,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "hidden_act": "silu",
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "hc_sinkhorn_iters": 20,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false,
            "max_position_embeddings": 65536,
            "rope_scaling": [
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            ],
        ]
        let configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath:
            (dir as NSString).appendingPathComponent("config.json")))

        // -- model.safetensors.index.json.
        var weightMap: [String: String] = [:]
        for (name, entry) in inventory { weightMap[name] = entry.shard }
        let indexObj: [String: Any] = [
            "metadata": ["format": "dsv4"],
            "weight_map": weightMap,
        ]
        let indexData = try JSONSerialization.data(withJSONObject: indexObj, options: [.sortedKeys])
        try indexData.write(to: URL(fileURLWithPath:
            (dir as NSString).appendingPathComponent("model.safetensors.index.json")))

        return Snapshot(shardPaths: [shard1Path, shard2Path],
                        tensors: inventory,
                        mtpTensorNames: mtpNames,
                        residentNames: residentNames)
    }

    // MARK: - Safetensors writer

    private static func writeV4Shard(path: String,
                                     tensors: [(String, String, [Int], [UInt8])]) throws {
        var off: UInt64 = 0
        var headerDict: [String: Any] = [:]
        for (name, dtype, shape, bytes) in tensors {
            let begin = off
            let end = begin + UInt64(bytes.count)
            headerDict[name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [begin, end],
            ]
            off = end
        }
        headerDict["__metadata__"] = ["format": "dsv4"]
        let headerData = try JSONSerialization.data(withJSONObject: headerDict,
                                                    options: [.sortedKeys])
        var padded = headerData
        while padded.count % 8 != 0 { padded.append(0x20) }

        let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        precondition(fd >= 0, "open failed for \(path)")
        defer { close(fd) }
        var headerLenLE = UInt64(padded.count).littleEndian
        withUnsafeBytes(of: &headerLenLE) { raw in
            _ = write(fd, raw.baseAddress, 8)
        }
        padded.withUnsafeBytes { raw in
            _ = write(fd, raw.baseAddress, padded.count)
        }
        for (_, _, _, bytes) in tensors {
            bytes.withUnsafeBufferPointer { ptr in
                _ = write(fd, ptr.baseAddress, ptr.count)
            }
        }
    }
}
