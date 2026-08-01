import Foundation

/// Parses `model.safetensors.index.json` and the quantization block of
/// `config.json`. Two config families are recognised:
///
/// - MLX (`quantization`): affine intN with `bits` / `group_size` / `mode`
///   plus per-tensor overrides (Gemma 4 community re-pack).
/// - DeepSeek (`quantization_config`): native FP8 `quant_method: "fp8"`,
///   `fmt: "e4m3"`, `scale_fmt: "ue8m0"`, `weight_block_size: [128, 128]`,
///   with the FP4 expert scheme described by the top-level `expert_dtype`.
enum IndexLoader {

    struct SourceMetadata {
        let indexPath: String
        let configPath: String
        let indexSha256Hex: String
        /// `tensor_name -> shard_filename`
        let weightMap: [String: String]
        /// Which config family produced this metadata.
        let family: ModelFamily
        /// Base bits / group_size / mode for any tensor not in the override table.
        let baseBits: Int
        let baseGroupSize: Int
        let baseMode: String
        /// Per-tensor overrides (keyed by tensor name **without** the trailing
        /// `.weight` — matches the way `config.json` writes them). MLX only.
        let bitsOverrides: [String: QuantSpec]
        /// DeepSeek FP8/FP4 quant facts (nil for MLX sources).
        let deepseekQuant: DeepSeekQuantDescriptor?
        /// Resolved set of shard files referenced by the index, in
        /// encounter order. Order is stable enough for sequential I/O.
        let shardFilenames: [String]
    }

    static func load(snapshotDir: String) throws -> SourceMetadata {
        let indexPath  = (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json")
        let configPath = (snapshotDir as NSString).appendingPathComponent("config.json")

        let weightMap: [String: String]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let m = root["weight_map"] as? [String: String] else {
                throw RepackError.indexJsonInvalid(path: indexPath, detail: "no weight_map")
            }
            weightMap = m
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.indexJsonInvalid(path: indexPath, detail: "\(error)")
        }

        let indexSha = try Sha256Stream.hashFile(path: indexPath)

        let configRoot: [String: Any]
        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
            }
            configRoot = root
        } catch let e as RepackError {
            throw e
        } catch {
            throw RepackError.configJsonInvalid(path: configPath, detail: "\(error)")
        }

        let family: ModelFamily
        let baseBits: Int
        let baseGroup: Int
        let baseMode: String
        var overrides: [String: QuantSpec] = [:]
        var deepseekQuant: DeepSeekQuantDescriptor?

        if configRoot["quantization_config"] != nil {
            // DeepSeek native FP8/FP4 scheme.
            family = .deepseekV4Flash
            let quant = try parseDeepSeekQuantizationConfig(configRoot: configRoot,
                                                            configPath: configPath)
            deepseekQuant = quant
            baseBits = 8
            baseGroup = quant.weightBlockRows
            baseMode = DeepSeekQuantDescriptor.fp8SchemeName
        } else if let quant = configRoot["quantization"] as? [String: Any] {
            // MLX affine scheme.
            family = .gemma4
            var bits = 4
            var group = 64
            var mode = "affine"
            if let b = quant["bits"] as? Int       { bits  = b }
            if let g = quant["group_size"] as? Int { group = g }
            if let m = quant["mode"] as? String    { mode  = m }
            for (k, v) in quant where !(k == "bits" || k == "group_size" || k == "mode") {
                guard let entry = v as? [String: Any] else { continue }
                let entryBits = (entry["bits"] as? Int) ?? bits
                let g         = (entry["group_size"] as? Int) ?? group
                guard g == group else {
                    throw RepackError.configJsonInvalid(
                        path: configPath,
                        detail: "quantization override \(k) group_size \(g) != base \(group)")
                }
                overrides[k] = .mlxAffine(bits: entryBits)
            }
            baseBits = bits
            baseGroup = group
            baseMode = mode
        } else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "no quantization slot")
        }

        var seen = Set<String>()
        var shards: [String] = []
        for k in weightMap.keys.sorted() {
            let shard = weightMap[k]!
            if !seen.contains(shard) { seen.insert(shard); shards.append(shard) }
        }

        return SourceMetadata(indexPath: indexPath, configPath: configPath,
                              indexSha256Hex: indexSha,
                              weightMap: weightMap,
                              family: family,
                              baseBits: baseBits, baseGroupSize: baseGroup,
                              baseMode: baseMode,
                              bitsOverrides: overrides,
                              deepseekQuant: deepseekQuant,
                              shardFilenames: shards)
    }

    /// Validates `config.json -> quantization_config` for the DeepSeek native
    /// scheme and lifts the facts the planner/manifest need.
    private static func parseDeepSeekQuantizationConfig(configRoot: [String: Any],
                                                        configPath: String) throws
        -> DeepSeekQuantDescriptor {
        guard let quant = configRoot["quantization_config"] as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath,
                                                detail: "quantization_config is not a dict")
        }
        func reject(_ detail: String) -> RepackError {
            RepackError.configJsonInvalid(path: configPath, detail: detail)
        }
        guard (quant["quant_method"] as? String) == "fp8" else {
            throw reject("quantization_config.quant_method != \"fp8\"")
        }
        guard (quant["fmt"] as? String) == "e4m3" else {
            throw reject("quantization_config.fmt != \"e4m3\"")
        }
        let scaleFmt = (quant["scale_fmt"] as? String) ?? "ue8m0"
        guard scaleFmt == "ue8m0" else {
            throw reject("quantization_config.scale_fmt != \"ue8m0\"")
        }
        guard let block = quant["weight_block_size"] as? [Any], block.count == 2,
              let rows = (block[0] as? NSNumber)?.intValue,
              let cols = (block[1] as? NSNumber)?.intValue,
              rows > 0, cols > 0 else {
            throw reject("quantization_config.weight_block_size missing or malformed")
        }
        // FP4 expert storage is described by the top-level `expert_dtype`;
        // the on-disk scale blocking (per 32 along K) is fixed by the format
        // and confirmed from the shard headers.
        let expertDtype = (configRoot["expert_dtype"] as? String) ?? "fp4"
        guard expertDtype == "fp4" else {
            throw reject("unsupported expert_dtype \"\(expertDtype)\" (expected \"fp4\")")
        }
        return DeepSeekQuantDescriptor(weightBlockRows: rows,
                                       weightBlockCols: cols,
                                       scaleFmt: scaleFmt,
                                       expertDtype: expertDtype,
                                       expertScaleGroupK: DeepSeekQuantDescriptor.fp4ExpertScaleGroupK)
    }

    /// Resolves the spec for one tensor name (with or without `.weight`).
    /// MLX sources consult the override table; DeepSeek sources are uniform
    /// per tensor class, so the planner resolves those by dtype instead.
    static func quantSpec(forTensor name: String,
                                 meta: SourceMetadata) -> QuantSpec {
        let stripped = name.hasSuffix(".weight")
            ? String(name.dropLast(".weight".count))
            : name
        if let o = meta.bitsOverrides[stripped] { return o }
        if let q = meta.deepseekQuant {
            return .fp8BlockwiseE4M3(blockRows: q.weightBlockRows,
                                     blockCols: q.weightBlockCols,
                                     scaleFmt: q.scaleFmt)
        }
        return .mlxAffine(bits: meta.baseBits)
    }
}

/// FP8/FP4 quant facts for a DeepSeek-family source.
struct DeepSeekQuantDescriptor: Sendable, Hashable {
    /// Manifest scheme strings (also used as `RepackPlan.baseMode`).
    static let fp8SchemeName = "fp8-blockwise-e4m3"
    static let fp4SchemeName = "fp4-e2m1-e8m0"
    /// FP4 expert weights carry one ue8m0 scale per 32 elements along K.
    static let fp4ExpertScaleGroupK = 32
    /// FP4 storage packs two e2m1 values per byte, low nibble first along K.
    static let fp4ElementsPerByte = 2

    let weightBlockRows: Int
    let weightBlockCols: Int
    let scaleFmt: String
    let expertDtype: String
    let expertScaleGroupK: Int
}
