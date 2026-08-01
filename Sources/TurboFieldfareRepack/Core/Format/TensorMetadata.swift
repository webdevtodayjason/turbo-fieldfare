import Foundation

/// One physical tensor in a safetensors shard. Coordinates are absolute file
/// offsets so the writer can map and copy a single tensor without re-parsing
/// the shard header.
struct SourceTensor: Sendable, Hashable {
    enum Dtype: UInt8, Sendable, Hashable {
        case u32  = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
        /// Raw int8 container. DeepSeek V4 FP4 expert weights use this as the
        /// storage type for two e2m1 values packed low-nibble-first per byte.
        case i8   = 4
        /// Raw int64 (e.g. the hash-routing `tid2eid` token->expert table).
        case i64  = 5
        case f8e4m3 = 6
        case f8e8m0 = 7

        var elementBytes: Int {
            switch self {
            case .u32, .fp32: 4
            case .bf16, .fp16: 2
            case .i8, .f8e4m3, .f8e8m0: 1
            case .i64: 8
            }
        }
    }

    let name: String
    let shardPath: String
    let dtype: Dtype
    let shape: [UInt64]
    let absoluteOffset: UInt64
    let sizeBytes: UInt64
}

/// Per-scheme quantization descriptor for one tensor.
///
/// The repacker copies bytes unchanged, so the spec only needs to describe
/// the logical shape derivation and the manifest quant table; dequantization
/// is a runtime (V4F-02) concern.
enum QuantSpec: Sendable, Hashable {
    /// MLX affine: intN packed into U32, BF16 scales/biases per `group_size`.
    case mlxAffine(bits: Int)
    /// DeepSeek FP8: one e4m3 byte per element, one ue8m0 scale per
    /// `blockRows` x `blockCols` block (128x128 on disk for V4-Flash).
    case fp8BlockwiseE4M3(blockRows: Int, blockCols: Int, scaleFmt: String)
    /// DeepSeek FP4 expert weights: two e2m1 values per byte in an I8
    /// container (low nibble first along K), one ue8m0 scale per
    /// `scaleGroupK` elements along K (32 on disk for V4-Flash).
    case fp4E2M1E8M0(scaleGroupK: Int)

    /// Nominal weight precision in bits. Keeps the manifest probing code and
    /// the packed-shape helpers source-compatible with the old `bits` field.
    var bits: Int {
        switch self {
        case .mlxAffine(let bits): return bits
        case .fp8BlockwiseE4M3: return 8
        case .fp4E2M1E8M0: return 4
        }
    }
}
