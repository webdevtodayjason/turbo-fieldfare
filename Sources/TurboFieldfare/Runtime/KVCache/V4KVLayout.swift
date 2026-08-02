import Foundation

/// DeepSeek V4-Flash per-layer attention variant, derived from the
/// checkpoint's `compress_ratios` list (recon: V4F-reference-notes.md §1-3).
///
/// - `.csa`: compress_ratio 4 — overlapped channel-split pooling, lightning
///   indexer top-512 sparse selection over compressed entries.
/// - `.hca`: compress_ratio 128 — non-overlapped pooling, dense attention
///   over all compressed entries (no indexer).
/// - `.passthrough`: compress_ratio 0 (Flash layers 0, 1, 42) — pure
///   128-token sliding-window MQA, no compressor, no indexer.
public enum V4LayerKind: Sendable, Equatable {
    case csa
    case hca
    case passthrough
}

/// Static geometry for the V4-Flash attention stack (43 layers, head_dim
/// 512, 64 q heads, shared-KV MQA, trailing 64 RoPE dims, 128-token
/// uncompressed window branch).
public struct V4CacheConfig: Sendable, Equatable {
    public let compressRatios: [Int]
    public let headDim: Int
    public let ropeDim: Int
    public let window: Int
    public let numQHeads: Int

    public init(compressRatios: [Int],
                headDim: Int = 512,
                ropeDim: Int = 64,
                window: Int = 128,
                numQHeads: Int = 64) {
        precondition(!compressRatios.isEmpty, "compressRatios must not be empty")
        precondition(headDim > ropeDim, "headDim must exceed ropeDim")
        precondition((headDim - ropeDim) % 64 == 0,
                     "non-rope dims must tile into 64-wide FP8 blocks")
        precondition(ropeDim % 2 == 0, "ropeDim must be even")
        self.compressRatios = compressRatios
        self.headDim = headDim
        self.ropeDim = ropeDim
        self.window = window
        self.numQHeads = numQHeads
    }

    public var numLayers: Int { compressRatios.count }
    public var nonRopeDim: Int { headDim - ropeDim }

    /// Published DeepSeek-V4-Flash layout: ratio 0 on layers 0 and 1;
    /// ratio 4 (CSA) on even layers 2...40; ratio 128 (HCA) on odd 3...41.
    public static var deepSeekV4Flash: V4CacheConfig {
        let ratios = (0..<43).map { layer -> Int in
            if layer < 2 { return 0 }
            return layer % 2 == 0 ? 4 : 128
        }
        return V4CacheConfig(compressRatios: ratios)
    }

    public func kind(layer: Int) -> V4LayerKind {
        switch compressRatios[layer] {
        case 4:   return .csa
        case 128: return .hca
        default:  return .passthrough
        }
    }

    public func compressRatio(layer: Int) -> Int { compressRatios[layer] }
}

/// Storage layout of one compressed KV entry (CSA or HCA).
///
/// Per the V4 storage plan the non-RoPE 448 dims are stored as true FP8
/// e4m3 (not QAT-simulated BF16 as in the PyTorch reference): 7 blocks of
/// 64 dims, each with one ue8m0 (power-of-2-only) scale. The trailing 64
/// RoPE dims stay FP16 for positional precision (the reference keeps them
/// BF16; this runtime's half-precision carrier is FP16).
public enum V4KVLayout {
    /// FP8 block width along the channel axis (matches act_quant block=64).
    public static let fp8Block = 64

    /// Bytes per entry of the e4m3 value buffer (== nonRopeDim).
    public static func valueStride(config: V4CacheConfig) -> Int {
        config.nonRopeDim
    }

    /// ue8m0 scale bytes per entry (one per 64-dim block, padded to 8).
    public static func scaleStride(config: V4CacheConfig) -> Int {
        let blocks = config.nonRopeDim / fp8Block
        return (blocks + 3) & ~3   // 4-byte alignment padding
    }

    /// Bytes per entry of the FP16 rope-dim buffer.
    public static func ropeStride(config: V4CacheConfig) -> Int {
        config.ropeDim * MemoryLayout<Float16>.size
    }

    public static var fp16Size: Int { MemoryLayout<Float16>.size }
}

/// Scalar FP8 helpers shared by the cache manager's CPU-side quantize
/// helpers (tests, synthetic writes) and used as the reference for the
/// Metal compressor kernel.
public enum V4FP8 {
    /// Largest finite e4m3 magnitude (S.1111.110).
    public static let e4m3Max: Float = 448.0

    /// Round-to-nearest-even e4m3 encoding. Values above 448 clamp to 448
    /// (matching act_quant's saturating cast); NaN maps to S.1111.111.
    public static func e4m3Encode(_ x: Float) -> UInt8 {
        if x.isNaN { return 0x7F }
        let sign: UInt8 = x.sign == .minus ? 0x80 : 0x00
        var ax = abs(x)
        if ax > e4m3Max { ax = e4m3Max }
        if ax < 0x1p-6 {
            // Subnormal grid: value = m * 2^-9, m in 0...8 (8 == min normal).
            let m = Int((ax * 512.0).rounded(.toNearestOrEven))
            if m >= 8 { return sign | 0x08 }
            return sign | UInt8(m)
        }
        var e = Int(ax.log2roundedDown())
        let mantissa = ((ax / scalbnf(1.0, Int32(e))) - 1.0) * 8.0
        var m = Int(mantissa.rounded(.toNearestOrEven))
        if m == 8 { m = 0; e += 1 }
        if e > 8 { return sign | 0x7E }      // clamp to 448 (exp field 15, mantissa 6)
        let expField = UInt8(e + 7)
        return sign | (expField << 3) | UInt8(m)
    }

    public static func e4m3Decode(_ b: UInt8) -> Float {
        let sign: Float = (b & 0x80) != 0 ? -1.0 : 1.0
        let e = Int((b >> 3) & 0x0F)
        let m = Int(b & 0x07)
        if e == 0 { return sign * Float(m) * 0x1p-9 }
        if e == 15 && m == 7 { return .nan }
        return sign * (1.0 + Float(m) / 8.0) * scalbnf(1.0, Int32(e - 7))
    }

    /// ue8m0: exponent-only float, value = 2^(b - 127). The reference
    /// `fast_round_scale` rounds a scale UP to the next power of two.
    public static func ue8m0Encode(_ scale: Float) -> UInt8 {
        precondition(scale > 0 && scale.isFinite, "ue8m0 scale must be positive finite")
        let e = Int(ceil(log2(scale)))
        let clamped = max(-127, min(128, e))
        return UInt8(clamped + 127)
    }

    public static func ue8m0Decode(_ b: UInt8) -> Float {
        scalbnf(1.0, Int32(Int(b) - 127))
    }

    /// Per-64-block activation scale, mirroring the reference `act_quant`:
    /// amax floored at 1e-4, divided by 448, rounded up to a power of two.
    public static func blockScale(amax: Float) -> Float {
        ue8m0Decode(ue8m0Encode(max(amax, 1e-4) / e4m3Max))
    }
}

private extension Float {
    /// floor(log2(self)) computed on the exact binary representation
    /// (avoids log2() rounding at exact powers of two).
    func log2roundedDown() -> Float {
        Float(exponent)
    }
}
