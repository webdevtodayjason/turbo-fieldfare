import Foundation

/// CPU reference and fixture-generation helpers for the DeepSeek V4-Flash
/// quantization formats (V4F-02). Mirror of `Quantization` for the V4 stack:
///
///   FP4 (routed experts): e2m1 codes, two per byte, low nibble = element 2i,
///   one ue8m0 (power-of-two) scale per 32 elements along K. No bias.
///
///   FP8 (everything else): e4m3 codes, one byte per element, one ue8m0 scale
///   per 128x128 2-D block, grid row-major over (ceil(M/128), ceil(N/128)).
///   No bias.
///
/// Decode of both formats is exact in FP32 (LUT values and powers of two are
/// exactly representable), so CPU dequant comparisons are exact-equality
/// checks; only GEMV accumulation-order noise needs a tolerance.
public enum V4Quantization {

    /// Elements per ue8m0 scale along K in the FP4 expert format.
    public static let fp4GroupSize: Int = 32

    /// Edge length of one FP8 block-scale tile.
    public static let fp8BlockSize: Int = 128

    /// Largest finite e2m1 magnitude.
    public static let e2m1Max: Float = 6

    /// Largest finite e4m3 magnitude.
    public static let e4m3Max: Float = 448

    // MARK: - e2m1 (FP4)

    /// e2m1 value table, index = 4-bit code (sign in bit 3). Exact in FP32.
    public static let e2m1Table: [Float] = [
        0, 0.5, 1, 1.5, 2, 3, 4, 6,
        -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    ]

    @inline(__always)
    public static func e2m1Decode(_ code: UInt8) -> Float {
        e2m1Table[Int(code & 0x0F)]
    }

    /// Nearest-code e2m1 encode of an already scale-normalized value.
    /// Ties resolve toward the smaller magnitude (first match wins).
    public static func e2m1Encode(_ v: Float) -> UInt8 {
        let negative = v < 0
        let mag = min(abs(v), e2m1Max)
        var best: UInt8 = 0
        var bestDiff = Float.infinity
        for code in 0..<UInt8(8) {
            let diff = abs(e2m1Table[Int(code)] - mag)
            if diff < bestDiff {
                bestDiff = diff
                best = code
            }
        }
        return negative ? best | 0x8 : best
    }

    // MARK: - e4m3 (FP8)

    /// e4m3 decode: 1 sign, 4 exponent bits (bias 7), 3 mantissa bits.
    /// The spec's NaN code (e=15, m=7) decodes here as 480; checkpoints never
    /// carry it and the encoder below never emits it.
    @inline(__always)
    public static func e4m3Decode(_ q: UInt8) -> Float {
        let negative = (q & 0x80) != 0
        let e = Int((q >> 3) & 0x0F)
        let m = Int(q & 0x07)
        let mag: Float
        if e == 0 {
            mag = Float(m) * 0x1p-9   // subnormal step 2^(1-7-3) = 2^-9
        } else {
            // (8 + m)/8 * 2^(e-7); significand in [1, 2).
            mag = Float(sign: .plus, exponent: e - 7, significand: Float(8 + m) / 8)
        }
        return negative ? -mag : mag
    }

    /// Nearest-code e4m3 encode with saturation to +/-448. Never emits the
    /// NaN code 0x7F. Linear scan over the 127 magnitude codes; fixture-only.
    public static func e4m3Encode(_ v: Float) -> UInt8 {
        let negative = v < 0
        let mag = min(abs(v), e4m3Max)
        var best: UInt8 = 0
        var bestDiff = Float.infinity
        for code in 0..<UInt8(0x7F) {
            let diff = abs(e4m3Decode(code) - mag)
            if diff < bestDiff {
                bestDiff = diff
                best = code
            }
        }
        return negative ? best | 0x80 : best
    }

    // MARK: - ue8m0 scales

    /// ue8m0 decode: unsigned 8-bit exponent, value 2^(b - 127). Exact power
    /// of two in FP32 for all b (b=0 and b=255 land in subnormal/inf; real
    /// checkpoints stay in [1, 254]).
    @inline(__always)
    public static func ue8m0Decode(_ b: UInt8) -> Float {
        Float(sign: .plus, exponent: Int(b) - 127, significand: 1.0)
    }

    /// Smallest power-of-two scale such that `scale * maxCodeValue >= amax`
    /// (ceil-log2, matching the reference `fast_round_scale` behavior).
    /// All-zero groups get scale 1 (byte 127).
    public static func ue8m0Encode(forMaxMagnitude amax: Float, maxCodeValue: Float) -> UInt8 {
        guard amax > 0, amax.isFinite else { return 127 }
        let exponent = ceil(log2(Double(amax / maxCodeValue)))
        let byte = Int(exponent) + 127
        return UInt8(clamping: max(1, min(254, byte)))
    }

    // MARK: - FP4 rows (routed experts)

    /// One quantized FP4 row: N/2 packed e2m1 bytes (low nibble = element 2i,
    /// high = 2i+1) plus N/32 ue8m0 scale bytes.
    public struct FP4Row {
        public let packed: [UInt8]
        public let scales: [UInt8]

        public init(packed: [UInt8], scales: [UInt8]) {
            self.packed = packed
            self.scales = scales
        }
    }

    public static func quantizeFP4Row(_ row: [Float]) -> FP4Row {
        precondition(row.count % fp4GroupSize == 0,
                     "row length \(row.count) is not a multiple of \(fp4GroupSize)")
        let nGroups = row.count / fp4GroupSize
        var packed = [UInt8](repeating: 0, count: row.count / 2)
        var scales = [UInt8](repeating: 0, count: nGroups)
        for g in 0..<nGroups {
            var amax: Float = 0
            for k in 0..<fp4GroupSize {
                amax = max(amax, abs(row[g * fp4GroupSize + k]))
            }
            let sByte = ue8m0Encode(forMaxMagnitude: amax, maxCodeValue: e2m1Max)
            scales[g] = sByte
            let invScale = 1.0 / ue8m0Decode(sByte)
            for k in 0..<fp4GroupSize {
                let code = e2m1Encode(row[g * fp4GroupSize + k] * invScale)
                let byteIdx = g * (fp4GroupSize / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | code
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (code << 4)
                }
            }
        }
        return FP4Row(packed: packed, scales: scales)
    }

    /// Exact dequant: `w[i] = e2m1Table[code[i]] * 2^(scale[i/32] - 127)`.
    public static func dequantizeFP4Row(_ r: FP4Row, n: Int) -> [Float] {
        precondition(n == r.packed.count * 2)
        precondition(n % fp4GroupSize == 0)
        var out = [Float](repeating: 0, count: n)
        for g in 0..<(n / fp4GroupSize) {
            let scale = ue8m0Decode(r.scales[g])
            for k in 0..<fp4GroupSize {
                let byte = r.packed[g * (fp4GroupSize / 2) + (k / 2)]
                let code = (k & 1) == 0 ? (byte & 0x0F) : (byte >> 4)
                out[g * fp4GroupSize + k] = e2m1Decode(code) * scale
            }
        }
        return out
    }

    /// Bulk-dequant FP32 GEMV reference: dequantize each row exactly, dot in
    /// FP32. Independent summation order from the SIMD kernel.
    public static func gemvFP4(rows: [FP4Row], x: [Float], n: Int) -> [Float] {
        rows.map { row in
            let w = dequantizeFP4Row(row, n: n)
            var acc: Float = 0
            for i in 0..<n { acc += w[i] * x[i] }
            return acc
        }
    }

    // MARK: - FP8 128x128 block matrices (dense weights)

    /// One quantized FP8 matrix: M*N e4m3 bytes row-major plus a ue8m0 scale
    /// grid of (M/128) x (N/128) bytes, row-major over the grid.
    public struct FP8BlockMatrix {
        public let m: Int
        public let n: Int
        public let codes: [UInt8]
        public let scales: [UInt8]

        public init(m: Int, n: Int, codes: [UInt8], scales: [UInt8]) {
            self.m = m
            self.n = n
            self.codes = codes
            self.scales = scales
        }
    }

    public static func quantizeFP8BlockMatrix(_ rows: [[Float]]) -> FP8BlockMatrix {
        let m = rows.count
        let n = rows.first?.count ?? 0
        precondition(m > 0 && n > 0)
        precondition(m % fp8BlockSize == 0 && n % fp8BlockSize == 0,
                     "FP8 block matrix \(m)x\(n) must be a multiple of \(fp8BlockSize) in both dims")
        for row in rows { precondition(row.count == n) }

        let gridRows = m / fp8BlockSize
        let gridCols = n / fp8BlockSize
        var codes = [UInt8](repeating: 0, count: m * n)
        var scales = [UInt8](repeating: 0, count: gridRows * gridCols)
        for br in 0..<gridRows {
            for bc in 0..<gridCols {
                var amax: Float = 0
                for i in 0..<fp8BlockSize {
                    for j in 0..<fp8BlockSize {
                        amax = max(amax, abs(rows[br * fp8BlockSize + i][bc * fp8BlockSize + j]))
                    }
                }
                let sByte = ue8m0Encode(forMaxMagnitude: amax, maxCodeValue: e4m3Max)
                scales[br * gridCols + bc] = sByte
                let invScale = 1.0 / ue8m0Decode(sByte)
                for i in 0..<fp8BlockSize {
                    for j in 0..<fp8BlockSize {
                        let r = br * fp8BlockSize + i
                        let c = bc * fp8BlockSize + j
                        codes[r * n + c] = e4m3Encode(rows[r][c] * invScale)
                    }
                }
            }
        }
        return FP8BlockMatrix(m: m, n: n, codes: codes, scales: scales)
    }

    /// Exact dequant to a row-major FP32 matrix.
    public static func dequantizeFP8BlockMatrix(_ mat: FP8BlockMatrix) -> [Float] {
        let gridCols = mat.n / fp8BlockSize
        var out = [Float](repeating: 0, count: mat.m * mat.n)
        for r in 0..<mat.m {
            for c in 0..<mat.n {
                let scale = ue8m0Decode(mat.scales[(r / fp8BlockSize) * gridCols + (c / fp8BlockSize)])
                out[r * mat.n + c] = e4m3Decode(mat.codes[r * mat.n + c]) * scale
            }
        }
        return out
    }

    /// Bulk-dequant FP32 GEMV reference.
    public static func gemvFP8(matrix: FP8BlockMatrix, x: [Float]) -> [Float] {
        let w = dequantizeFP8BlockMatrix(matrix)
        return (0..<matrix.m).map { r in
            var acc: Float = 0
            let base = r * matrix.n
            for i in 0..<matrix.n { acc += w[base + i] * x[i] }
            return acc
        }
    }
}
