import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the Metal `dequant_fp4_e2m1_gemv_simd` kernel against
/// `V4Quantization.gemvFP4`, which bulk-dequantizes each row to FP32
/// (exact: LUT value × power of two) and dots in FP32. The kernel interleaves
/// LUT decode + per-32-group ue8m0 scale + FMA in one inner loop with a
/// different summation order, so GEMV checks carry the fp16-chained 1e-2 bar
/// while the decode/packing checks are exact-equality.
@Suite struct DequantFP4E2M1GEMVTests {

    /// Hand-built payload: known codes and known scale bytes must dequantize
    /// to exactly lut[code] * 2^(b-127). Any mismatch is a packing/layout bug,
    /// not tolerance.
    @Test func cpuDequantIsExactAgainstLUT() {
        var packed = [UInt8]()
        var expectedPerGroup = [[Float]]()
        // Cycle every code through both nibble positions.
        var codes = [UInt8]()
        for i in 0..<V4Quantization.fp4GroupSize { codes.append(UInt8(i % 16)) }
        let scaleBytes: [UInt8] = [120, 127, 133]  // 2^-7, 2^0, 2^6
        for sByte in scaleBytes {
            for k in 0..<V4Quantization.fp4GroupSize {
                let code = codes[k]
                if (k & 1) == 0 {
                    packed.append(code)
                } else {
                    packed[packed.count - 1] |= code << 4
                }
            }
            let scale = V4Quantization.ue8m0Decode(sByte)
            expectedPerGroup.append(codes.map { V4Quantization.e2m1Decode($0) * scale })
        }
        let row = V4Quantization.FP4Row(packed: packed, scales: scaleBytes)
        let n = V4Quantization.fp4GroupSize * scaleBytes.count
        let decoded = V4Quantization.dequantizeFP4Row(row, n: n)
        #expect(decoded == expectedPerGroup.flatMap { $0 })
    }

    /// Encoder round-trip: quantized random rows dequantize within one e2m1
    /// step of the originals (never more than scale * 0.25 absolute), which
    /// proves quantize/dequantize agree on nibble order and scale indexing.
    @Test func quantizeRoundTripStaysWithinHalfStep() {
        var rng = SeedTree(0xF4A).key("fp4-roundtrip")
        let n = 32 * 9  // 9 groups: exercises group-boundary indexing
        let raw = (0..<n).map { _ in rng.uniform(-3.0, 3.0) }
        let row = V4Quantization.quantizeFP4Row(raw)
        let decoded = V4Quantization.dequantizeFP4Row(row, n: n)
        for g in 0..<(n / V4Quantization.fp4GroupSize) {
            let scale = V4Quantization.ue8m0Decode(row.scales[g])
            for k in 0..<V4Quantization.fp4GroupSize {
                let i = g * V4Quantization.fp4GroupSize + k
                // Coarsest e2m1 absolute half-gap is 1.0 (the 4→6 step), so
                // nearest-code error never exceeds one scale quantum.
                #expect(abs(decoded[i] - raw[i]) <= scale * 1.0 + 1e-7,
                        "element \(i): raw=\(raw[i]) decoded=\(decoded[i]) scale=\(scale)")
            }
        }
    }

    private static func runAndCompare(m: Int, n: Int, seed: UInt64,
                                      weightByteOffset: Int = 0) throws {
        precondition(n % V4Quantization.fp4GroupSize == 0)
        precondition(weightByteOffset % 4 == 0)
        let baseKey = "fp4-gemv-kernel-m\(m)-n\(n)"
        let rngKey = weightByteOffset == 0 ? baseKey : "\(baseKey)-off\(weightByteOffset)"
        var rng = SeedTree(seed).key(rngKey)

        var rows: [V4Quantization.FP4Row] = []
        rows.reserveCapacity(m)
        for _ in 0..<m {
            let raw = (0..<n).map { _ in rng.uniform(-2.0, 2.0) }
            rows.append(V4Quantization.quantizeFP4Row(raw))
        }
        let packed = rows.flatMap(\.packed)
        let scales = rows.flatMap(\.scales)

        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try DequantFP4E2M1GEMV(context: ctx)

        var paddedPacked = [UInt8](repeating: 0, count: packed.count + weightByteOffset)
        for i in 0..<packed.count { paddedPacked[weightByteOffset + i] = packed[i] }

        guard let wBuf = ctx.device.makeBuffer(
                bytes: paddedPacked, length: paddedPacked.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: scales, length: scales.count, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("Failed to allocate buffers"); return
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        kernel.encode(commandBuffer: cmd,
                      weights: wBuf, weightsOffset: weightByteOffset,
                      scales: sBuf,
                      x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let ref = V4Quantization.gemvFP4(rows: rows, x: xRef, n: n)
        let actual = Fp16Buffer.read(yBuf, count: m)

        let rel = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "M=\(m) N=\(n) off=\(weightByteOffset): rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// Remainder-only path: 7 groups < one full 8-group block.
    @Test func gemv_m16_n224_remainderOnly() throws {
        try Self.runAndCompare(m: 16, n: 32 * 7, seed: 0xD1)
    }

    /// One full block + 4 remainder groups.
    @Test func gemv_m32_n384_fullBlockPlusRemainder() throws {
        try Self.runAndCompare(m: 32, n: 32 * 12, seed: 0xD2)
    }

    /// Exact multiple of the 8-group block: 64 groups, no remainder.
    @Test func gemv_m64_n2048() throws {
        try Self.runAndCompare(m: 64, n: 2048, seed: 0xD3)
    }

    /// Expert w1/w3 K-dimension geometry: N = 4096 → 128 groups, 16 blocks.
    @Test func gemv_m128_n4096() throws {
        try Self.runAndCompare(m: 128, n: 4096, seed: 0xD4)
    }

    /// Non-zero 4-aligned weight base: the uint loads must stay correct when
    /// the packed region starts at a padded sub-tensor offset.
    @Test func gemv_weightsAt4AlignedOffset() throws {
        try Self.runAndCompare(m: 32, n: 2048, seed: 0xD5, weightByteOffset: 12)
    }

    /// Expert w2 geometry (M=4096 rows, K=1024) at reduced row count to keep
    /// the CPU reference fast; row count only shifts the grid.
    @Test func gemv_m512_n1024() throws {
        try Self.runAndCompare(m: 512, n: 1024, seed: 0xD6)
    }
}
