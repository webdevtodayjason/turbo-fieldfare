import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// Compares the Metal `dequant_fp8_e4m3_gemv_simd` and `embed_lookup_fp8`
/// kernels against `V4Quantization` FP8 references. Decode is exact (e4m3
/// values and ue8m0 powers of two are FP32-representable); the GEMV checks
/// carry a 5e-3 bar for FP32-accumulation order + half output rounding.
@Suite struct DequantFP8BlockGEMVTests {

    /// e4m3 decode must hit the documented format values exactly: subnormal
    /// step 2^-9, normals (8+m)*2^(e-10), sign bit, saturation value 448.
    @Test func e4m3DecodeHitsFormatValuesExactly() {
        #expect(V4Quantization.e4m3Decode(0x00) == 0)
        #expect(V4Quantization.e4m3Decode(0x01) == 0x1p-9)          // min subnormal
        #expect(V4Quantization.e4m3Decode(0x07) == 7 * 0x1p-9)      // max subnormal
        #expect(V4Quantization.e4m3Decode(0x08) == 0x1p-6)          // min normal
        #expect(V4Quantization.e4m3Decode(0x38) == 1.0)             // e=7, m=0
        #expect(V4Quantization.e4m3Decode(0x3E) == 1.75)            // e=7, m=6
        #expect(V4Quantization.e4m3Decode(0x7E) == 448)             // max finite
        #expect(V4Quantization.e4m3Decode(0x80) == 0)               // negative zero
        #expect(V4Quantization.e4m3Decode(0xBE) == -1.75)
        #expect(V4Quantization.e4m3Decode(0xFE) == -448)
    }

    private static func runAndCompare(m: Int, n: Int, seed: UInt64) throws {
        precondition(m % V4Quantization.fp8BlockSize == 0)
        precondition(n % V4Quantization.fp8BlockSize == 0)
        var rng = SeedTree(seed).key("fp8-gemv-kernel-m\(m)-n\(n)")

        let raw = (0..<m).map { _ in (0..<n).map { _ in rng.uniform(-4.0, 4.0) } }
        let matrix = V4Quantization.quantizeFP8BlockMatrix(raw)

        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try DequantFP8BlockGEMV(context: ctx)

        guard let wBuf = ctx.device.makeBuffer(
                bytes: matrix.codes, length: matrix.codes.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: matrix.scales, length: matrix.scales.count, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("Failed to allocate buffers"); return
        }

        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        kernel.encode(commandBuffer: cmd,
                      weights: wBuf, scales: sBuf,
                      x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let ref = V4Quantization.gemvFP8(matrix: matrix, x: xRef)
        let actual = Fp16Buffer.read(yBuf, count: m)

        let rel = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(rel < Tolerance.fp16Reduction,
                "M=\(m) N=\(n): rel=\(rel) maxAbs=\(maxAbs)")
    }

    /// Single scale block in both dims.
    @Test func gemv_m128_n128() throws {
        try Self.runAndCompare(m: 128, n: 128, seed: 0xE1)
    }

    /// 2x3 scale grid: exercises the 2-D grid indexing in both directions.
    @Test func gemv_m256_n384() throws {
        try Self.runAndCompare(m: 256, n: 384, seed: 0xE2)
    }

    /// lm_head K-dimension: N = 4096 → 32-wide scale grid.
    @Test func gemv_m128_n4096() throws {
        try Self.runAndCompare(m: 128, n: 4096, seed: 0xE3)
    }

    /// Wide dynamic range across blocks: one block near zero forces very
    /// different ue8m0 exponents per block.
    @Test func gemv_blocksWithDisparateScales() throws {
        let m = 256, n = 256
        var rng = SeedTree(0xE4).key("fp8-disparate-scales")
        var raw = (0..<m).map { _ in (0..<n).map { _ in rng.uniform(-1.0, 1.0) } }
        // Block (0,0) tiny, block (1,1) large.
        for i in 0..<128 { for j in 0..<128 { raw[i][j] *= 1e-3 } }
        for i in 128..<256 { for j in 128..<256 { raw[i][j] *= 300.0 } }
        let matrix = V4Quantization.quantizeFP8BlockMatrix(raw)
        let xFp16 = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let xRef = xFp16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try DequantFP8BlockGEMV(context: ctx)
        guard let wBuf = ctx.device.makeBuffer(
                bytes: matrix.codes, length: matrix.codes.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: matrix.scales, length: matrix.scales.count, options: .storageModeShared),
              let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: m) else {
            Issue.record("Failed to allocate buffers"); return
        }
        guard let cmd = ctx.queue.makeCommandBuffer() else {
            Issue.record("Failed to make command buffer"); return
        }
        kernel.encode(commandBuffer: cmd,
                      weights: wBuf, scales: sBuf, x: xBuf, y: yBuf,
                      m: UInt32(m), n: UInt32(n))
        cmd.commit()
        cmd.waitUntilCompleted()
        #expect(cmd.error == nil)

        let ref = V4Quantization.gemvFP8(matrix: matrix, x: xRef)
        let actual = Fp16Buffer.read(yBuf, count: m)
        // Large block dominates the max-ref norm; check the small-scale rows
        // with an absolute floor instead.
        let headRef = Array(ref[0..<128])
        let headAct = Array(actual[0..<128])
        #expect(RelError.boundedRel(actual: headAct, reference: headRef, absFloor: 5e-2) < 5e-3)
        let tailRef = Array(ref[128..<256])
        let tailAct = Array(actual[128..<256])
        #expect(RelError.compute(actual: tailAct, reference: tailRef) < 5e-3)
    }

    /// embed_lookup_fp8: row gather + block scale + out_scale. Each output
    /// element is one exact product, so only the final half rounding costs
    /// precision.
    @Test func embedLookupMatchesReference() throws {
        let vocab = 256, d = 256
        var rng = SeedTree(0xE5).key("fp8-embed-lookup")
        let raw = (0..<vocab).map { _ in (0..<d).map { _ in rng.uniform(-2.0, 2.0) } }
        let matrix = V4Quantization.quantizeFP8BlockMatrix(raw)
        let dequant = V4Quantization.dequantizeFP8BlockMatrix(matrix)
        let outScale: Float = 2.0

        let ctx = try MetalContext()
        let kernel = try EmbedLookupFP8(context: ctx)
        guard let tBuf = ctx.device.makeBuffer(
                bytes: matrix.codes, length: matrix.codes.count, options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(
                bytes: matrix.scales, length: matrix.scales.count, options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: d) else {
            Issue.record("Failed to allocate buffers"); return
        }

        for token in [UInt32(0), 17, 127, 128, 200, 255] {
            guard let cmd = ctx.queue.makeCommandBuffer() else {
                Issue.record("Failed to make command buffer"); return
            }
            kernel.encode(commandBuffer: cmd,
                          table: tBuf, scales: sBuf, out: outBuf,
                          tokenID: token, d: UInt32(d), outScale: outScale)
            cmd.commit()
            cmd.waitUntilCompleted()
            #expect(cmd.error == nil)

            let actual = Fp16Buffer.read(outBuf, count: d)
            let base = Int(token) * d
            let ref = (0..<d).map { dequant[base + $0] * outScale }
            let rel = RelError.boundedRel(actual: actual, reference: ref, absFloor: 1e-3)
            #expect(rel < 2e-3, "token \(token): rel=\(rel)")
        }
    }
}
