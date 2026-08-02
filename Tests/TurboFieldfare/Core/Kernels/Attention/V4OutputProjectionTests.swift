import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-03 wave-2: grouped output projection — 8 groups of (1024 x 4096) bf16
/// wo_a into the o_lora_rank low-rank, then the FP8 8192 -> 4096 wo_b sum.
/// CPU reference dequantizes both stages exactly; the gate absorbs fp16
/// round-trips at the o, low-rank, and output boundaries.
@Suite struct V4OutputProjectionTests {

    private static let groups = 8
    private static let groupDim = 4096
    private static let loraRank = 1024
    private static let hidden = 4096

    // MARK: - bf16 helpers (wo_a arrives bf16 per recon §2)

    static func bf16Encode(_ v: Float) -> UInt16 {
        let b = v.bitPattern
        let rounded = (b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    static func bf16Decode(_ e: UInt16) -> Float {
        Float(bitPattern: UInt32(e) << 16)
    }

    // MARK: - CPU reference

    /// lowRank[g*1024 + r] = dot(woA[(g*1024+r), :], o[g*4096 .. +4096]).
    private static func refGrouped(o: [Float], woA: [Float]) -> [Float] {
        var y = [Float](repeating: 0, count: groups * loraRank)
        for g in 0..<groups {
            for r in 0..<loraRank {
                let row = (g * loraRank + r) * groupDim
                var acc: Float = 0
                for i in 0..<groupDim {
                    acc += woA[row + i] * o[g * groupDim + i]
                }
                y[g * loraRank + r] = acc
            }
        }
        return y
    }

    @Test func groupedDownStage_matchesCPUReference() throws {
        var rng = SeedTree(0xC01).key("oproj-down")
        let o16 = (0..<(Self.groups * Self.groupDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let woA16 = (0..<(Self.groups * Self.loraRank * Self.groupDim)).map { _ -> UInt16 in
            Self.bf16Encode(rng.uniform(-0.05, 0.05))
        }
        let ctx = try MetalContext()
        let proj = try V4OutputProjection(device: ctx.device)
        guard let oBuf = Fp16Buffer.make(ctx.device, halves: o16),
              let wBuf = ctx.device.makeBuffer(
                bytes: woA16, length: woA16.count * 2,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encodeGrouped(commandBuffer: cb, o: oBuf, woA: wBuf)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(proj.lowRankBuffer, count: Self.groups * Self.loraRank)
        let ref = Self.refGrouped(o: o16.map { Float($0) },
                                  woA: woA16.map { Self.bf16Decode($0) })
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16Reduction, "grouped down rel=\(rel)")
    }

    @Test func fullProjection_matchesCPUReferenceChain() throws {
        var rng = SeedTree(0xC02).key("oproj-full")
        let o16 = (0..<(Self.groups * Self.groupDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let woA16 = (0..<(Self.groups * Self.loraRank * Self.groupDim)).map { _ -> UInt16 in
            Self.bf16Encode(rng.uniform(-0.05, 0.05))
        }
        // FP8 wo_b [4096, 8192]: random codes + tame power-of-two scales.
        let nCodes = Self.hidden * Self.groups * Self.loraRank
        let codes = (0..<nCodes).map { _ -> UInt8 in
            var b = UInt8(rng.uniform(0, 256)) & 0x7F
            if b == 0x7F { b = 0x7E }
            if rng.uniform(0, 1) < 0.5 { b |= 0x80 }
            return b
        }
        let gridRows = Self.hidden / 128, gridCols = (Self.groups * Self.loraRank) / 128
        let scales = (0..<(gridRows * gridCols)).map { _ -> UInt8 in
            UInt8(clamping: 127 + Int(rng.uniform(-4, 1)))
        }
        let woB = V4Quantization.FP8BlockMatrix(m: Self.hidden, n: Self.groups * Self.loraRank,
                                                codes: codes, scales: scales)

        let ctx = try MetalContext()
        let proj = try V4OutputProjection(device: ctx.device)
        guard let oBuf = Fp16Buffer.make(ctx.device, halves: o16),
              let wABuf = ctx.device.makeBuffer(bytes: woA16, length: woA16.count * 2,
                                                options: .storageModeShared),
              let wBBuf = ctx.device.makeBuffer(bytes: codes, length: codes.count,
                                                options: .storageModeShared),
              let sBBuf = ctx.device.makeBuffer(bytes: scales, length: scales.count,
                                                options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.hidden) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        proj.encode(commandBuffer: cb,
                    o: oBuf, woA: wABuf,
                    woBWeights: wBBuf, woBScales: sBBuf,
                    out: outBuf)
        cb.commit(); cb.waitUntilCompleted()

        // Reference: grouped stage -> fp16 round-trip (the kernel's low-rank
        // scratch is fp16) -> FP8 GEMV reference.
        let low = Self.refGrouped(o: o16.map { Float($0) },
                                  woA: woA16.map { Self.bf16Decode($0) })
            .map { Float(Float16($0)) }
        let ref = V4Quantization.gemvFP8(matrix: woB, x: low)
        let actual = Fp16Buffer.read(outBuf, count: Self.hidden)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "full projection rel=\(rel)")
    }
}
