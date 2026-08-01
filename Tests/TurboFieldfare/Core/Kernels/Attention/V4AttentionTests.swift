import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// CPU reference for the merged V4 decode attention: softmax over the
/// concatenation of compressed entries and window tokens (K == V, MQA),
/// with the per-head sink folded into the denominator only. FP32, no
/// online merging — a different summation order from the kernel.
enum V4AttentionRef {
    static func apply(q: [Float],                 // [64, 512]
                      sparse: [[Float]],          // selected compressed entries [n][512]
                      window: [[Float]],          // window tokens [w][512]
                      sinks: [Float],             // [64]
                      scale: Float) -> [Float] {
        let heads = sinks.count
        let headDim = 512
        var out = [Float](repeating: 0, count: heads * headDim)
        for h in 0..<heads {
            let qBase = h * headDim
            var scores: [Float] = []
            scores.reserveCapacity(sparse.count + window.count)
            var values: [[Float]] = []
            values.reserveCapacity(sparse.count + window.count)
            for kv in sparse + window {
                var dot: Float = 0
                for d in 0..<headDim { dot += q[qBase + d] * kv[d] }
                scores.append(dot * scale)
                values.append(kv)
            }
            let m = scores.max() ?? 0
            var dsum = exp(sinks[h] - m)          // sink: denominator only
            for s in scores { dsum += exp(s - m) }
            for d in 0..<headDim {
                var acc: Float = 0
                for (i, s) in scores.enumerated() {
                    acc += exp(s - m) * values[i][d]
                }
                out[qBase + d] = acc / dsum
            }
        }
        return out
    }
}

/// V4F-03 milestone 2: ratio-0 sliding-window MQA decode path and the HCA
/// dense variant of the merged sparse kernel, both with per-head sinks,
/// against the CPU reference. Synthetic FP16 data; runs the real Metal
/// kernels, no model process.
@Suite struct V4AttentionTests {

    private static let heads = 64
    private static let headDim = 512

    private func makeFixture(tokenCount: Int, seed: UInt64)
        -> (q: [Float16], window: [Float16], windowLogical: [[Float]], sinks: [Float]) {
        var rng = SeedTree(seed).key("v4attn-t\(tokenCount)")
        let q = (0..<(Self.heads * Self.headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        // Fill the ring with tokenCount tokens (only the last min(128, n)
        // survive as logical window contents).
        let nWindow = min(128, tokenCount)
        var ring = [Float16](repeating: 0, count: 128 * Self.headDim)
        var logical: [[Float]] = []
        logical.reserveCapacity(nWindow)
        let firstLogical = tokenCount - nWindow
        for pos in firstLogical..<tokenCount {
            let token = (0..<Self.headDim).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
            let slot = pos % 128
            for d in 0..<Self.headDim { ring[slot * Self.headDim + d] = token[d] }
            logical.append(token.map { Float($0) })
        }
        let sinks = (0..<Self.heads).map { _ in rng.uniform(-1, 1) }
        return (q, ring, logical, sinks)
    }

    private func runWindowMQA(tokenCount: Int, seed: UInt64) throws -> Float {
        let fixture = makeFixture(tokenCount: tokenCount, seed: seed)
        let ctx = try MetalContext()
        let attn = try V4Attention(device: ctx.device, maxContext: 4096)

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: fixture.q),
              let wBuf = Fp16Buffer.make(ctx.device, halves: fixture.window),
              let sinkBuf = ctx.device.makeBuffer(
                bytes: fixture.sinks,
                length: fixture.sinks.count * MemoryLayout<Float>.size,
                options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim) else {
            Issue.record("alloc failed"); return 1
        }
        let cb = ctx.queue.makeCommandBuffer()!
        attn.encodeWindowMQADecode(commandBuffer: cb,
                                   q: qBuf,
                                   windowK: wBuf,
                                   tokenCount: tokenCount,
                                   sinks: sinkBuf,
                                   out: outBuf)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(outBuf, count: Self.heads * Self.headDim)
        let ref = V4AttentionRef.apply(q: fixture.q.map { Float($0) },
                                       sparse: [],
                                       window: fixture.windowLogical,
                                       sinks: fixture.sinks,
                                       scale: V4Attention.softmaxScale)
        return RelError.compute(actual: actual, reference: ref)
    }

    @Test func windowMQA_shortSequence_matchesReference() throws {
        let rel = try runWindowMQA(tokenCount: 5, seed: 0xE55)
        #expect(rel < Tolerance.fp16ChainedReduction, "rel=\(rel)")
    }

    @Test func windowMQA_fullWindowNoWrap_matchesReference() throws {
        let rel = try runWindowMQA(tokenCount: 128, seed: 0xE56)
        #expect(rel < Tolerance.fp16ChainedReduction, "rel=\(rel)")
    }

    @Test func windowMQA_ringWrapped_matchesReference() throws {
        let rel = try runWindowMQA(tokenCount: 300, seed: 0xE57)
        #expect(rel < Tolerance.fp16ChainedReduction, "rel=\(rel)")
    }

    /// The sink must enter the denominator only: with a huge sink logit
    /// the output is pulled toward zero (all probability mass on the
    /// value-less sink), and the sink must NOT shift the running max.
    @Test func windowMQA_largeSink_drivesOutputToZero() throws {
        var fixture = makeFixture(tokenCount: 40, seed: 0xE58)
        fixture.sinks = [Float](repeating: 30, count: Self.heads)
        let ctx = try MetalContext()
        let attn = try V4Attention(device: ctx.device, maxContext: 4096)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: fixture.q),
              let wBuf = Fp16Buffer.make(ctx.device, halves: fixture.window),
              let sinkBuf = ctx.device.makeBuffer(
                bytes: fixture.sinks,
                length: fixture.sinks.count * MemoryLayout<Float>.size,
                options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        attn.encodeWindowMQADecode(commandBuffer: cb,
                                   q: qBuf, windowK: wBuf, tokenCount: 40,
                                   sinks: sinkBuf, out: outBuf)
        cb.commit(); cb.waitUntilCompleted()
        let actual = Fp16Buffer.read(outBuf, count: Self.heads * Self.headDim)
        let maxAbs = actual.map { abs($0) }.max() ?? 0
        // QK logits are O(512 * 0.25 * 512^-0.5) ~ O(5); sink 30 dominates
        // by e^25, so outputs should be ~1e-11 of the values.
        #expect(maxAbs < 1e-6, "sink-dominated output maxAbs=\(maxAbs)")
    }

    // MARK: - HCA dense (merged sparse kernel, iota gather)

    @Test func hcaDense_sparsePlusWindow_matchesReference() throws {
        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: [128])
        let kv = try CompressedKVCacheManager(device: ctx.device,
                                              config: config,
                                              maxContext: 4096)
        let attn = try V4Attention(device: ctx.device, maxContext: 4096)

        var rng = SeedTree(0xE59).key("hca")
        // 5 completed groups + a 40-token window (tokenCount 680).
        let nGroups = 5
        var entries: [[Float]] = []
        for g in 0..<nGroups {
            let entry = (0..<Self.headDim).map { _ in rng.uniform(-1, 1) }
            kv.writeEntry(layer: 0, group: g, entry: entry)
            entries.append(CompressedKVCacheManager.dequantizeEntry(
                values: CompressedKVCacheManager.quantizeEntry(entry, config: config).values,
                scales: CompressedKVCacheManager.quantizeEntry(entry, config: config).scales,
                rope: CompressedKVCacheManager.quantizeEntry(entry, config: config).rope,
                config: config))
        }
        var windowLogical: [[Float]] = []
        var ring = [Float16](repeating: 0, count: 128 * Self.headDim)
        for i in 0..<40 {
            let token = (0..<Self.headDim).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
            for d in 0..<Self.headDim { ring[i * Self.headDim + d] = token[d] }
            windowLogical.append(token.map { Float($0) })
        }
        let q = (0..<(Self.heads * Self.headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let sinks = (0..<Self.heads).map { _ in rng.uniform(-1, 1) }

        let slot = kv.compressedSlot(layer: 0, group: 0)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
              let wBuf = Fp16Buffer.make(ctx.device, halves: ring),
              let sinkBuf = ctx.device.makeBuffer(
                bytes: sinks, length: sinks.count * MemoryLayout<Float>.size,
                options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim) else {
            Issue.record("alloc failed"); return
        }
        let tokenCount = 40   // window branch attends min(128, tokenCount)
        let cb = ctx.queue.makeCommandBuffer()!
        let nSparse = attn.encodeHCADecode(commandBuffer: cb,
                                           q: qBuf,
                                           nVisible: nGroups,
                                           compressedValues: slot.values.buffer,
                                           compressedScales: slot.scales.buffer,
                                           compressedRope: slot.rope.buffer,
                                           windowK: wBuf,
                                           tokenCount: tokenCount,
                                           sinks: sinkBuf,
                                           out: outBuf)
        cb.commit(); cb.waitUntilCompleted()
        #expect(nSparse == nGroups)

        let actual = Fp16Buffer.read(outBuf, count: Self.heads * Self.headDim)
        let ref = V4AttentionRef.apply(q: q.map { Float($0) },
                                       sparse: entries,
                                       window: windowLogical,
                                       sinks: sinks,
                                       scale: V4Attention.softmaxScale)
        let rel = RelError.compute(actual: actual, reference: ref)
        // FP8-compressed entries feed both sides identically (reference
        // uses the dequantized store), so this stays an FP16-level gate.
        #expect(rel < Tolerance.fp16ChainedReduction, "rel=\(rel)")
    }
}
