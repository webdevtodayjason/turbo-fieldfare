import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-03 milestone 3: CSA decode path — lightning indexer scores, the
/// top-512 selection RECALL GATE (the project's recorded top risk: a
/// selection mismatch is a discrete divergence no output-tolerance gate
/// absorbs), the CSA compressor kernel, and a CSA end-to-end decode
/// against the CPU reference chain.
@Suite struct V4CSAIndexerTests {

    private static let heads = 64
    private static let headDim = 512
    private static let indexDim = 128

    // MARK: - CPU references

    /// score[b] = sum_h w[h] * relu(q[h] . kv[b]) — ReLU-then-weighted-sum.
    private static func refIndexScores(q: [Float], kv: [Float],
                                       weights: [Float], nBlocks: Int) -> [Float] {
        var out = [Float](repeating: 0, count: nBlocks)
        for b in 0..<nBlocks {
            var acc: Float = 0
            for h in 0..<heads {
                var dot: Float = 0
                for d in 0..<indexDim {
                    dot += q[h * indexDim + d] * kv[b * indexDim + d]
                }
                acc += weights[h] * max(dot, 0)
            }
            out[b] = acc
        }
        return out
    }

    /// Selection with the kernel's exact ordering: score descending, ties
    /// toward the lower block index.
    private static func refTopK(_ scores: [Float], k: Int) -> [Int] {
        Array(scores.enumerated()
            .sorted { lhs, rhs in
                lhs.element != rhs.element
                    ? lhs.element > rhs.element
                    : lhs.offset < rhs.offset
            }
            .prefix(k)
            .map { $0.offset })
    }

    /// DeepSeek YaRN frequency for rope pair `i` (rope dim 64), mirroring
    /// `v4_yarn_freq` and the reference precompute_freqs_cis.
    private static func yarnFreq(_ i: Int,
                                 theta: Float = 160_000,
                                 factor: Float = 16,
                                 origSeqLen: Float = 65_536,
                                 betaFast: Float = 32,
                                 betaSlow: Float = 1) -> Float {
        let dim: Float = 64
        let freq = pow(theta, -2 * Float(i) / dim)
        let logBase = log(theta)
        let lowRot = dim * log(origSeqLen / (betaFast * 2 * .pi)) / (2 * logBase)
        let highRot = dim * log(origSeqLen / (betaSlow * 2 * .pi)) / (2 * logBase)
        let low = max(floor(lowRot), 0)
        var high = min(ceil(highRot), dim - 1)
        if high == low { high = low + 0.001 }
        let ramp = min(max((Float(i) - low) / (high - low), 0), 1)
        let smooth = 1 - ramp
        return (freq / factor) * (1 - smooth) + freq * smooth
    }

    /// CSA pooling reference (recon note #7): overlapped channel-split
    /// softmax pooling + RMSNorm + group-start partial RoPE. Returns the
    /// 512-dim entry BEFORE FP8 quantization.
    private static func refCompressGroup(prevKV: [Float], curKV: [Float],
                                         prevGate: [Float], curGate: [Float],
                                         ape: [Float], gamma: [Float],
                                         ropePosition: Int) -> [Float] {
        var x = [Float](repeating: 0, count: headDim)
        for d in 0..<headDim {
            var kv = [Float](repeating: 0, count: 8)
            var gate = [Float](repeating: 0, count: 8)
            for j in 0..<4 {
                kv[j] = prevKV[j * 1024 + d]
                gate[j] = prevGate[j * 1024 + d] + ape[j * 1024 + d]
                kv[4 + j] = curKV[j * 1024 + 512 + d]
                gate[4 + j] = curGate[j * 1024 + 512 + d] + ape[j * 1024 + 512 + d]
            }
            let mx = gate.max()!
            var sum: Float = 0
            for j in 0..<8 { gate[j] = exp(gate[j] - mx); sum += gate[j] }
            var acc: Float = 0
            for j in 0..<8 { acc += gate[j] * kv[j] }
            x[d] = acc / sum
        }
        let ms = x.reduce(0) { $0 + $1 * $1 } / Float(headDim)
        let inv = 1 / (ms + 1e-6).squareRoot()
        for d in 0..<headDim { x[d] = x[d] * inv * gamma[d] }
        // Partial RoPE on the trailing 64 dims (slice pairs (448+i, 480+i))
        // at the group-start position, compress theta + YaRN.
        for i in 0..<32 {
            let angle = Float(ropePosition) * yarnFreq(i)
            let cs = cos(angle), sn = sin(angle)
            let x0 = x[448 + i], x1 = x[448 + 32 + i]
            x[448 + i] = x0 * cs - x1 * sn
            x[448 + 32 + i] = x0 * sn + x1 * cs
        }
        return x
    }

    // MARK: - Indexer score kernel

    @Test func indexerScores_matchCPUReference() throws {
        let ctx = try MetalContext()
        let attn = try V4Attention(device: ctx.device, maxContext: 65536)
        let nBlocks = 700
        var rng = SeedTree(0xF66).key("idx-score")
        let q16 = (0..<(Self.heads * Self.indexDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let kv16 = (0..<(nBlocks * Self.indexDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let weights = (0..<Self.heads).map { _ in rng.uniform(-1, 1) }

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q16),
              let kvBuf = Fp16Buffer.make(ctx.device, halves: kv16),
              let wBuf = ctx.device.makeBuffer(bytes: weights,
                                               length: weights.count * 4,
                                               options: .storageModeShared),
              let sinkBuf = ctx.device.makeBuffer(length: 4, options: .storageModeShared),
              let winBuf = Fp16Buffer.make(ctx.device, count: 128 * Self.headDim),
              let mainQ = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim),
              let cv = ctx.device.makeBuffer(length: 1024, options: .storageModeShared),
              let cs = ctx.device.makeBuffer(length: 1024, options: .storageModeShared),
              let cr = ctx.device.makeBuffer(length: 4096, options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim) else {
            Issue.record("alloc failed"); return
        }
        // Drive the indexer through the CSA entry point with a 1-token
        // window so the full chain runs; scores land in the scratch buffer.
        let cb = ctx.queue.makeCommandBuffer()!
        attn.encodeCSADecode(commandBuffer: cb,
                             q: mainQ, indexQ: qBuf, indexKV: kvBuf,
                             indexWeights: wBuf, nVisible: nBlocks,
                             compressedValues: cv, compressedScales: cs,
                             compressedRope: cr, windowK: winBuf,
                             tokenCount: 1, sinks: sinkBuf, out: outBuf)
        cb.commit(); cb.waitUntilCompleted()

        var actual = [Float](repeating: 0, count: nBlocks)
        let ptr = attn.indexerScoresBuffer.contents()
        for b in 0..<nBlocks {
            actual[b] = ptr.load(fromByteOffset: b * 4, as: Float.self)
        }
        let ref = Self.refIndexScores(q: q16.map { Float($0) },
                                      kv: kv16.map { Float($0) },
                                      weights: weights, nBlocks: nBlocks)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "indexer scores rel=\(rel)")
    }

    // MARK: - Selection recall gate (top risk)

    /// Planted, exactly-representable scores: the dot of head 0 must equal
    /// block index `b` EXACTLY in fp32. A single fp16 dim cannot hold
    /// 20000 distinct values, so the rank is split across two dims:
    /// `kv[b].dim0 = b % 2048` (fp16-exact) and `kv[b].dim1 = (b/2048)*2048`
    /// (power-of-two multiple, fp16-exact), with q dims 0 and 1 set to 1.
    /// Kernel and CPU selection must then agree EXACTLY. Recall 1.0 is the
    /// gate; anything less is a discrete selection divergence.
    private func runRecallGate(nBlocks: Int, maxContext: Int, seed: UInt64) throws -> Float {
        let ctx = try MetalContext()
        let attn = try V4Attention(device: ctx.device, maxContext: maxContext)

        var q16 = [Float16](repeating: 0, count: Self.heads * Self.indexDim)
        q16[0] = 1
        q16[1] = 1
        var weights = [Float](repeating: 0, count: Self.heads)
        weights[0] = 1
        var kv16 = [Float16](repeating: 0, count: nBlocks * Self.indexDim)
        for b in 0..<nBlocks {
            kv16[b * Self.indexDim] = Float16(b % 2048)
            kv16[b * Self.indexDim + 1] = Float16((b / 2048) * 2048)
        }

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q16),
              let kvBuf = Fp16Buffer.make(ctx.device, halves: kv16),
              let wBuf = ctx.device.makeBuffer(bytes: weights,
                                               length: weights.count * 4,
                                               options: .storageModeShared),
              let sinkBuf = ctx.device.makeBuffer(length: 4, options: .storageModeShared),
              let winBuf = Fp16Buffer.make(ctx.device, count: 128 * Self.headDim),
              let mainQ = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim),
              let cv = ctx.device.makeBuffer(length: maxContext / 4 * 448, options: .storageModeShared),
              let cs = ctx.device.makeBuffer(length: maxContext / 4 * 8, options: .storageModeShared),
              let cr = ctx.device.makeBuffer(length: maxContext / 4 * 128, options: .storageModeShared),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim) else {
            Issue.record("alloc failed"); return 0
        }
        let cb = ctx.queue.makeCommandBuffer()!
        let nSparse = attn.encodeCSADecode(commandBuffer: cb,
                                           q: mainQ, indexQ: qBuf, indexKV: kvBuf,
                                           indexWeights: wBuf, nVisible: nBlocks,
                                           compressedValues: cv, compressedScales: cs,
                                           compressedRope: cr, windowK: winBuf,
                                           tokenCount: 1, sinks: sinkBuf, out: outBuf)
        cb.commit(); cb.waitUntilCompleted()

        let k = min(512, nBlocks)
        #expect(nSparse == k)
        var selected = Set<Int>()
        let gptr = attn.gatherListBuffer.contents()
        for i in 0..<k {
            selected.insert(Int(gptr.load(fromByteOffset: i * 4, as: Int32.self)))
        }
        let scores = (0..<nBlocks).map { Float($0) }
        let expected = Set(Self.refTopK(scores, k: k))
        return Float(selected.intersection(expected).count) / Float(k)
    }

    @Test func recallGate_singleChunk_isExact() throws {
        // 600 blocks: one bitonic chunk, k = 512.
        let recall = try runRecallGate(nBlocks: 600, maxContext: 4096, seed: 0xF67)
        #expect(recall == 1.0, "RECALL GATE FAILED: \(recall)")
    }

    @Test func recallGate_multiChunk_isExact() throws {
        // 3000 blocks: two chunks -> candidate merge -> final pass.
        let recall = try runRecallGate(nBlocks: 3000, maxContext: 16384, seed: 0xF68)
        #expect(recall == 1.0, "RECALL GATE FAILED: \(recall)")
    }

    @Test func recallGate_threePasses_isExact() throws {
        // 20000 blocks: three ping-pong passes before the final sort.
        let recall = try runRecallGate(nBlocks: 20000, maxContext: 131072, seed: 0xF69)
        #expect(recall == 1.0, "RECALL GATE FAILED: \(recall)")
    }

    @Test func recallGate_fewerBlocksThanK_selectsAll() throws {
        let recall = try runRecallGate(nBlocks: 37, maxContext: 4096, seed: 0xF6A)
        #expect(recall == 1.0, "RECALL GATE FAILED: \(recall)")
    }

    // MARK: - Compressor kernel

    @Test func csaCompressGroup_matchesCPUReference() throws {
        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: [4])
        let kv = try CompressedKVCacheManager(device: ctx.device,
                                              config: config, maxContext: 256)
        let attn = try V4Attention(device: ctx.device, maxContext: 256)

        var rng = SeedTree(0xF6B).key("compressor")
        let prevKV = (0..<(4 * 1024)).map { _ in rng.uniform(-1, 1) }
        let curKV = (0..<(4 * 1024)).map { _ in rng.uniform(-1, 1) }
        let prevGate = (0..<(4 * 1024)).map { _ in rng.uniform(-1, 1) }
        let curGate = (0..<(4 * 1024)).map { _ in rng.uniform(-1, 1) }
        let ape = (0..<(4 * 1024)).map { _ in rng.uniform(-0.25, 0.25) }
        let gamma = (0..<512).map { _ in rng.uniform(0.5, 1.5) }

        func f32buf(_ values: [Float]) -> MTLBuffer? {
            ctx.device.makeBuffer(bytes: values, length: values.count * 4,
                                  options: .storageModeShared)
        }
        guard let pk = f32buf(prevKV), let ck = f32buf(curKV),
              let pg = f32buf(prevGate), let cg = f32buf(curGate),
              let ab = f32buf(ape), let gb = f32buf(gamma) else {
            Issue.record("alloc failed"); return
        }

        let group = 7
        let slot = kv.compressedSlot(layer: 0, group: group)
        let cb = ctx.queue.makeCommandBuffer()!
        attn.encodeCSACompressGroup(commandBuffer: cb,
                                    prevKV: pk, curKV: ck,
                                    prevGate: pg, curGate: cg,
                                    ape: ab, gamma: gb,
                                    outValues: slot.values.buffer,
                                    valuesOffset: slot.values.offset,
                                    outScales: slot.scales.buffer,
                                    scalesOffset: slot.scales.offset,
                                    outRope: slot.rope.buffer,
                                    ropeOffset: slot.rope.offset,
                                    ropePosition: UInt32(kv.ropePosition(layer: 0, group: group)))
        cb.commit(); cb.waitUntilCompleted()

        let actual = kv.readEntry(layer: 0, group: group)
        let ref = Self.refCompressGroup(prevKV: prevKV, curKV: curKV,
                                        prevGate: prevGate, curGate: curGate,
                                        ape: ape, gamma: gamma,
                                        ropePosition: kv.ropePosition(layer: 0, group: group))
        // Rope dims: FP16 storage of the same fp32 math — near-exact.
        for d in 448..<512 {
            let err = abs(actual[d] - ref[d])
            #expect(err < 1e-2, "rope dim \(d) err \(err)")
        }
        // Non-rope dims: compare against the CPU reference put through the
        // same quantizer; allow 2 FP8 ulps of slack for fast::exp and
        // summation-order differences at rounding boundaries.
        let qref = CompressedKVCacheManager.dequantizeEntry(
            values: CompressedKVCacheManager.quantizeEntry(ref, config: config).values,
            scales: CompressedKVCacheManager.quantizeEntry(ref, config: config).scales,
            rope: CompressedKVCacheManager.quantizeEntry(ref, config: config).rope,
            config: config)
        let rel = RelError.compute(actual: Array(actual[0..<448]),
                                   reference: Array(qref[0..<448]))
        #expect(rel < 8e-2, "compressed entry rel=\(rel)")
    }

    // MARK: - CSA end-to-end

    @Test func csaDecode_endToEnd_matchesCPUReferenceChain() throws {
        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: [4])
        let maxContext = 8192
        let kv = try CompressedKVCacheManager(device: ctx.device,
                                              config: config, maxContext: maxContext)
        let attn = try V4Attention(device: ctx.device, maxContext: maxContext)

        var rng = SeedTree(0xF6C).key("csa-e2e")
        // tokenCount 4000: window [3872, 4000), completed 1000 groups,
        // visible = 3872/4 = 968 > 512: the top-k genuinely truncates.
        let tokenCount = 4000
        let windowStart = kv.windowRange(tokenCount: tokenCount).lowerBound
        let nVisible = kv.visibleGroupCount(layer: 0, windowStart: windowStart,
                                            tokenCount: tokenCount)
        #expect(nVisible == 968)

        var entries: [[Float]] = []
        for g in 0..<nVisible {
            let entry = (0..<Self.headDim).map { _ in rng.uniform(-1, 1) }
            kv.writeEntry(layer: 0, group: g, entry: entry)
            let q = CompressedKVCacheManager.quantizeEntry(entry, config: config)
            entries.append(CompressedKVCacheManager.dequantizeEntry(
                values: q.values, scales: q.scales, rope: q.rope, config: config))
        }
        // Indexer cache contents (synthetic: unrelated to entries, which is
        // fine — the CPU chain uses the same buffers).
        let idxKV16 = (0..<(nVisible * Self.indexDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        kv.indexerBuffer(layer: 0).contents()
            .copyMemory(from: idxKV16, byteCount: idxKV16.count * 2)

        let idxQ16 = (0..<(Self.heads * Self.indexDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let weights = (0..<Self.heads).map { _ in rng.uniform(-1, 1) }
        let q16 = (0..<(Self.heads * Self.headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let sinks = (0..<Self.heads).map { _ in rng.uniform(-1, 1) }
        var ring = [Float16](repeating: 0, count: 128 * Self.headDim)
        var windowLogical: [[Float]] = []
        for i in 0..<128 {
            let token = (0..<Self.headDim).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
            for d in 0..<Self.headDim { ring[i * Self.headDim + d] = token[d] }
            windowLogical.append(token.map { Float($0) })
        }

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q16),
              let iqBuf = Fp16Buffer.make(ctx.device, halves: idxQ16),
              let wBuf = ctx.device.makeBuffer(bytes: weights,
                                               length: weights.count * 4,
                                               options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: sinks,
                                               length: sinks.count * 4,
                                               options: .storageModeShared),
              let winBuf = Fp16Buffer.make(ctx.device, halves: ring),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.heads * Self.headDim) else {
            Issue.record("alloc failed"); return
        }

        let slot = kv.compressedSlot(layer: 0, group: 0)
        let cb = ctx.queue.makeCommandBuffer()!
        let nSparse = attn.encodeCSADecode(commandBuffer: cb,
                                           q: qBuf,
                                           indexQ: iqBuf,
                                           indexKV: kv.indexerBuffer(layer: 0),
                                           indexWeights: wBuf,
                                           nVisible: nVisible,
                                           compressedValues: slot.values.buffer,
                                           compressedScales: slot.scales.buffer,
                                           compressedRope: slot.rope.buffer,
                                           windowK: winBuf,
                                           tokenCount: tokenCount,
                                           sinks: sBuf,
                                           out: outBuf)
        cb.commit(); cb.waitUntilCompleted()
        #expect(nSparse == 512)

        // CPU chain: same scores, same top-512, same merged softmax.
        let scores = Self.refIndexScores(q: idxQ16.map { Float($0) },
                                         kv: idxKV16.map { Float($0) },
                                         weights: weights, nBlocks: nVisible)
        let selected = Self.refTopK(scores, k: 512)
        let sparseVecs = selected.map { entries[$0] }
        let actual = Fp16Buffer.read(outBuf, count: Self.heads * Self.headDim)
        let ref = V4AttentionRef.apply(q: q16.map { Float($0) },
                                       sparse: sparseVecs,
                                       window: windowLogical,
                                       sinks: sinks,
                                       scale: V4Attention.softmaxScale)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction, "CSA end-to-end rel=\(rel)")
    }
}
