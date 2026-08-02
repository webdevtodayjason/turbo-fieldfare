import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct V4ChunkedPrefillExecutorTests {
    private static let heads = V4Attention.numQHeads
    private static let dim = V4Attention.headDim

    private func objects(_ config: V4CacheConfig, maxContext: Int = 512) throws -> (MetalContext, CompressedKVCacheManager, V4ChunkedPrefillExecutor) {
        let ctx = try MetalContext()
        let cache = try CompressedKVCacheManager(device: ctx.device, config: config, maxContext: maxContext)
        let attn = try V4Attention(device: ctx.device, maxContext: maxContext)
        return (ctx, cache, try V4ChunkedPrefillExecutor(device: ctx.device, cache: cache, attention: attn))
    }

    @Test func passthroughChunk_matchesPerRowCausalReference() throws {
        let rows = 6
        let (ctx, cache, exec) = try objects(V4CacheConfig(compressRatios: [0]))
        var rng = SeedTree(0xC001).key("pass")
        let q = (0..<(rows * Self.heads * Self.dim)).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
        let kv = (0..<(rows * Self.dim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let sinks = (0..<Self.heads).map { _ in rng.uniform(-0.75, 0.75) }
        let input = inputs(ctx.device, rows: rows, q: q, kv: kv, sinks: sinks)
        let cb = ctx.queue.makeCommandBuffer()!
        exec.encode(commandBuffer: cb, layer: 0, startPosition: 0, rowCount: rows, inputs: input)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed); #expect(cache.position == 0)
        cache.advance(by: rows)
        #expect(cache.position == rows)
        let actual = Fp16Buffer.read(input.output, count: rows * Self.heads * Self.dim)
        for row in 0..<rows {
            let qBase = row * Self.heads * Self.dim
            let qRow = Array(q[qBase..<(qBase + Self.heads * Self.dim)]).map { Float($0) }
            let win = (0...row).map { r in Array(kv[(r * Self.dim)..<((r + 1) * Self.dim)]).map { Float($0) } }
            let ref = V4AttentionRef.apply(q: qRow, sparse: [], window: win, sinks: sinks, scale: V4Attention.softmaxScale)
            let a = Array(actual[qBase..<(qBase + Self.heads * Self.dim)])
            #expect(RelError.compute(actual: a, reference: ref) < Tolerance.fp16ChainedReduction)
        }
    }

    @Test func commandBufferSequencing_rowsSeeEarlierQueuedWrites() throws {
        let rows = 3
        let (ctx, _, exec) = try objects(V4CacheConfig(compressRatios: [0]))
        var q = [Float16](repeating: 0, count: rows * Self.heads * Self.dim)
        var kv = [Float16](repeating: 0, count: rows * Self.dim)
        for row in 0..<rows { for h in 0..<Self.heads { q[row * Self.heads * Self.dim + h * Self.dim] = 1 }; kv[row * Self.dim] = Float16(row + 1) }
        let input = inputs(ctx.device, rows: rows, q: q, kv: kv, sinks: [Float](repeating: -20, count: Self.heads))
        let cb = ctx.queue.makeCommandBuffer()!
        exec.encode(commandBuffer: cb, layer: 0, startPosition: 0, rowCount: rows, inputs: input)
        cb.commit(); cb.waitUntilCompleted()
        let out = Fp16Buffer.read(input.output, count: rows * Self.heads * Self.dim)
        #expect(out[0] < out[2 * Self.heads * Self.dim])
    }

    @Test func twoLayerExecutorsShareStartPosition_beforeExternalAdvance() throws {
        let rows = 3
        let ctx = try MetalContext()
        let cache = try CompressedKVCacheManager(device: ctx.device,
                                                 config: V4CacheConfig(compressRatios: [0, 0]),
                                                 maxContext: 128)
        let attn = try V4Attention(device: ctx.device, maxContext: 128)
        let layer0 = try V4ChunkedPrefillExecutor(device: ctx.device, cache: cache, attention: attn)
        let layer1 = try V4ChunkedPrefillExecutor(device: ctx.device, cache: cache, attention: attn)
        var q = [Float16](repeating: 0, count: rows * Self.heads * Self.dim)
        for row in 0..<rows { for h in 0..<Self.heads { q[row * Self.heads * Self.dim + h * Self.dim] = 1 } }
        let kv = [Float16](repeating: 0.125, count: rows * Self.dim)
        let sinks = [Float](repeating: -10, count: Self.heads)
        let input0 = inputs(ctx.device, rows: rows, q: q, kv: kv, sinks: sinks)
        let input1 = inputs(ctx.device, rows: rows, q: q, kv: kv, sinks: sinks)
        let cb = ctx.queue.makeCommandBuffer()!
        layer0.encode(commandBuffer: cb, layer: 0, startPosition: 0, rowCount: rows, inputs: input0)
        layer1.encode(commandBuffer: cb, layer: 1, startPosition: 0, rowCount: rows, inputs: input1)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed)
        #expect(cache.position == 0)
        cache.advance(by: rows)
        #expect(cache.position == rows)
    }

    @Test func csaGroupCompletionCarryAndDisjointness() throws {
        let (ctx, cache, exec) = try objects(V4CacheConfig(compressRatios: [4]), maxContext: 256)
        let c = compressedInputs(ctx.device, rows: 8, kind: .csa)
        let cb = ctx.queue.makeCommandBuffer()!
        exec.encode(commandBuffer: cb, layer: 0, startPosition: 0, rowCount: 8, inputs: c.inputs, compressorWeights: c.weights, rope: .passthroughLayer)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed); #expect(cache.position == 0)
        #expect(cache.completedGroupCount(layer: 0, tokenCount: 8) == 2)
        #expect(cache.visibleGroupCount(layer: 0, windowStart: 4, tokenCount: 8) == 1)
        cache.assertDisjointCoverage(layer: 0, groupCount: 0, tokenCount: 8)
    }

    @Test func hcaGroupCompletionFlushesOneEntry() throws {
        let (ctx, cache, exec) = try objects(V4CacheConfig(compressRatios: [128]), maxContext: 256)
        let c = compressedInputs(ctx.device, rows: 128, kind: .hca)
        let cb = ctx.queue.makeCommandBuffer()!
        exec.encode(commandBuffer: cb, layer: 0, startPosition: 0, rowCount: 128, inputs: c.inputs, compressorWeights: c.weights, rope: .passthroughLayer)
        cb.commit(); cb.waitUntilCompleted()
        #expect(cb.status == .completed); #expect(cache.position == 0)
        #expect(cache.completedGroupCount(layer: 0, tokenCount: 128) == 1)
    }

    @Test func rowCausality_windowDoesNotReadFutureRows() throws {
        let (ctx, _, exec) = try objects(V4CacheConfig(compressRatios: [0]))
        var q = [Float16](repeating: 0, count: 2 * Self.heads * Self.dim)
        for h in 0..<Self.heads { q[h * Self.dim] = 1 }
        var kv = [Float16](repeating: 0, count: 2 * Self.dim); kv[0] = 1; kv[Self.dim] = 100
        let input = inputs(ctx.device, rows: 2, q: q, kv: kv, sinks: [Float](repeating: -20, count: Self.heads))
        let cb = ctx.queue.makeCommandBuffer()!
        exec.encode(commandBuffer: cb, layer: 0, startPosition: 0, rowCount: 2, inputs: input)
        cb.commit(); cb.waitUntilCompleted()
        #expect(Fp16Buffer.read(input.output, count: 2 * Self.heads * Self.dim)[0] < 2)
    }

    private func inputs(_ device: MTLDevice, rows: Int, q: [Float16], kv: [Float16], sinks: [Float]) -> V4ChunkedPrefillExecutor.Inputs {
        V4ChunkedPrefillExecutor.Inputs(q: Fp16Buffer.make(device, halves: q)!, windowKV: Fp16Buffer.make(device, halves: kv)!, sinks: device.makeBuffer(bytes: sinks, length: sinks.count * 4, options: .storageModeShared)!, output: Fp16Buffer.make(device, count: rows * Self.heads * Self.dim)!)
    }

    private func compressedInputs(_ device: MTLDevice, rows: Int, kind: V4LayerKind) -> (inputs: V4ChunkedPrefillExecutor.Inputs, weights: V4ChunkedPrefillExecutor.CompressorWeights) {
        let compDim = kind == .csa ? 1024 : 512
        let q = Fp16Buffer.make(device, count: rows * Self.heads * Self.dim)!
        let kv = Fp16Buffer.make(device, count: rows * Self.dim)!
        let sinks = zeroBuffer(device, Self.heads * 4)
        let out = Fp16Buffer.make(device, count: rows * Self.heads * Self.dim)!
        let comp = zeroBuffer(device, rows * compDim * 4)
        let gate = zeroBuffer(device, rows * compDim * 4)
        let idxComp = zeroBuffer(device, rows * 256 * 4)
        let idxGate = zeroBuffer(device, rows * 256 * 4)
        let idxQ = Fp16Buffer.make(device, count: rows * Self.heads * V4Attention.indexHeadDim)!
        let idxW = f32Buffer(device, [Float](repeating: 1, count: rows * Self.heads))
        let ape = zeroBuffer(device, (kind == .csa ? 4 : 128) * compDim * 4)
        let gamma = zeroBuffer(device, Self.dim * 4)
        gamma.contents().bindMemory(to: Float.self, capacity: Self.dim).initialize(repeating: 1, count: Self.dim)
        let idxAPE = zeroBuffer(device, 4 * 128 * 4)
        let idxGamma = zeroBuffer(device, 128 * 4)
        idxGamma.contents().bindMemory(to: Float.self, capacity: 128).initialize(repeating: 1, count: 128)
        return (V4ChunkedPrefillExecutor.Inputs(q: q, windowKV: kv, indexQ: kind == .csa ? idxQ : nil, indexWeights: kind == .csa ? idxW : nil, compressorKV: comp, compressorGate: gate, indexerCompressorKV: kind == .csa ? idxComp : nil, indexerCompressorGate: kind == .csa ? idxGate : nil, sinks: sinks, output: out, rowStrideCompressor: compDim * 4),
                V4ChunkedPrefillExecutor.CompressorWeights(ape: ape, gamma: gamma, indexerAPE: kind == .csa ? idxAPE : nil, indexerGamma: kind == .csa ? idxGamma : nil))
    }

    private func zeroBuffer(_ device: MTLDevice, _ bytes: Int) -> MTLBuffer {
        let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
        memset(buffer.contents(), 0, bytes)
        return buffer
    }

    private func f32Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer {
        device.makeBuffer(bytes: values, length: values.count * 4, options: .storageModeShared)!
    }
}
