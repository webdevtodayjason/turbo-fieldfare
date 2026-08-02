import Testing
import Foundation
import Metal
@testable import TurboFieldfare

@Suite struct V4ChunkedPrefillGlueTests {
    static func bf16Encode(_ v: Float) -> UInt16 {
        let b = v.bitPattern
        let rounded = (b &+ 0x7FFF &+ ((b >> 16) & 1)) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    static func bf16Decode(_ e: UInt16) -> Float {
        Float(bitPattern: UInt32(e) << 16)
    }

    static func sqrtSoftplus(_ x: Float) -> Float {
        let sp = max(x, 0) + log1p(exp(-abs(x)))
        return sqrt(sp)
    }

    static func makeF32(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: values.count * MemoryLayout<Float>.size,
                              options: .storageModeShared)
        }
    }

    static func makeI32(_ device: MTLDevice, _ values: [Int32]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: values.count * MemoryLayout<Int32>.size,
                              options: .storageModeShared)
        }
    }

    static func readF32(_ buf: MTLBuffer, count: Int) -> [Float] {
        let base = buf.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    static func makeF16(_ device: MTLDevice, _ values: [Float16]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: values.count * MemoryLayout<Float16>.size,
                              options: .storageModeShared)
        }
    }

    static func readF16(_ buf: MTLBuffer, count: Int) -> [Float16] {
        let base = buf.contents().bindMemory(to: Float16.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    static func readI32(_ buf: MTLBuffer, count: Int) -> [Int32] {
        let base = buf.contents().bindMemory(to: Int32.self, capacity: count)
        return Array(UnsafeBufferPointer(start: base, count: count))
    }

    static func refTop6(logits: [Float], bias: [Float], routeScale: Float) -> ([Int32], [Float]) {
        let ids = (0..<256).sorted { lhs, rhs in
            let lk = sqrtSoftplus(logits[lhs] + bias[lhs])
            let rk = sqrtSoftplus(logits[rhs] + bias[rhs])
            if lk == rk { return lhs < rhs }
            return lk > rk
        }.prefix(6).map(Int32.init)
        let scores = ids.map { sqrtSoftplus(logits[Int($0)]) }
        let sum = scores.reduce(0, +)
        return (ids, scores.map { $0 * routeScale / sum })
    }

    @Test func bf16EmbeddingGatherBroadcast_exactAcrossRowsAndStreams() throws {
        let ctx = try MetalContext()
        let glue = try V4ChunkedPrefillGlue(device: ctx.device)
        let vocab = 7, dim = 13, rows = 4
        let embeddingsF = (0..<(vocab * dim)).map { Float($0 - 40) / 17 }
        let embeddingsBF16 = embeddingsF.map(Self.bf16Encode)
        let tokenIDs: [Int32] = [3, 0, 6, 2]
        guard let embBuf = ctx.device.makeBuffer(bytes: embeddingsBF16,
                                                 length: embeddingsBF16.count * 2,
                                                 options: .storageModeShared),
              let idBuf = Self.makeI32(ctx.device, tokenIDs),
              let outBuf = ctx.device.makeBuffer(length: rows * 4 * dim * MemoryLayout<Float>.size,
                                                 options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        glue.encodeBF16EmbeddingGatherBroadcast(commandBuffer: cb,
                                                embeddings: embBuf,
                                                tokenIDs: idBuf,
                                                out: outBuf,
                                                rows: rows,
                                                dim: dim)
        cb.commit(); cb.waitUntilCompleted()

        let got = Self.readF32(outBuf, count: rows * 4 * dim)
        for row in 0..<rows {
            for stream in 0..<4 {
                for d in 0..<dim {
                    let expected = Self.bf16Decode(embeddingsBF16[Int(tokenIDs[row]) * dim + d])
                    let actual = got[(row * 4 + stream) * dim + d]
                    #expect(actual == expected, "row=\(row) stream=\(stream) d=\(d)")
                }
            }
        }
    }

    @Test func routerTop6_matchesCPUAcrossRowsBiasAndLowerIndexTies() throws {
        let ctx = try MetalContext()
        let glue = try V4ChunkedPrefillGlue(device: ctx.device)
        let rows = 3
        let routeScale: Float = 1.75
        var logits = [Float](repeating: -8, count: rows * 256)
        var bias = [Float](repeating: 0, count: 256)

        // Row 0: exact equal keys ensure deterministic lower-index ties.
        for eid in [4, 5, 6, 7, 8, 9, 10] { logits[eid] = 2 }
        // Row 1: static bias affects selection only. Expert 42 wins selection
        // while its output weight still comes from un-biased logits[42].
        let row1 = 256
        logits[row1 + 1] = 4.0
        logits[row1 + 2] = 3.5
        logits[row1 + 3] = 3.0
        logits[row1 + 4] = 2.5
        logits[row1 + 5] = 2.0
        logits[row1 + 6] = 1.5
        logits[row1 + 42] = -1.0
        bias[42] = 10.0
        // Row 2: scattered winners exercise token-major row layout.
        let row2 = 2 * 256
        for (eid, value) in [(255, 7), (128, 6), (17, 5), (88, 4), (3, 3), (200, 2), (5, 1)] {
            logits[row2 + eid] = Float(value)
        }

        guard let logitsBuf = Self.makeF32(ctx.device, logits),
              let biasBuf = Self.makeF32(ctx.device, bias),
              let idsBuf = ctx.device.makeBuffer(length: rows * 6 * MemoryLayout<Int32>.size,
                                                 options: .storageModeShared),
              let weightsBuf = ctx.device.makeBuffer(length: rows * 6 * MemoryLayout<Float>.size,
                                                     options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        glue.encodeRouterTop6(commandBuffer: cb,
                              logits: logitsBuf,
                              staticBias: biasBuf,
                              outIDs: idsBuf,
                              outWeights: weightsBuf,
                              rows: rows,
                              routeScale: routeScale)
        cb.commit(); cb.waitUntilCompleted()

        let gotIDs = Self.readI32(idsBuf, count: rows * 6)
        let gotWeights = Self.readF32(weightsBuf, count: rows * 6)
        for row in 0..<rows {
            let (refIDs, refWeights) = Self.refTop6(logits: Array(logits[(row * 256)..<((row + 1) * 256)]),
                                                   bias: bias,
                                                   routeScale: routeScale)
            #expect(Array(gotIDs[(row * 6)..<((row + 1) * 6)]) == refIDs)
            let gotSum = gotWeights[(row * 6)..<((row + 1) * 6)].reduce(0, +)
            #expect(abs(gotSum - routeScale) < 1e-5)
            for i in 0..<6 {
                #expect(abs(gotWeights[row * 6 + i] - refWeights[i]) < 1e-5,
                        "row=\(row) i=\(i) got=\(gotWeights[row * 6 + i]) ref=\(refWeights[i])")
            }
        }
        #expect(Array(gotIDs[0..<6]) == [4, 5, 6, 7, 8, 9])
        #expect(gotIDs[6] == 42)
    }

    @Test func hashRouterWeights_gathersProvidedIDsAndNormalizes() throws {
        let ctx = try MetalContext()
        let glue = try V4ChunkedPrefillGlue(device: ctx.device)
        let rows = 3
        let routeScale: Float = 2.25
        let logits = (0..<(rows * 256)).map { i in sin(Float(i) * 0.17) * 3 - 0.5 }
        let ids: [Int32] = [0, 5, 42, 88, 128, 255,
                            7, 7, 8, 9, 10, 11,
                            200, 3, 199, 4, 198, 5]
        guard let logitsBuf = Self.makeF32(ctx.device, logits),
              let idsBuf = Self.makeI32(ctx.device, ids),
              let weightsBuf = ctx.device.makeBuffer(length: rows * 6 * MemoryLayout<Float>.size,
                                                     options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        glue.encodeHashRouterWeights(commandBuffer: cb,
                                     logits: logitsBuf,
                                     tid2eid: idsBuf,
                                     outWeights: weightsBuf,
                                     rows: rows,
                                     routeScale: routeScale)
        cb.commit(); cb.waitUntilCompleted()

        let got = Self.readF32(weightsBuf, count: rows * 6)
        for row in 0..<rows {
            var scores: [Float] = []
            for i in 0..<6 { scores.append(Self.sqrtSoftplus(logits[row * 256 + Int(ids[row * 6 + i])])) }
            let sum = scores.reduce(0, +)
            for i in 0..<6 {
                let ref = scores[i] * routeScale / sum
                #expect(abs(got[row * 6 + i] - ref) < 1e-5,
                        "row=\(row) i=\(i) got=\(got[row * 6 + i]) ref=\(ref)")
            }
            #expect(abs(got[(row * 6)..<((row + 1) * 6)].reduce(0, +) - routeScale) < 1e-5)
        }
    }

    @Test func addF16_addsElementwiseForArbitraryCount() throws {
        let ctx = try MetalContext()
        let glue = try V4ChunkedPrefillGlue(device: ctx.device)
        let lhs = (0..<333).map { Float16(Float($0 % 17) * 0.25 - 2.0) }
        let rhs = (0..<333).map { Float16(3.0 - Float($0 % 11) * 0.125) }
        guard let lhsBuf = Self.makeF16(ctx.device, lhs),
              let rhsBuf = Self.makeF16(ctx.device, rhs),
              let outBuf = ctx.device.makeBuffer(length: lhs.count * MemoryLayout<Float16>.size,
                                                 options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        glue.encodeAddF16(commandBuffer: cb,
                          a: lhsBuf,
                          b: rhsBuf,
                          out: outBuf,
                          count: lhs.count)
        cb.commit(); cb.waitUntilCompleted()

        let got = Self.readF16(outBuf, count: lhs.count)
        for i in lhs.indices {
            #expect(got[i] == lhs[i] + rhs[i], "i=\(i) got=\(got[i])")
        }
    }
}
