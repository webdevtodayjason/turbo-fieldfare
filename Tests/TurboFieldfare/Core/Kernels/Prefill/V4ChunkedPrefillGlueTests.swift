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
        let sp = x > 20 ? x : log(1 + exp(x))
        return sqrt(max(sp, 0))
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

    static func makeBF16(_ device: MTLDevice, _ values: [UInt16]) -> MTLBuffer? {
        values.withUnsafeBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: values.count * MemoryLayout<UInt16>.size,
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
            let lk = sqrtSoftplus(logits[lhs]) + bias[lhs]
            let rk = sqrtSoftplus(logits[rhs]) + bias[rhs]
            if lk == rk { return lhs < rhs }
            return lk > rk
        }.prefix(6).map(Int32.init)
        let scores = ids.map { sqrtSoftplus(logits[Int($0)]) }
        let sum = scores.reduce(0, +)
        let inv = 1 / sum
        return (ids, scores.map { $0 * inv * routeScale })
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
        // This fixture discriminates s(logit)+bias from the incorrect
        // s(logit+bias): expert 42 wins only under the reference formula.
        let row1 = 256
        logits[row1 + 1] = 0.0
        logits[row1 + 2] = -0.1
        logits[row1 + 3] = -0.2
        logits[row1 + 4] = -0.3
        logits[row1 + 5] = -0.4
        logits[row1 + 6] = -0.5
        logits[row1 + 42] = -1.0
        bias[42] = 1.0
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

    @Test func bf16GEMM_matchesSerialKernelBitExactlyAcrossRows() throws {
        let ctx = try MetalContext()
        let glue = try V4ChunkedPrefillGlue(device: ctx.device)
        let serial = try V4DecodeGlue(context: ctx)
        let rows = 3, experts = 17, dim = 128
        let weights = (0..<(experts * dim)).map { index in
            Self.bf16Encode(sin(Float(index) * 0.031) * 0.2 + cos(Float(index) * 0.007) * 0.05)
        }
        let hiddenRows = (0..<(rows * dim)).map { index in
            Float16(sin(Float(index) * 0.019) * 1.5 - cos(Float(index) * 0.043) * 0.25)
        }
        guard let weightsBuffer = Self.makeBF16(ctx.device, weights),
              let hiddenBuffer = Self.makeF16(ctx.device, hiddenRows),
              let batchedOut = ctx.device.makeBuffer(
                length: rows * experts * MemoryLayout<Float>.size,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        glue.encodeBF16GEMMSerialOrder(commandBuffer: cb,
                                       weights: weightsBuffer,
                                       hidden: hiddenBuffer,
                                       output: batchedOut,
                                       rows: rows,
                                       outputRows: experts,
                                       dim: dim)
        var serialOutputs: [MTLBuffer] = []
        for row in 0..<rows {
            let hidden = Array(hiddenRows[(row * dim)..<((row + 1) * dim)])
            guard let hiddenRow = Self.makeF16(ctx.device, hidden),
                  let out = ctx.device.makeBuffer(
                    length: experts * MemoryLayout<Float>.size,
                    options: .storageModeShared) else {
                Issue.record("alloc failed"); return
            }
            serial.encodeBF16GEMV(commandBuffer: cb,
                                  weights: weightsBuffer,
                                  x: hiddenRow,
                                  out: out,
                                  m: UInt32(experts),
                                  d: UInt32(dim))
            serialOutputs.append(out)
        }
        cb.commit(); cb.waitUntilCompleted()

        let got = Self.readF32(batchedOut, count: rows * experts)
        for row in 0..<rows {
            let expected = Self.readF32(serialOutputs[row], count: experts)
            for expert in 0..<experts {
                #expect(got[row * experts + expert].bitPattern == expected[expert].bitPattern,
                        "row=\(row) expert=\(expert) got=\(got[row * experts + expert]) expected=\(expected[expert])")
            }
        }
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

    @Test func hashRouterWeights_batchedSerialMatchesDecodeBitExactly() throws {
        let ctx = try MetalContext()
        let serial = try V4DecodeGlue(context: ctx)
        let rows = 4
        let routeScale: Float = 1.5
        let logits = (0..<(rows * 256)).map { index in
            sin(Float(index) * 0.071) * 2.75 + cos(Float(index) * 0.013) * 0.4
        }
        let ids: [Int32] = [0, 5, 42, 88, 128, 255,
                            7, 7, 8, 9, 10, 11,
                            200, 3, 199, 4, 198, 5,
                            17, 31, 63, 95, 127, 191]
        guard let logitsBuffer = Self.makeF32(ctx.device, logits),
              let idsBuffer = Self.makeI32(ctx.device, ids),
              let batchedOut = ctx.device.makeBuffer(
                length: rows * 6 * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        serial.encodeRouterWeightsAtIndicesBatchedSerial(
            commandBuffer: cb,
            logits: logitsBuffer,
            indices: idsBuffer,
            outWeights: batchedOut,
            rows: rows,
            expertCount: 256,
            k: 6,
            routeScale: routeScale)
        var scalarOutputs: [MTLBuffer] = []
        for row in 0..<rows {
            guard let rowLogits = Self.makeF32(
                    ctx.device, Array(logits[(row * 256)..<((row + 1) * 256)])),
                  let rowIDs = Self.makeI32(
                    ctx.device, Array(ids[(row * 6)..<((row + 1) * 6)])),
                  let rowOut = ctx.device.makeBuffer(
                    length: 6 * MemoryLayout<Float>.stride,
                    options: .storageModeShared) else {
                Issue.record("alloc failed"); return
            }
            serial.encodeRouterWeightsAtIndices(
                commandBuffer: cb,
                logits: rowLogits,
                indices: rowIDs,
                outWeights: rowOut,
                k: 6,
                routeScale: routeScale)
            scalarOutputs.append(rowOut)
        }
        cb.commit(); cb.waitUntilCompleted()

        let actual = Self.readF32(batchedOut, count: rows * 6)
        for row in 0..<rows {
            let expected = Self.readF32(scalarOutputs[row], count: 6)
            for rank in 0..<6 {
                #expect(actual[row * 6 + rank].bitPattern == expected[rank].bitPattern,
                        "row=\(row) rank=\(rank)")
            }
        }
    }

    @Test func top6Router_batchedSerialMatchesIndependentRowsBitExactly() throws {
        let ctx = try MetalContext()
        let moe = try MoEV4(context: ctx)
        let rows = 3
        let routeScale: Float = 1.5
        var logits = (0..<(rows * 256)).map { index in
            sin(Float(index) * 0.043) * 3.25 - cos(Float(index) * 0.017) * 0.3
        }
        for expert in 4...10 { logits[expert] = 2 }
        let prefix = 17
        var paddedBias = [Float](repeating: -999, count: prefix)
        paddedBias += (0..<256).map { index in cos(Float(index) * 0.051) * 0.08 }
        guard let logitsBuffer = Self.makeF32(ctx.device, logits),
              let biasBuffer = Self.makeF32(ctx.device, paddedBias),
              let batchedIDs = ctx.device.makeBuffer(
                length: rows * 6 * MemoryLayout<UInt32>.stride,
                options: .storageModeShared),
              let batchedWeights = ctx.device.makeBuffer(
                length: rows * 6 * MemoryLayout<Float>.stride,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        moe.encodeRouterSelectV4BatchedSerial(
            commandBuffer: cb,
            logits: logitsBuffer,
            bias: biasBuffer,
            biasOffset: prefix * MemoryLayout<Float>.stride,
            outIndices: batchedIDs,
            outWeights: batchedWeights,
            rows: rows,
            numExperts: 256,
            routeScale: routeScale)
        var scalarIDs: [MTLBuffer] = []
        var scalarWeights: [MTLBuffer] = []
        for row in 0..<rows {
            guard let rowLogits = Self.makeF32(
                    ctx.device, Array(logits[(row * 256)..<((row + 1) * 256)])),
                  let rowIDs = ctx.device.makeBuffer(
                    length: 6 * MemoryLayout<UInt32>.stride,
                    options: .storageModeShared),
                  let rowWeights = ctx.device.makeBuffer(
                    length: 6 * MemoryLayout<Float>.stride,
                    options: .storageModeShared) else {
                Issue.record("alloc failed"); return
            }
            moe.encodeRouterSelectV4BatchedSerial(
                commandBuffer: cb,
                logits: rowLogits,
                bias: biasBuffer,
                biasOffset: prefix * MemoryLayout<Float>.stride,
                outIndices: rowIDs,
                outWeights: rowWeights,
                rows: 1,
                numExperts: 256,
                routeScale: routeScale)
            scalarIDs.append(rowIDs)
            scalarWeights.append(rowWeights)
        }
        cb.commit(); cb.waitUntilCompleted()

        let actualIDs = Self.readI32(batchedIDs, count: rows * 6)
        let actualWeights = Self.readF32(batchedWeights, count: rows * 6)
        for row in 0..<rows {
            let expectedIDs = Self.readI32(scalarIDs[row], count: 6)
            let expectedWeights = Self.readF32(scalarWeights[row], count: 6)
            #expect(Array(actualIDs[(row * 6)..<((row + 1) * 6)]) == expectedIDs)
            for rank in 0..<6 {
                #expect(actualWeights[row * 6 + rank].bitPattern == expectedWeights[rank].bitPattern,
                        "row=\(row) rank=\(rank)")
            }
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
