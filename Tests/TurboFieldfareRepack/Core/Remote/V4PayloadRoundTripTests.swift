import Foundation
import Testing

@testable import TurboFieldfareRepackCore

/// V4F-01 gate: synthetic DeepSeek-V4 fixture -> repack -> byte-identical
/// digest round-trip on every resident tensor and every expert blob slice,
/// plus manifest/layout schema and MTP-drop checks. No real checkpoint, no
/// network beyond the in-process fake HF protocol.
@Suite(.serialized)
struct V4PayloadRoundTripTests {

    // MARK: - Full round trip

    @Test func v4RepackRoundTripsAllPayloadBytes() async throws {
        let snapshotDir = tmpDirForRemote("v4snap")
        let output = tmpPathForRemote("v4remote")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticV4Snapshot.build(at: snapshotDir,
                                                     seed: 0xB10B_5EED_0001)

        resetFakeHF()
        FakeHFURLProtocol.files = try v4RemoteFiles(snapshotDir: snapshotDir,
                                                    snap: snapshot)
        let audit = RepackAudit()
        let result = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output, session: fakeHFSession()),
            audit: audit
        ).run()

        #expect(result.dryRun == false)
        #expect(result.reusedBytes == 0)

        for relativePath in [
            "model_weights.bin",
            "packed_experts/layout.json",
            "packed_experts/layer_00.bin",
            "packed_experts/layer_01.bin",
            "manifest.json",
        ] {
            #expect(FileManager.default.fileExists(
                atPath: (output as NSString).appendingPathComponent(relativePath)))
        }

        // -- Resident file: parse the binary index, digest every entry's
        // weight bytes (and FP8 scale bytes) against the source payloads.
        let residentEntries = try readResidentIndex(
            path: (output as NSString).appendingPathComponent("model_weights.bin"))
        #expect(Set(residentEntries.keys) == Set(snapshot.residentNames))
        #expect(residentEntries.values.allSatisfy { !$0.name.hasPrefix("mtp.") })

        let residentData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("model_weights.bin")))
        for (name, entry) in residentEntries {
            let source = try #require(snapshot.tensors[name])
            let installed = residentData.subdata(in: entry.fileOffset..<(entry.fileOffset + entry.sizeBytes))
            #expect(sha256Hex(installed) == sha256Hex(Data(source.bytes)),
                    "resident weight digest mismatch: \(name)")
            if entry.dtype == 6 {  // F8_E4M3 pairs with an F8_E8M0 .scale
                let scaleName = String(name.dropLast(".weight".count)) + ".scale"
                let scaleSource = try #require(snapshot.tensors[scaleName])
                #expect(entry.scaleSize == scaleSource.bytes.count)
                #expect(entry.biasSize == 0)
                let installedScale = residentData.subdata(
                    in: entry.scaleOffset..<(entry.scaleOffset + entry.scaleSize))
                #expect(sha256Hex(installedScale) == sha256Hex(Data(scaleSource.bytes)),
                        "resident scale digest mismatch: \(scaleName)")
            } else {
                #expect(entry.scaleSize == 0)
                #expect(entry.biasSize == 0)
            }
        }
        // Embedding first, head last, per-layer groups in layer order.
        let orderedNames = residentEntries.values.sorted { $0.index < $1.index }.map(\.name)
        #expect(orderedNames.first == "embed.weight")
        #expect(orderedNames.last == "head.weight")
        let layerIndices = orderedNames.compactMap { name -> Int? in
            guard name.hasPrefix("layers.") else { return nil }
            return Int(name.dropFirst("layers.".count).prefix(while: { $0 != "." }))
        }
        #expect(layerIndices == layerIndices.sorted())

        // -- Expert blobs: digest all 6 slices of every expert in both layers.
        let layoutData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("packed_experts/layout.json")))
        let layout = try #require(
            try JSONSerialization.jsonObject(with: layoutData) as? [String: Any])
        let arch = SyntheticV4Snapshot.Arch()
        #expect(layout["expertsPerLayer"] as? Int == arch.numExperts)
        let expertStride = try #require((layout["expertStride"] as? NSNumber)?.uint64Value)
        #expect(expertStride % 16_384 == 0)

        let expectedBlobBytes = UInt64(
            3 * (arch.moeIntermediate * arch.hidden / 2)   // w1/w2/w3 FP4 weights
            + 3 * (arch.moeIntermediate * arch.hidden / 32) // per-32 ue8m0 scales
        )
        #expect(expertStride >= expectedBlobBytes)

        let layerObjs = try #require(layout["layers"] as? [[String: Any]])
        #expect(layerObjs.count == arch.numLayers)
        let roleKeys = ["gate", "gate_scales", "up", "up_scales", "down", "down_scales"]
        for layerObj in layerObjs {
            let layerIndex = try #require(layerObj["layer"] as? Int)
            let layerData = try Data(contentsOf: URL(fileURLWithPath:
                (output as NSString)
                    .appendingPathComponent("packed_experts/layer_\(String(format: "%02d", layerIndex)).bin")))
            #expect(UInt64(layerData.count) == UInt64(arch.numExperts) * expertStride)
            let experts = try #require(layerObj["experts"] as? [[String: Any]])
            #expect(experts.count == arch.numExperts)
            for expertObj in experts {
                let expert = try #require(expertObj["expert"] as? Int)
                let offset = try #require((expertObj["offset"] as? NSNumber)?.uint64Value)
                #expect(offset == UInt64(expert) * expertStride)
                let tensors = try #require(expertObj["tensors"] as? [String: Any])
                #expect(Set(tensors.keys) == Set(roleKeys))
                for (key, value) in tensors {
                    let t = try #require(value as? [String: Any])
                    let sliceOffset = try #require((t["offset"] as? NSNumber)?.uint64Value)
                    let sliceSize = try #require((t["size"] as? NSNumber)?.uint64Value)
                    #expect(sliceOffset % 4 == 0)
                    let role: String
                    let component: String
                    switch key {
                    case "gate", "up", "down":
                        role = key == "gate" ? "w1" : (key == "up" ? "w3" : "w2")
                        component = "weight"
                        #expect(t["dtype"] as? String == "I8")
                        #expect(t["bits"] as? Int == 4)
                    case "gate_scales", "up_scales", "down_scales":
                        role = key.hasPrefix("gate") ? "w1" : (key.hasPrefix("up") ? "w3" : "w2")
                        component = "scale"
                        #expect(t["dtype"] as? String == "F8_E8M0")
                    default:
                        throw RepackError.configurationInvalid(detail: "unexpected key \(key)")
                    }
                    let sourceName = "layers.\(layerIndex).ffn.experts.\(expert).\(role).\(component)"
                    let source = try #require(snapshot.tensors[sourceName])
                    #expect(sliceSize == UInt64(source.bytes.count))
                    let begin = Int(offset + sliceOffset)
                    let installed = layerData.subdata(in: begin..<(begin + Int(sliceSize)))
                    #expect(sha256Hex(installed) == sha256Hex(Data(source.bytes)),
                            "expert blob digest mismatch: \(sourceName)")
                }
            }
        }

        // -- MTP drops recorded, scratch bound respected (IO-10 bound).
        #expect(audit.tensorsDroppedMTP == snapshot.mtpTensorNames)
        #expect(audit.tensorsDroppedMultimodal == [])
        #expect(audit.largestScratchBytes <= BoundedScratch.defaultLimitBytes)
        #expect(audit.wholeFileHeapBuffers == false)
    }

    // MARK: - Manifest schema

    @Test func v4ManifestCarriesFamilySchema() async throws {
        let snapshotDir = tmpDirForRemote("v4snap-manifest")
        let output = tmpPathForRemote("v4remote-manifest")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticV4Snapshot.build(at: snapshotDir,
                                                     seed: 0xB10B_5EED_0002)

        resetFakeHF()
        FakeHFURLProtocol.files = try v4RemoteFiles(snapshotDir: snapshotDir,
                                                    snap: snapshot)
        _ = try await RemoteStreamingRepacker(
            options: remoteOptions(outputDir: output, session: fakeHFSession())
        ).run()

        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try #require(
            try JSONSerialization.jsonObject(with: manifestData) as? [String: Any])

        #expect(manifest["magic"] as? String == "GTURBO")
        #expect(manifest["versionMajor"] as? Int == 1)
        #expect(manifest["versionMinor"] as? Int == 1)
        #expect(manifest["modelFamily"] as? String == "deepseek-v4-flash")
        #expect(manifest["numLayers"] as? Int == 2)
        #expect(manifest["expertsPerLayer"] as? Int == 4)

        let arch = try #require(manifest["arch"] as? [String: Any])
        #expect(arch["num_hidden_layers"] as? Int == 2)
        #expect(arch["hidden_size"] as? Int == 128)
        #expect(arch["n_routed_experts"] as? Int == 4)
        #expect(arch["num_experts_per_tok"] as? Int == 2)
        #expect(arch["num_hash_layers"] as? Int == 1)
        #expect(arch["num_nextn_predict_layers"] as? Int == 1)
        #expect(arch["compress_ratios"] as? [Int] == [0, 4])
        #expect(arch["vocab_size"] as? Int == 256)

        let quant = try #require(manifest["quant"] as? [String: Any])
        let routed = try #require(quant["routedExpert"] as? [String: Any])
        #expect(routed["scheme"] as? String == "fp4-e2m1-e8m0")
        #expect(routed["weightBits"] as? Int == 4)
        #expect(routed["scaleType"] as? String == "F8_E8M0")
        #expect(routed["groupSize"] as? Int == 32)
        #expect(routed["biasType"] as? String == "none")
        let attention = try #require(quant["attention"] as? [String: Any])
        #expect(attention["scheme"] as? String == "fp8-blockwise-e4m3")
        #expect(attention["weightBits"] as? Int == 8)
        #expect(attention["blockRows"] as? Int == 128)
        #expect(attention["blockCols"] as? Int == 128)
        let embedding = try #require(quant["embedding"] as? [String: Any])
        #expect(embedding["scheme"] as? String == "none")
        #expect(embedding["weightBits"] as? Int == 16)
    }

    // MARK: - Real-geometry blob layout (no payloads, plan only)

    @Test func v4RealGeometryBlobLayoutMatchesSpec() throws {
        // Plan one layer at true V4-Flash geometry (43-layer arch values but a
        // 1-layer registry) and check the fixed-stride page-aligned blob.
        let hidden: UInt64 = 4096
        let moe: UInt64 = 2048
        let experts = 256
        var registry: [String: SourceTensor] = [:]
        var headers: [Safetensors.Header] = []
        var tensors: [SourceTensor] = []
        var cursor: UInt64 = 1_000_000
        func fake(_ name: String, _ dtype: SourceTensor.Dtype, _ shape: [UInt64]) {
            let size = shape.reduce(UInt64(1), *) * UInt64(dtype.elementBytes)
            let t = SourceTensor(name: name, shardPath: "shard-a", dtype: dtype,
                                 shape: shape, absoluteOffset: cursor, sizeBytes: size)
            cursor += size
            registry[name] = t
            tensors.append(t)
        }
        fake("embed.weight", .bf16, [129_280, hidden])
        fake("norm.weight", .bf16, [hidden])
        for e in 0..<experts {
            fake("layers.0.ffn.experts.\(e).w1.weight", .i8, [moe, hidden / 2])
            fake("layers.0.ffn.experts.\(e).w1.scale", .f8e8m0, [moe, hidden / 32])
            fake("layers.0.ffn.experts.\(e).w2.weight", .i8, [hidden, moe / 2])
            fake("layers.0.ffn.experts.\(e).w2.scale", .f8e8m0, [hidden, moe / 32])
            fake("layers.0.ffn.experts.\(e).w3.weight", .i8, [moe, hidden / 2])
            fake("layers.0.ffn.experts.\(e).w3.scale", .f8e8m0, [moe, hidden / 32])
        }
        headers.append(Safetensors.Header(tensors: tensors))

        let quant = DeepSeekQuantDescriptor(weightBlockRows: 128, weightBlockCols: 128,
                                            scaleFmt: "ue8m0", expertDtype: "fp4",
                                            expertScaleGroupK: 32)
        let meta = IndexLoader.SourceMetadata(
            indexPath: "index", configPath: "config",
            indexSha256Hex: String(repeating: "0", count: 64),
            weightMap: Dictionary(uniqueKeysWithValues: registry.keys.map { ($0, "shard-a") }),
            family: .deepseekV4Flash,
            baseBits: 8, baseGroupSize: 128,
            baseMode: DeepSeekQuantDescriptor.fp8SchemeName,
            bitsOverrides: [:], deepseekQuant: quant,
            shardFilenames: ["shard-a"])
        let arch = ArchInfoV4(
            numLayers: 1, hiddenSize: Int(hidden), vocabSize: 129_280,
            numExperts: experts, numSharedExperts: 1, topKExperts: 6,
            moeIntermediateSize: Int(moe), numHashLayers: 3, numMTPLayers: 1,
            numHeads: 64, numKVHeads: 1, headDim: 512, qLoraRank: 1024,
            qkRopeHeadDim: 64, oGroups: 8, oLoraRank: 1024,
            indexNHeads: 64, indexHeadDim: 128, indexTopk: 512,
            slidingWindow: 128, ropeTheta: 10_000, compressRopeTheta: 160_000,
            compressRatios: [0], routedScalingFactor: 1.5, swigluLimit: 10,
            normTopkProb: true, scoringFunc: "sqrtsoftplus", topkMethod: "noaux_tc",
            hiddenActivation: "silu", hcMult: 4, hcEps: 1e-6, hcSinkhornIters: 20,
            rmsNormEps: 1e-6, tieWordEmbeddings: false,
            maxPositionEmbeddings: 1_048_576,
            yarnFactor: 16, yarnOriginalMaxPositions: 65_536,
            yarnBetaFast: 32, yarnBetaSlow: 1)

        let plan = try RepackPlanner.plan(meta: meta, arch: .deepseekV4(arch),
                                          shardHeaders: headers, outputDir: "/tmp/v4geom")
        #expect(plan.family == .deepseekV4Flash)
        #expect(plan.layers.count == 1)
        let layer = plan.layers[0]
        #expect(layer.expertsPerLayer == experts)
        #expect(layer.subTensors.count == 6)

        // Per-expert blob: 3 x 4,194,304 FP4 bytes + 3 x 262,144 scale bytes
        // = 13,369,344, already an exact page multiple (~13.4 MB).
        let expectedBlob: UInt64 = 3 * 4_194_304 + 3 * 262_144
        #expect(expectedBlob == 13_369_344)
        #expect(layer.expertStride == expectedBlob)
        #expect(layer.expertStride % Layout.pageBytes == 0)
        #expect(layer.fileSize == UInt64(experts) * expectedBlob)

        var expectedOffset: UInt64 = 0
        let expectedRoles = ["gate", "up", "down"]
        for (index, slice) in layer.subTensors.enumerated() {
            #expect(slice.role == expectedRoles[index / 2])
            #expect(slice.component == (index % 2 == 0 ? "weights" : "scales"))
            #expect(slice.offsetInExpertBlob == expectedOffset)
            #expect(slice.offsetInExpertBlob % 4 == 0)
            #expect(slice.sizeInExpertBlob == (index % 2 == 0 ? 4_194_304 : 262_144))
            #expect(slice.perExpertSources?.count == experts)
            if slice.component == "weights" {
                #expect(slice.bitsForWeights == 4)
                #expect(slice.dtype == 4)  // I8 container
                #expect(slice.logicalShape == (slice.role == "down"
                                               ? [hidden, moe] : [moe, hidden]))
            } else {
                #expect(slice.bitsForWeights == nil)
                #expect(slice.dtype == 7)  // F8_E8M0
                #expect(slice.logicalShape == (slice.role == "down"
                                               ? [hidden, moe / 32] : [moe, hidden / 32]))
            }
            expectedOffset += slice.sizeInExpertBlob
        }
        // Per-expert source resolution: expert 17's w1 comes from its own
        // per-expert source tensor, not a stride.
        let w1 = layer.subTensors[0]
        #expect(w1.source(forExpert: 17).name == "layers.0.ffn.experts.17.w1.weight")
        #expect(w1.source(forExpert: 17).absoluteOffset
                    == registry["layers.0.ffn.experts.17.w1.weight"]?.absoluteOffset)
    }

    // MARK: - Metadata + arch parsing

    @Test func v4IndexLoaderAndArchParseDeepSeekConfig() throws {
        let snapshotDir = tmpDirForRemote("v4snap-meta")
        defer { cleanUpRemote([snapshotDir]) }
        _ = try SyntheticV4Snapshot.build(at: snapshotDir, seed: 0xB10B_5EED_0003)

        let meta = try IndexLoader.load(snapshotDir: snapshotDir)
        #expect(meta.family == .deepseekV4Flash)
        #expect(meta.baseMode == DeepSeekQuantDescriptor.fp8SchemeName)
        #expect(meta.baseGroupSize == 128)
        #expect(meta.deepseekQuant?.weightBlockRows == 128)
        #expect(meta.deepseekQuant?.weightBlockCols == 128)
        #expect(meta.deepseekQuant?.scaleFmt == "ue8m0")
        #expect(meta.deepseekQuant?.expertDtype == "fp4")
        #expect(meta.deepseekQuant?.expertScaleGroupK == 32)
        #expect(meta.shardFilenames == ["model-00001-of-00002.safetensors",
                                        "model-00002-of-00002.safetensors"])

        let arch = try PlanArch.load(
            configPath: (snapshotDir as NSString).appendingPathComponent("config.json"),
            family: meta.family)
        guard case .deepseekV4(let v4) = arch else {
            Issue.record("expected DeepSeek V4 arch, got \(arch.family)")
            return
        }
        #expect(v4.numLayers == 2)
        #expect(v4.hiddenSize == 128)
        #expect(v4.numExperts == 4)
        #expect(v4.numHashLayers == 1)
        #expect(v4.numMTPLayers == 1)
        #expect(v4.compressRatios == [0, 4])
        #expect(v4.yarnFactor == 16)
    }

    @Test func v4IndexLoaderRejectsMalformedQuantConfig() throws {
        let snapshotDir = tmpDirForRemote("v4snap-badquant")
        defer { cleanUpRemote([snapshotDir]) }
        _ = try SyntheticV4Snapshot.build(at: snapshotDir, seed: 0xB10B_5EED_0004)
        let configPath = (snapshotDir as NSString).appendingPathComponent("config.json")
        var config = try #require(JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: configPath))) as? [String: Any])
        config["quantization_config"] = ["quant_method": "fp8", "fmt": "e4m3"]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: configPath))

        #expect(throws: RepackError.self) {
            try IndexLoader.load(snapshotDir: snapshotDir)
        }
    }

    // MARK: - Helpers

    private struct ResidentIndexEntry {
        let index: Int
        let name: String
        let dtype: UInt8
        let fileOffset: Int
        let sizeBytes: Int
        let scaleOffset: Int
        let scaleSize: Int
        let biasOffset: Int
        let biasSize: Int
    }

    /// Decodes the 24-byte header + 72-byte entries + string table written by
    /// `GTurboBinary` at the start of `model_weights.bin`.
    private func readResidentIndex(path: String) throws -> [String: ResidentIndexEntry] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        func u64(_ at: Int) -> Int {
            var value: UInt64 = 0
            for i in 0..<8 { value |= UInt64(data[at + i]) << UInt64(i * 8) }
            return Int(value)
        }
        func u32(_ at: Int) -> Int {
            var value: UInt32 = 0
            for i in 0..<4 { value |= UInt32(data[at + i]) << UInt32(i * 8) }
            return Int(value)
        }
        let indexSize = u64(0)
        #expect(indexSize <= BoundedScratch.defaultLimitBytes)
        let entryCount = u64(16)
        var entries: [String: ResidentIndexEntry] = [:]
        for i in 0..<entryCount {
            let base = 24 + i * 72
            let nameOffset = u32(base)
            let nameLen = Int(data[base + 4]) | (Int(data[base + 5]) << 8)
            let nameData = data.subdata(in: nameOffset..<(nameOffset + nameLen))
            let name = String(decoding: nameData, as: UTF8.self)
            entries[name] = ResidentIndexEntry(
                index: i,
                name: name,
                dtype: data[base + 6],
                fileOffset: u64(base + 8),
                sizeBytes: u64(base + 16),
                scaleOffset: u64(base + 40),
                scaleSize: u64(base + 48),
                biasOffset: u64(base + 56),
                biasSize: u64(base + 64))
        }
        return entries
    }

    private func sha256Hex(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private func v4RemoteFiles(snapshotDir: String,
                               snap: SyntheticV4Snapshot.Snapshot) throws -> [String: Data] {
        var files: [String: Data] = [
            "config.json": try Data(contentsOf: URL(fileURLWithPath:
                (snapshotDir as NSString).appendingPathComponent("config.json"))),
            "model.safetensors.index.json": try Data(contentsOf: URL(fileURLWithPath:
                (snapshotDir as NSString).appendingPathComponent("model.safetensors.index.json"))),
            "tokenizer.json": remoteTokenizerJSON,
            "tokenizer_config.json": remoteTokenizerConfigJSON,
        ]
        for shardPath in snap.shardPaths {
            files[(shardPath as NSString).lastPathComponent] =
                try Data(contentsOf: URL(fileURLWithPath: shardPath))
        }
        return files
    }
}
