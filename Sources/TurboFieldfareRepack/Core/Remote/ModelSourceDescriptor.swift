import Foundation

/// Model family identifier. Drives the repack planner fork, the manifest
/// schema variant, and (later) the runtime loader's validation branch.
public enum ModelFamily: String, Sendable, Hashable, CaseIterable {
    case gemma4 = "gemma4"
    case deepseekV4Flash = "deepseek-v4-flash"
}

/// Pinned description of one supported upstream checkpoint. Adding a model
/// means adding one instance here plus its weight-index SHA-256 (validated
/// against a fresh upload of the source).
public struct ModelSourceDescriptor: Sendable, Hashable {
    /// CLI key for `--model` (e.g. "gemma4", "v4flash").
    public let key: String
    public let family: ModelFamily
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public init(key: String,
                family: ModelFamily,
                displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64) {
        self.key = key
        self.family = family
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }
}

public extension SupportedModelSource {
    /// Gemma 4 26B-A4B IT 4-bit (MLX community re-pack). Unchanged values.
    static let gemma4 = ModelSourceDescriptor(
        key: "gemma4",
        family: .gemma4,
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: 1_073_741_824)

    /// DeepSeek V4-Flash. FP4 routed experts (I8 containers of e2m1 pairs,
    /// ue8m0 scales per 32 along K) plus FP8 e4m3 resident tensors with
    /// ue8m0 128x128 block scales. The MTP module is dropped at repack time.
    /// Sizes: ~147.2 GB routed experts + ~8 GB resident; index validated
    /// against revision `60d8d70` (2026-06-22 upload).
    static let deepseekV4Flash = ModelSourceDescriptor(
        key: "v4flash",
        family: .deepseekV4Flash,
        displayName: "DeepSeek V4-Flash (FP4 experts + FP8 resident)",
        repoID: "deepseek-ai/DeepSeek-V4-Flash",
        revision: "60d8d70770c6776ff598c94bb586a859a38244f1",
        sourceIndexSHA256:
            "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a",
        approximateDownloadBytes: 155_100_000_000,
        installedBytes: 155_100_000_000,
        reserveBytes: 1_073_741_824)

    static let all: [ModelSourceDescriptor] = [gemma4, deepseekV4Flash]

    /// Default model when `--model` is not passed.
    static let `default` = gemma4

    static func descriptor(forKey key: String) -> ModelSourceDescriptor? {
        all.first { $0.key == key }
    }
}
