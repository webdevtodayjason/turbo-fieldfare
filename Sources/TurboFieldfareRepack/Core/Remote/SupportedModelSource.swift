import Foundation

/// Backward-compatible entry points for the default (Gemma) source.
/// New code should use `SupportedModelSource.descriptor(forKey:)` or the
/// concrete `ModelSourceDescriptor` values in `ModelSourceDescriptor.swift`.
public enum SupportedModelSource {
    public static let displayName = SupportedModelSource.gemma4.displayName
    public static let repoID = SupportedModelSource.gemma4.repoID
    public static let revision = SupportedModelSource.gemma4.revision
    public static let sourceIndexSHA256 = SupportedModelSource.gemma4.sourceIndexSHA256
    public static let approximateDownloadBytes = SupportedModelSource.gemma4.approximateDownloadBytes
    public static let installedBytes = SupportedModelSource.gemma4.installedBytes
    public static let reserveBytes = SupportedModelSource.gemma4.reserveBytes

    public static func installOptions(outputDirectory: URL,
                                      overwrite: Bool,
                                      token: String?,
                                      resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        SupportedModelSource.gemma4.installOptions(outputDirectory: outputDirectory,
                                                   overwrite: overwrite,
                                                   token: token,
                                                   resume: resume)
    }
}
