import Testing
@testable import TurboFieldfareCLICore

@Suite struct V4ChunkedPrefillEnvTests {
    @Test func disabledUnlessFlagIsExactlyOneAndIgnoresChunkSize() throws {
        let missing = try v4ChunkedPrefillConfigFromEnv([:])
        #expect(missing.mode == .off)
        #expect(missing.chunkTokens == 128)

        let notExact = try v4ChunkedPrefillConfigFromEnv([
            "TURBO_V4_CHUNKED_PREFILL": "true",
            "TURBO_V4_PREFILL_CHUNK_TOKENS": "bogus",
        ])
        #expect(notExact.mode == .off)
        #expect(notExact.chunkTokens == 128)
    }

    @Test func enabledDefaultsTo128AndAcceptsAllowedChunkSizes() throws {
        let defaulted = try v4ChunkedPrefillConfigFromEnv(["TURBO_V4_CHUNKED_PREFILL": "1"])
        #expect(defaulted.mode == .chunked)
        #expect(defaulted.chunkTokens == 128)

        for chunk in [32, 64, 128, 256] {
            let config = try v4ChunkedPrefillConfigFromEnv([
                "TURBO_V4_CHUNKED_PREFILL": "1",
                "TURBO_V4_PREFILL_CHUNK_TOKENS": "\(chunk)",
            ])
            #expect(config.mode == .chunked)
            #expect(config.chunkTokens == chunk)
        }
    }

    @Test func enabledRejectsInvalidChunkSizeWithClearError() {
        #expect(throws: V4ChunkedPrefillEnvError.invalidChunkTokens("16")) {
            _ = try v4ChunkedPrefillConfigFromEnv([
                "TURBO_V4_CHUNKED_PREFILL": "1",
                "TURBO_V4_PREFILL_CHUNK_TOKENS": "16",
            ])
        }

        #expect(throws: V4ChunkedPrefillEnvError.invalidChunkTokens("abc")) {
            _ = try v4ChunkedPrefillConfigFromEnv([
                "TURBO_V4_CHUNKED_PREFILL": "1",
                "TURBO_V4_PREFILL_CHUNK_TOKENS": "abc",
            ])
        }
    }

    @Test func trustedInstallRequiresExactOptIn() {
        #expect(v4IntegrityPolicyFromEnv([:]) == .fullSha256)
        #expect(v4IntegrityPolicyFromEnv(["TURBO_V4_TRUSTED_INSTALL": "true"]) == .fullSha256)
        #expect(v4IntegrityPolicyFromEnv(["TURBO_V4_TRUSTED_INSTALL": "1"]) == .sizeCheckTrustedReceipt)
    }
}
