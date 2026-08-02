import Foundation
import Metal
import TurboFieldfare

/// LogitProducer adapter for the DeepSeek V4 decode path (V4F-04/05).
///
/// The shared completion loop uses serial decode by default. Setting
/// `TURBO_V4_CHUNKED_PREFILL=1` selects the layer-major batched prompt path.
/// The runner produces FP32 logits; the loop's sampler consumes FP16, so
/// `copyLogits` converts on CPU. This remains a small follow-up optimization.
final class V4LogitProducer: ChunkedPrefillRunner, @unchecked Sendable {
    private let model: V4Model
    private let maxContext: Int
    private var runner: V4ForwardRunner
    private var vocab: Int { model.config.vocabSize }

    init(model: V4Model, maxContext: Int) throws {
        self.model = model
        self.maxContext = maxContext
        self.runner = try V4ForwardRunner(model: model, maxContext: maxContext)
    }

    /// Fresh KV/compressor state for a new generation. Allocation failure
    /// here is unrecoverable mid-loop, matching the runner's own preconditions.
    func reset() {
        runner = try! V4ForwardRunner(model: model, maxContext: maxContext)
    }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        let out = try await runner.forward(token: UInt32(bitPattern: token),
                                           position: position)
        copyLogits(from: out, into: logits)
    }

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        _ = outputMode
        let out = try await runner.prefillChunked(tokens: tokens,
                                                  startPosition: startPosition,
                                                  config: config)
        copyLogits(from: out, into: logits)
        onProgress(tokens.count)
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func copyLogits(from out: MTLBuffer, into logits: MTLBuffer) {
        let src = out.contents().assumingMemoryBound(to: Float.self)
        let dst = logits.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<vocab {
            dst[i] = Float16(src[i])
        }
    }
}

enum V4ChunkedPrefillEnvError: Error, CustomStringConvertible, Equatable {
    case invalidChunkTokens(String)

    var description: String {
        switch self {
        case .invalidChunkTokens(let value):
            return "TURBO_V4_PREFILL_CHUNK_TOKENS must be one of 32, 64, 128, or 256 when TURBO_V4_CHUNKED_PREFILL=1; got \"\(value)\""
        }
    }
}

func v4ChunkedPrefillConfigFromEnv(_ env: [String: String] = ProcessInfo.processInfo.environment) throws -> PrefillRuntimeConfig {
    guard env["TURBO_V4_CHUNKED_PREFILL"] == "1" else { return .off }
    let rawChunkTokens = env["TURBO_V4_PREFILL_CHUNK_TOKENS"] ?? "128"
    guard let chunkTokens = Int(rawChunkTokens), [32, 64, 128, 256].contains(chunkTokens) else {
        throw V4ChunkedPrefillEnvError.invalidChunkTokens(rawChunkTokens)
    }
    return .v4Production(chunkTokens: chunkTokens)
}

func v4IntegrityPolicyFromEnv(_ env: [String: String] = ProcessInfo.processInfo.environment) -> ModelIntegrityPolicy {
    env["TURBO_V4_TRUSTED_INSTALL"] == "1" ? .sizeCheckTrustedReceipt : .fullSha256
}

/// V4-family entry point. Raw `--prompt` completion only for now; the DSML
/// chat path (`V4ChatFormat`) wires in with the server/front-end work.
func runV4(args: Args,
           modelURL: URL,
           tokenizer: GFTokenizer,
           promptIds: [Int32],
           config: GenerationConfig,
           stdout: FileHandle,
           stderr: FileHandle) async -> RunResult {
    do {
        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try V4Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: .deepSeekV4Flash,
            streamingMode: .pread(slotCount: RuntimeConfiguration().expertCacheSlots),
            expertCachePolicy: .lfu,
            integrityPolicy: v4IntegrityPolicyFromEnv())
        let prefillConfig = try v4ChunkedPrefillConfigFromEnv()
        let producer = try V4LogitProducer(model: model, maxContext: args.maxContext)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize)
        var v4Config = config
        v4Config.extraStopTokens.formUnion([1])   // DeepSeek EOS
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let stats = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: v4Config,
            context: context,
            scratch: scratch,
            prefillConfig: prefillConfig) { progress in
                switch progress {
                case .prefill:
                    break
                case .token(_, _, let delta):
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }
        _ = start
        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let footer = "\n[family=v4flash stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok prefill_s=\(String(format: "%.3f", stats.prefillSeconds)) new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))]\n"
            stderr.write(Data(footer.utf8))
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}
