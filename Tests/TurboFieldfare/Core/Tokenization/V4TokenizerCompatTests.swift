import Foundation
import Testing
import Tokenizers
@testable import TurboFieldfare

/// Compatibility check: the existing tokenizer loader path
/// (swift-transformers `AutoTokenizer.from(modelFolder:)`, the same loader
/// `GFTokenizer` uses) must consume DeepSeek V4's `tokenizer.json`
/// (GPT-2-family BPE: byte-level pretokenizer with Split patterns for
/// number groups and CJK runs, byte-level decoder).
///
/// These tests are gated on the fetched fixture (`scratch/v4f-recon/
/// tokenizer.json` or `$V4F_TOKENIZER_DIR`); without it they skip so the
/// suite stays self-contained. Round-trip checks are self-validating
/// (encode -> decode must reproduce the input); the special-token ID table
/// is validated against the values pinned from the HF repo.
@Suite("V4TokenizerCompat")
struct V4TokenizerCompatTests {
    private func loadTokenizer() async throws -> (any Tokenizer)? {
        guard let url = V4TestFixtures.tokenizerJSONURL else { return nil }
        return try await AutoTokenizer.from(modelFolder: url.deletingLastPathComponent())
    }

    @Test("Loader consumes DeepSeek V4 tokenizer.json and resolves the special-token table")
    func loadsAndResolvesSpecialTokens() async throws {
        guard let tokenizer = try await loadTokenizer() else { return }
        let pinned = V4SpecialTokenIDs.pinned
        let expectations: [(String, Int32)] = [
            (V4SpecialToken.bos, pinned.bos),
            (V4SpecialToken.eos, pinned.eos),
            (V4SpecialToken.thinkStart, pinned.thinkStart),
            (V4SpecialToken.thinkEnd, pinned.thinkEnd),
            (V4SpecialToken.dsml, pinned.dsml),
            (V4SpecialToken.user, pinned.user),
            (V4SpecialToken.assistant, pinned.assistant),
            (V4SpecialToken.latestReminder, pinned.latestReminder),
            (V4SpecialToken.taskAction, pinned.taskAction),
            (V4SpecialToken.taskQuery, pinned.taskQuery),
            (V4SpecialToken.taskAuthority, pinned.taskAuthority),
            (V4SpecialToken.taskDomain, pinned.taskDomain),
            (V4SpecialToken.taskTitle, pinned.taskTitle),
            (V4SpecialToken.taskReadURL, pinned.taskReadURL),
        ]
        for (content, id) in expectations {
            #expect(tokenizer.convertTokenToId(content) == Int(id),
                    "unexpected id for \(content)")
        }
    }

    @Test("Special tokens encode to their pinned single IDs")
    func specialTokensEncodeAtomically() async throws {
        guard let tokenizer = try await loadTokenizer() else { return }
        let pinned = V4SpecialTokenIDs.pinned
        let cases: [(String, Int32)] = [
            (V4SpecialToken.user, pinned.user),
            (V4SpecialToken.assistant, pinned.assistant),
            (V4SpecialToken.dsml, pinned.dsml),
            (V4SpecialToken.thinkEnd, pinned.thinkEnd),
        ]
        for (text, id) in cases {
            #expect(tokenizer.encode(text: text, addSpecialTokens: false) == [Int(id)],
                    "expected \(text) to encode to [\(id)]")
        }
    }

    @Test("Round trips: English, CJK, digits, emoji, DSML markup")
    func roundTrips() async throws {
        guard let tokenizer = try await loadTokenizer() else { return }
        let cases = [
            "The capital of France is",
            "Hello, world! 12345",
            "DeepSeek V4 uses 256 routed experts and top-6 routing.",
            "你好，世界",
            "日本語のテキスト",
            "café — naïve “quotes” 😀",
            "def f(x):\n    return x * 2\n",
            "<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n</｜DSML｜invoke>",
        ]
        for text in cases {
            let ids = tokenizer.encode(text: text, addSpecialTokens: false)
            #expect(!ids.isEmpty)
            let decoded = tokenizer.decode(tokens: ids, skipSpecialTokens: false)
            #expect(decoded == text, "round trip failed for \(text)")
        }
    }

    @Test("A framed chat prompt encodes with the expected special-token layout")
    func framedPromptLayout() async throws {
        guard let tokenizer = try await loadTokenizer() else { return }
        let framed = try V4ChatFormat.encodeMessages(
            [V4Message(role: .system, content: "You are helpful."),
             V4Message(role: .user, content: "Hi")],
            thinkingMode: .chat)
        let ids = tokenizer.encode(text: framed, addSpecialTokens: false).map(Int32.init)
        let pinned = V4SpecialTokenIDs.pinned
        #expect(ids.first == pinned.bos)
        #expect(ids.contains(pinned.user))
        #expect(ids.contains(pinned.assistant))
        #expect(ids.last == pinned.thinkEnd)
        // Full round trip of the framed prompt.
        #expect(tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: false) == framed)
    }
}
