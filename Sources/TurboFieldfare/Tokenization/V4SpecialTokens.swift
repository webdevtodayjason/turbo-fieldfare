import Foundation

/// DeepSeek V4 special-token text and ID table.
///
/// The tokenizer ships no `chat_template`; the chat format lives in the
/// reference `encoding/encoding_dsv4.py` and the tokens below are its
/// framing vocabulary. `pinned` is the ID table from
/// `deepseek-ai/DeepSeek-V4-Flash` `tokenizer.json` (`added_tokens`,
/// vocab size 129280); `load(fromTokenizerJSON:)` re-derives the table
/// from any V4-family `tokenizer.json` and validates it against the
/// expected token text so a mismatched tokenizer fails loudly.
public struct V4SpecialTokenIDs: Sendable, Equatable {
    public let bos: Int32
    public let eos: Int32
    public let thinkStart: Int32
    public let thinkEnd: Int32
    public let dsml: Int32
    public let user: Int32
    public let assistant: Int32
    public let latestReminder: Int32
    public let taskAction: Int32
    public let taskQuery: Int32
    public let taskAuthority: Int32
    public let taskDomain: Int32
    public let taskTitle: Int32
    public let taskReadURL: Int32

    /// Generation stops for V4 chat: EOS terminates an assistant turn in
    /// both plain and DSML tool-call completions (the reference decoder
    /// asserts EOS after the `</｜DSML｜tool_calls>` block).
    public var stopTokenIDs: Set<Int32> { [eos] }
}

public enum V4SpecialToken {
    public static let bos = "<｜begin▁of▁sentence｜>"
    public static let eos = "<｜end▁of▁sentence｜>"
    public static let thinkStart = "<think>"
    public static let thinkEnd = "</think>"
    /// Atomic added token (id 128825). DSML tags like `<｜DSML｜invoke` are
    /// composed: literal `<` + this token + the tag name.
    public static let dsml = "｜DSML｜"
    public static let user = "<｜User｜>"
    public static let assistant = "<｜Assistant｜>"
    public static let latestReminder = "<｜latest_reminder｜>"
    public static let taskAction = "<｜action｜>"
    public static let taskQuery = "<｜query｜>"
    public static let taskAuthority = "<｜authority｜>"
    public static let taskDomain = "<｜domain｜>"
    public static let taskTitle = "<｜title｜>"
    public static let taskReadURL = "<｜read_url｜>"

    /// Tokens the reference parser forbids inside generated content.
    public static let contentForbidden: [String] = [bos, eos, "<think>", "</think>", dsml]
}

public enum V4SpecialTokenError: Error, Equatable, CustomStringConvertible {
    case unreadableTokenizerJSON(String)
    case missingToken(String)

    public var description: String {
        switch self {
        case .unreadableTokenizerJSON(let detail):
            return "cannot read V4 tokenizer.json: \(detail)"
        case .missingToken(let content):
            return "tokenizer.json added_tokens missing V4 token \(content)"
        }
    }
}

extension V4SpecialTokenIDs {
    /// IDs pinned from `deepseek-ai/DeepSeek-V4-Flash` `tokenizer.json`.
    public static let pinned = V4SpecialTokenIDs(
        bos: 0,
        eos: 1,
        thinkStart: 128821,
        thinkEnd: 128822,
        dsml: 128825,
        user: 128803,
        assistant: 128804,
        latestReminder: 128828,
        taskAction: 128829,
        taskQuery: 128830,
        taskAuthority: 128831,
        taskDomain: 128832,
        taskTitle: 128836,
        taskReadURL: 128845
    )

    /// Derives the table from a V4-family `tokenizer.json` (`added_tokens`).
    /// Throws if any required token text is absent.
    public static func load(fromTokenizerJSON url: URL) throws -> V4SpecialTokenIDs {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw V4SpecialTokenError.unreadableTokenizerJSON(error.localizedDescription)
        }
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let added = root["added_tokens"] as? [[String: Any]] else {
            throw V4SpecialTokenError.unreadableTokenizerJSON("missing added_tokens array")
        }
        var idByContent: [String: Int32] = [:]
        for entry in added {
            guard let content = entry["content"] as? String,
                  let id = entry["id"] as? Int else { continue }
            idByContent[content] = Int32(id)
        }
        func require(_ content: String) throws -> Int32 {
            guard let id = idByContent[content] else {
                throw V4SpecialTokenError.missingToken(content)
            }
            return id
        }
        return try V4SpecialTokenIDs(
            bos: require(V4SpecialToken.bos),
            eos: require(V4SpecialToken.eos),
            thinkStart: require(V4SpecialToken.thinkStart),
            thinkEnd: require(V4SpecialToken.thinkEnd),
            dsml: require(V4SpecialToken.dsml),
            user: require(V4SpecialToken.user),
            assistant: require(V4SpecialToken.assistant),
            latestReminder: require(V4SpecialToken.latestReminder),
            taskAction: require(V4SpecialToken.taskAction),
            taskQuery: require(V4SpecialToken.taskQuery),
            taskAuthority: require(V4SpecialToken.taskAuthority),
            taskDomain: require(V4SpecialToken.taskDomain),
            taskTitle: require(V4SpecialToken.taskTitle),
            taskReadURL: require(V4SpecialToken.taskReadURL)
        )
    }
}
