import Foundation

/// DeepSeek V4 chat framing, ported from the reference
/// `deepseek-ai/DeepSeek-V4-Pro` `encoding/encoding_dsv4.py`
/// (`encode_messages`, `render_message`, `merge_tool_messages`,
/// `sort_tool_results_by_call_order`, `encode_arguments_to_dsml`).
///
/// The tokenizer ships no `chat_template`; this is the authoritative
/// message format. Framing in chat (non-thinking) mode:
///
/// ```
/// <bos>{system}<｜User｜>{user}<｜Assistant｜></think>{response}<eos>
/// ```
///
/// Tool calls are DSML blocks emitted by the assistant, tool results are
/// merged into user messages as `<tool_result>...</tool_result>` blocks.
public enum V4ChatFormatError: Error, Equatable, CustomStringConvertible {
    case invalidMessage(String)
    case invalidArguments(String)

    public var description: String {
        switch self {
        case .invalidMessage(let detail): return "invalid V4 chat message: \(detail)"
        case .invalidArguments(let detail): return "invalid V4 tool-call arguments: \(detail)"
        }
    }
}

public enum V4ThinkingMode: String, Sendable {
    case chat
    case thinking
}

public enum V4ReasoningEffort: String, Sendable {
    case max
    case high
}

/// Quick-instruction task tokens for internal classification tasks.
public enum V4Task: String, Sendable, CaseIterable {
    case action
    case query
    case authority
    case domain
    case title
    case readURL = "read_url"

    public var specialToken: String {
        switch self {
        case .action: return V4SpecialToken.taskAction
        case .query: return V4SpecialToken.taskQuery
        case .authority: return V4SpecialToken.taskAuthority
        case .domain: return V4SpecialToken.taskDomain
        case .title: return V4SpecialToken.taskTitle
        case .readURL: return V4SpecialToken.taskReadURL
        }
    }
}

public enum V4Role: String, Sendable {
    case system
    case developer
    case user
    case assistant
    case latestReminder = "latest_reminder"
    /// OpenAI-style tool messages; merged into user messages by
    /// `mergeToolMessages` before rendering (V4 has no tool role).
    case tool
}

public enum V4ContentBlock: Sendable, Equatable {
    case text(String)
    /// Tool result merged into a user message; rendered as
    /// `<tool_result>{content}</tool_result>`.
    case toolResult(toolUseID: String, content: String)
}

/// One assistant tool call in OpenAI-ish form: a function name plus its
/// arguments as a JSON string (matching the reference's internal format).
public struct V4ToolCallSpec: Sendable, Equatable {
    public let id: String?
    public let name: String
    public let arguments: String

    public init(id: String? = nil, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }
}

/// A function tool schema, injected into the system prompt under `## Tools`.
public struct V4FunctionTool: Sendable, Equatable {
    public let name: String
    public let description: String
    /// Ordered JSON schema; preserves key order for byte-exact framing.
    public let parameters: V4JSONTerm

    public init(name: String, description: String, parameters: V4JSONTerm) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    /// Convenience init from a JSON schema string. Key order is preserved.
    public init(name: String, description: String, parametersJSON: String) throws {
        guard let term = try? V4JSONTerm.parse(parametersJSON) else {
            throw V4ChatFormatError.invalidArguments("tool \(name) parameters are not valid JSON")
        }
        self.init(name: name, description: description, parameters: term)
    }
}

public struct V4Message: Sendable, Equatable {
    public var role: V4Role
    public var content: String?
    /// User messages only: mixed text / tool-result blocks. When present,
    /// rendering uses blocks instead of `content`.
    public var contentBlocks: [V4ContentBlock]?
    /// Assistant messages only.
    public var toolCalls: [V4ToolCallSpec]?
    /// Assistant messages only; kept only for turns after the last user
    /// message when `dropThinking` is in effect.
    public var reasoningContent: String?
    /// System/developer messages only: tool schemas appended to content.
    public var tools: [V4FunctionTool]?
    /// System/developer messages only: response-format schema (ordered JSON).
    public var responseFormat: V4JSONTerm?
    /// Quick-instruction task token to append after this message.
    public var task: V4Task?
    /// Assistant messages only: omit the trailing EOS (used for prefixes).
    public var woEOS: Bool
    /// OpenAI-style tool messages: the call this result answers.
    public var toolCallID: String?

    public init(role: V4Role,
                content: String? = nil,
                contentBlocks: [V4ContentBlock]? = nil,
                toolCalls: [V4ToolCallSpec]? = nil,
                reasoningContent: String? = nil,
                tools: [V4FunctionTool]? = nil,
                responseFormat: V4JSONTerm? = nil,
                task: V4Task? = nil,
                woEOS: Bool = false,
                toolCallID: String? = nil) {
        self.role = role
        self.content = content
        self.contentBlocks = contentBlocks
        self.toolCalls = toolCalls
        self.reasoningContent = reasoningContent
        self.tools = tools
        self.responseFormat = responseFormat
        self.task = task
        self.woEOS = woEOS
        self.toolCallID = toolCallID
    }
}

public enum V4ChatFormat {
    /// Generation stops for V4 chat completions: EOS only. The reference
    /// decoder requires EOS after the DSML tool-call block as well.
    public static func stopTokenIDs(_ ids: V4SpecialTokenIDs = .pinned) -> Set<Int32> {
        ids.stopTokenIDs
    }

    private static let reasoningEffortMaxPrefix =
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        + "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
        + "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"

    private static let responseFormatTemplate =
        "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"

    /// `TOOLS_TEMPLATE` from the reference, with `｜DSML｜` substituted.
    private static func renderTools(_ tools: [V4FunctionTool]) -> String {
        let schemas = tools.map { tool in
            V4JSONTerm.object([
                ("name", .string(tool.name)),
                ("description", .string(tool.description)),
                ("parameters", tool.parameters),
            ]).dumped()
        }.joined(separator: "\n")
        let d = V4SpecialToken.dsml
        return """
        ## Tools

        You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<\(d)tool_calls>" block like the following:

        <\(d)tool_calls>
        <\(d)invoke name="$TOOL_NAME">
        <\(d)parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</\(d)parameter>
        ...
        </\(d)invoke>
        <\(d)invoke name="$TOOL_NAME2">
        ...
        </\(d)invoke>
        </\(d)tool_calls>

        String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

        If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

        Otherwise, output directly after </think> with tool calls or final response.

        ### Available Tool Schemas

        \(schemas)

        You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.

        """
    }

    // MARK: - encode_messages

    /// Encodes messages into the DeepSeek V4 prompt format.
    ///
    /// - Parameters:
    ///   - messages: Messages to render (tool-role messages are merged into
    ///     user messages first).
    ///   - context: Preceding messages that are already part of the encoded
    ///     prefix; only used for tool-result ordering and thinking-drop
    ///     calculations, not rendered.
    ///   - thinkingMode: `.chat` suppresses thinking (`</think>` right after
    ///     the assistant prefix); `.thinking` enables `<think>` reasoning.
    ///   - dropThinking: When true, reasoning content is dropped from
    ///     assistant turns before the last user message. Forced to false
    ///     when any message carries tool schemas (reference behavior).
    ///   - addBOS: Prepend `<｜begin▁of▁sentence｜>` when there is no context.
    ///   - reasoningEffort: `.max` prepends a fixed effort prefix in
    ///     thinking mode.
    public static func encodeMessages(
        _ messages: [V4Message],
        context: [V4Message] = [],
        thinkingMode: V4ThinkingMode,
        dropThinking: Bool = true,
        addBOS: Bool = true,
        reasoningEffort: V4ReasoningEffort? = nil
    ) throws -> String {
        var messages = mergeToolMessages(messages)
        var context = context

        messages = Array(sortToolResultsByCallOrder(context + messages).dropFirst(context.count))
        if !context.isEmpty {
            context = sortToolResultsByCallOrder(mergeToolMessages(context))
        }

        var fullMessages = context + messages
        var prompt = (addBOS && context.isEmpty) ? V4SpecialToken.bos : ""

        // Any message with tools defined disables thinking-dropping.
        let effectiveDropThinking =
            fullMessages.contains { $0.tools?.isEmpty == false } ? false : dropThinking

        let numToRender: Int
        let contextLength: Int
        if thinkingMode == .thinking && effectiveDropThinking {
            fullMessages = dropThinkingMessages(fullMessages)
            numToRender = fullMessages.count - dropThinkingMessages(context).count
            contextLength = fullMessages.count - numToRender
        } else {
            numToRender = messages.count
            contextLength = context.count
        }

        for i in 0..<numToRender {
            prompt += try renderMessage(
                index: i + contextLength,
                messages: fullMessages,
                thinkingMode: thinkingMode,
                dropThinking: effectiveDropThinking,
                reasoningEffort: reasoningEffort)
        }
        return prompt
    }

    // MARK: - render_message

    private static func renderMessage(
        index: Int,
        messages: [V4Message],
        thinkingMode: V4ThinkingMode,
        dropThinking: Bool,
        reasoningEffort: V4ReasoningEffort?
    ) throws -> String {
        let message = messages[index]
        let lastUserIndex = findLastUserIndex(messages)

        var prompt = ""
        if index == 0, thinkingMode == .thinking, reasoningEffort == .max {
            prompt += reasoningEffortMaxPrefix
        }

        switch message.role {
        case .system:
            prompt += message.content ?? ""
            if let tools = message.tools, !tools.isEmpty {
                prompt += "\n\n" + renderTools(tools)
            }
            if let responseFormat = message.responseFormat {
                prompt += "\n\n" + responseFormatTemplate + responseFormat.dumped()
            }

        case .developer:
            guard let content = message.content, !content.isEmpty else {
                throw V4ChatFormatError.invalidMessage("developer message requires content")
            }
            var developerContent = V4SpecialToken.user + content
            if let tools = message.tools, !tools.isEmpty {
                developerContent += "\n\n" + renderTools(tools)
            }
            if let responseFormat = message.responseFormat {
                developerContent += "\n\n" + responseFormatTemplate + responseFormat.dumped()
            }
            prompt += developerContent

        case .user:
            prompt += V4SpecialToken.user
            if let blocks = message.contentBlocks, !blocks.isEmpty {
                prompt += blocks.map { block -> String in
                    switch block {
                    case .text(let text):
                        return text
                    case .toolResult(_, let content):
                        return "<tool_result>\(content)</tool_result>"
                    }
                }.joined(separator: "\n\n")
            } else {
                prompt += message.content ?? ""
            }

        case .latestReminder:
            prompt += V4SpecialToken.latestReminder + (message.content ?? "")

        case .tool:
            throw V4ChatFormatError.invalidMessage(
                "V4 merges tool messages into user; call mergeToolMessages first")

        case .assistant:
            var toolCallsContent = ""
            if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                let rendered = try toolCalls.map { call in
                    "<\(V4SpecialToken.dsml)invoke name=\"\(call.name)\">\n"
                        + (try encodeArgumentsToDSML(call.arguments))
                        + "\n</\(V4SpecialToken.dsml)invoke>"
                }
                toolCallsContent = "\n\n<\(V4SpecialToken.dsml)tool_calls>\n"
                    + rendered.joined(separator: "\n")
                    + "\n</\(V4SpecialToken.dsml)tool_calls>"
            }

            let summary = message.content ?? ""
            let reasoning = message.reasoningContent ?? ""
            let previousHasTask = index > 0 && messages[index - 1].task != nil

            var thinkingPart = ""
            if thinkingMode == .thinking && !previousHasTask {
                if !dropThinking || index > lastUserIndex {
                    thinkingPart = reasoning + V4SpecialToken.thinkEnd
                }
            }

            prompt += thinkingPart + summary + toolCallsContent
            if !message.woEOS {
                prompt += V4SpecialToken.eos
            }
        }

        // Transition tokens based on what follows.
        if index + 1 < messages.count {
            let nextRole = messages[index + 1].role
            if nextRole != .assistant && nextRole != .latestReminder {
                return prompt
            }
        }

        if let task = message.task {
            if task != .action {
                prompt += task.specialToken
            } else {
                prompt += V4SpecialToken.assistant
                prompt += thinkingMode != .thinking
                    ? V4SpecialToken.thinkEnd : V4SpecialToken.thinkStart
                prompt += task.specialToken
            }
        } else if message.role == .user || message.role == .developer {
            prompt += V4SpecialToken.assistant
            if thinkingMode == .thinking && (!dropThinking || index >= lastUserIndex) {
                prompt += V4SpecialToken.thinkStart
            } else {
                prompt += V4SpecialToken.thinkEnd
            }
        }

        return prompt
    }

    // MARK: - encode_arguments_to_dsml

    /// Encodes an arguments JSON string into DSML parameter lines.
    ///
    /// Mirrors the reference: parse the JSON; on parse failure fall back to
    /// a single `arguments` string parameter. String values are emitted raw
    /// with `string="true"`; all other types are re-serialized with
    /// `json.dumps` semantics (", " / ": " separators) and `string="false"`.
    /// A valid JSON document that is not an object raises in the reference;
    /// here it throws `invalidArguments`.
    static func encodeArgumentsToDSML(_ arguments: String) throws -> String {
        let pairs: [(String, V4JSONTerm)]
        do {
            let parsed = try V4JSONTerm.parse(arguments)
            guard case .object(let object) = parsed else {
                throw V4ChatFormatError.invalidArguments(
                    "arguments JSON must be an object")
            }
            pairs = object
        } catch let error as V4ChatFormatError {
            throw error
        } catch {
            pairs = [("arguments", .string(arguments))]
        }
        let d = V4SpecialToken.dsml
        return pairs.map { key, value in
            switch value {
            case .string(let string):
                return "<\(d)parameter name=\"\(key)\" string=\"true\">\(string)</\(d)parameter>"
            default:
                return "<\(d)parameter name=\"\(key)\" string=\"false\">\(value.dumped())</\(d)parameter>"
            }
        }.joined(separator: "\n")
    }

    // MARK: - merge_tool_messages / sort_tool_results_by_call_order

    /// Merges OpenAI-style tool messages into user messages as
    /// `<tool_result>` content blocks. Consecutive plain user messages are
    /// appended as text blocks (reference behavior).
    public static func mergeToolMessages(_ messages: [V4Message]) -> [V4Message] {
        var merged: [V4Message] = []
        for message in messages {
            switch message.role {
            case .tool:
                let block = V4ContentBlock.toolResult(
                    toolUseID: message.toolCallID ?? "",
                    content: message.content ?? "")
                if var last = merged.last, last.role == .user, last.contentBlocks != nil {
                    last.contentBlocks?.append(block)
                    merged[merged.count - 1] = last
                } else {
                    merged.append(V4Message(role: .user, contentBlocks: [block]))
                }
            case .user:
                let textBlock = V4ContentBlock.text(message.content ?? "")
                if var last = merged.last, last.role == .user,
                   last.contentBlocks != nil, last.task == nil {
                    last.contentBlocks?.append(textBlock)
                    merged[merged.count - 1] = last
                } else {
                    var newMessage = V4Message(
                        role: .user,
                        content: message.content ?? "",
                        contentBlocks: [textBlock])
                    newMessage.task = message.task
                    newMessage.woEOS = message.woEOS
                    merged.append(newMessage)
                }
            default:
                merged.append(message)
            }
        }
        return merged
    }

    /// Sorts tool-result blocks within user messages by the order of the
    /// tool calls in the preceding assistant message.
    public static func sortToolResultsByCallOrder(_ messages: [V4Message]) -> [V4Message] {
        var messages = messages
        var lastToolCallOrder: [String: Int] = [:]

        for index in messages.indices {
            let message = messages[index]
            if message.role == .assistant, let toolCalls = message.toolCalls {
                lastToolCallOrder = [:]
                for (offset, call) in toolCalls.enumerated() {
                    if let id = call.id {
                        lastToolCallOrder[id] = offset
                    }
                }
            } else if message.role == .user, let blocks = message.contentBlocks {
                let toolBlocks = blocks.compactMap { block -> V4ContentBlock? in
                    guard case .toolResult = block else { return nil }
                    return block
                }
                if toolBlocks.count > 1 && !lastToolCallOrder.isEmpty {
                    let sorted = toolBlocks.sorted { lhs, rhs in
                        func order(of block: V4ContentBlock) -> Int {
                            guard case .toolResult(let id, _) = block else { return 0 }
                            return lastToolCallOrder[id] ?? 0
                        }
                        return order(of: lhs) < order(of: rhs)
                    }
                    var sortedIndex = 0
                    var newBlocks: [V4ContentBlock] = []
                    for block in blocks {
                        if case .toolResult = block {
                            newBlocks.append(sorted[sortedIndex])
                            sortedIndex += 1
                        } else {
                            newBlocks.append(block)
                        }
                    }
                    messages[index].contentBlocks = newBlocks
                }
            }
        }
        return messages
    }

    // MARK: - _drop_thinking_messages / helpers

    /// Drops reasoning content from assistant turns before the last user
    /// message, and developer messages before it entirely.
    static func dropThinkingMessages(_ messages: [V4Message]) -> [V4Message] {
        let lastUserIndex = findLastUserIndex(messages)
        var result: [V4Message] = []
        for (index, message) in messages.enumerated() {
            switch message.role {
            case .user, .system, .tool, .latestReminder:
                result.append(message)
            case _ where index >= lastUserIndex:
                result.append(message)
            case .assistant:
                var copy = message
                copy.reasoningContent = nil
                result.append(copy)
            case .developer:
                continue
            }
        }
        return result
    }

    private static func findLastUserIndex(_ messages: [V4Message]) -> Int {
        for index in messages.indices.reversed() {
            if messages[index].role == .user || messages[index].role == .developer {
                return index
            }
        }
        return -1
    }
}

// MARK: - Ordered JSON (json.dumps-compatible)

/// Ordered JSON term used for DSML argument encoding and tool-schema
/// rendering. Unlike `JSONValue`, object key order is preserved and the
/// serializer matches Python `json.dumps(v, ensure_ascii=False)` byte for
/// byte (", " between items, ": " after keys) so framed prompts match the
/// reference encoder exactly.
public indirect enum V4JSONTerm: Sendable, Equatable {
    case object([(String, V4JSONTerm)])
    case array([V4JSONTerm])
    case string(String)
    /// Integer literal, preserved verbatim.
    case integer(String)
    /// Float literal, normalized to Python `repr(float)` form at parse time.
    case float(String)
    case bool(Bool)
    case null

    public enum ParseError: Error, Equatable {
        case malformed
    }

    public static func == (lhs: V4JSONTerm, rhs: V4JSONTerm) -> Bool {
        switch (lhs, rhs) {
        case (.object(let a), .object(let b)):
            return a.count == b.count
                && zip(a, b).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        case (.array(let a), .array(let b)):
            return a == b
        case (.string(let a), .string(let b)):
            return a == b
        case (.integer(let a), .integer(let b)):
            return a == b
        case (.float(let a), .float(let b)):
            return a == b
        case (.bool(let a), .bool(let b)):
            return a == b
        case (.null, .null):
            return true
        default:
            return false
        }
    }

    /// Serializes with Python `json.dumps(ensure_ascii=False)` separators.
    public func dumped() -> String {
        switch self {
        case .object(let pairs):
            return "{"
                + pairs.map { "\(Self.dumpString($0)): \($1.dumped())" }.joined(separator: ", ")
                + "}"
        case .array(let items):
            return "[" + items.map { $0.dumped() }.joined(separator: ", ") + "]"
        case .string(let value):
            return Self.dumpString(value)
        case .integer(let literal), .float(let literal):
            return literal
        case .bool(let value):
            return value ? "true" : "false"
        case .null:
            return "null"
        }
    }

    /// JSON string escaping matching `json.dumps(ensure_ascii=False)`:
    /// `\"`, `\\`, short escapes for \b \f \n \r \t, `\u00xx` (lowercase)
    /// for other control characters, everything else verbatim.
    public static func dumpString(_ value: String) -> String {
        var result = "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case "\u{08}": result += "\\b"
            case "\u{0C}": result += "\\f"
            default:
                if scalar.value < 0x20 {
                    result += String(format: "\\u%04x", scalar.value)
                } else {
                    result.unicodeScalars.append(scalar)
                }
            }
        }
        result += "\""
        return result
    }

    /// Parses JSON preserving object key order. Floats are normalized to
    /// Python `repr(float)` form (e.g. `1e3` -> `1000.0`).
    public static func parse(_ text: String) throws -> V4JSONTerm {
        var parser = V4JSONParser(text)
        let value = try parser.parseValue()
        parser.skipWhitespace()
        guard parser.isAtEnd else { throw ParseError.malformed }
        return value
    }
}

private struct V4JSONParser {
    private let characters: [Character]
    private var index = 0

    init(_ text: String) {
        characters = Array(text)
    }

    var isAtEnd: Bool { index >= characters.count }

    mutating func skipWhitespace() {
        while index < characters.count, characters[index].isWhitespace { index += 1 }
    }

    mutating func parseValue() throws -> V4JSONTerm {
        skipWhitespace()
        guard index < characters.count else { throw V4JSONTerm.ParseError.malformed }
        switch characters[index] {
        case "{": return try parseObject()
        case "[": return try parseArray()
        case "\"": return .string(try parseString())
        case "t": try consume("true"); return .bool(true)
        case "f": try consume("false"); return .bool(false)
        case "n": try consume("null"); return .null
        default: return try parseNumber()
        }
    }

    mutating func parseObject() throws -> V4JSONTerm {
        try consume("{")
        skipWhitespace()
        var pairs: [(String, V4JSONTerm)] = []
        if take("}") { return .object(pairs) }
        while true {
            skipWhitespace()
            let key = try parseString()
            try consume(":")
            pairs.append((key, try parseValue()))
            skipWhitespace()
            if take("}") { return .object(pairs) }
            try consume(",")
        }
    }

    mutating func parseArray() throws -> V4JSONTerm {
        try consume("[")
        skipWhitespace()
        var items: [V4JSONTerm] = []
        if take("]") { return .array(items) }
        while true {
            items.append(try parseValue())
            skipWhitespace()
            if take("]") { return .array(items) }
            try consume(",")
        }
    }

    mutating func parseString() throws -> String {
        try consume("\"")
        var result = ""
        while index < characters.count {
            let character = characters[index]
            if character == "\"" {
                index += 1
                return result
            }
            if character == "\\" {
                index += 1
                guard index < characters.count else { throw V4JSONTerm.ParseError.malformed }
                let escape = characters[index]
                index += 1
                switch escape {
                case "\"": result += "\""
                case "\\": result += "\\"
                case "/": result += "/"
                case "b": result += "\u{08}"
                case "f": result += "\u{0C}"
                case "n": result += "\n"
                case "r": result += "\r"
                case "t": result += "\t"
                case "u": result += String(try parseUnicodeScalar())
                default: throw V4JSONTerm.ParseError.malformed
                }
            } else {
                result.append(character)
                index += 1
            }
        }
        throw V4JSONTerm.ParseError.malformed
    }

    mutating func parseUnicodeScalar() throws -> UnicodeScalar {
        let first = try parseCodeUnit()
        let scalar: UInt32
        if (0xD800...0xDBFF).contains(first) {
            guard index + 2 <= characters.count,
                  characters[index] == "\\",
                  characters[index + 1] == "u" else {
                throw V4JSONTerm.ParseError.malformed
            }
            index += 2
            let second = try parseCodeUnit()
            guard (0xDC00...0xDFFF).contains(second) else {
                throw V4JSONTerm.ParseError.malformed
            }
            scalar = 0x10000 + (UInt32(first - 0xD800) << 10) + UInt32(second - 0xDC00)
        } else {
            guard !(0xDC00...0xDFFF).contains(first) else {
                throw V4JSONTerm.ParseError.malformed
            }
            scalar = UInt32(first)
        }
        guard let value = UnicodeScalar(scalar) else {
            throw V4JSONTerm.ParseError.malformed
        }
        return value
    }

    mutating func parseCodeUnit() throws -> UInt16 {
        guard index + 4 <= characters.count else {
            throw V4JSONTerm.ParseError.malformed
        }
        var value: UInt16 = 0
        for _ in 0..<4 {
            guard let digit = characters[index].hexDigitValue else {
                throw V4JSONTerm.ParseError.malformed
            }
            value = value * 16 + UInt16(digit)
            index += 1
        }
        return value
    }

    mutating func parseNumber() throws -> V4JSONTerm {
        let start = index
        while index < characters.count,
              "-+0123456789.eE".contains(characters[index]) {
            index += 1
        }
        guard index > start else { throw V4JSONTerm.ParseError.malformed }
        let literal = String(characters[start..<index])
        guard literal.range(
            of: #"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$"#,
            options: .regularExpression) != nil else {
            throw V4JSONTerm.ParseError.malformed
        }
        if literal.contains(where: { ".eE".contains($0) }) {
            guard let value = Double(literal) else {
                throw V4JSONTerm.ParseError.malformed
            }
            return .float(pythonFloatRepr(value))
        }
        return .integer(literal)
    }

    mutating func consume(_ literal: String) throws {
        skipWhitespace()
        let value = Array(literal)
        guard index + value.count <= characters.count,
              Array(characters[index..<(index + value.count)]) == value else {
            throw V4JSONTerm.ParseError.malformed
        }
        index += value.count
    }

    mutating func take(_ literal: Character) -> Bool {
        skipWhitespace()
        guard index < characters.count, characters[index] == literal else { return false }
        index += 1
        return true
    }
}

/// Formats a Double the way Python's `repr(float)` / `json.dumps` does for
/// the magnitudes DSML arguments realistically carry: shortest round-trip
/// decimal, with a trailing `.0` for integral values and lowercase
/// exponents. `String(describing:)` on Double already matches Python repr
/// for these forms (e.g. `1000.0`, `1.5`, `1e-05`).
private func pythonFloatRepr(_ value: Double) -> String {
    String(describing: value)
}
