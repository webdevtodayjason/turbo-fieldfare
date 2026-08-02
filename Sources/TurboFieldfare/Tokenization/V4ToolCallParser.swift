import Foundation

public enum V4ToolCallParserError: Error, Equatable {
    case malformed
    case unknownTool(String)
    case oversized
}

/// Parsed V4 assistant completion: summary content, optional thinking, and
/// zero or more DSML tool calls.
public struct V4ParsedAssistantMessage: Equatable, Sendable {
    public let content: String
    public let reasoningContent: String
    public let toolCalls: [ParsedToolCall]

    public init(content: String, reasoningContent: String, toolCalls: [ParsedToolCall]) {
        self.content = content
        self.reasoningContent = reasoningContent
        self.toolCalls = toolCalls
    }
}

/// Parser for DeepSeek V4 DSML tool-call output, ported from the reference
/// `parse_message_from_completion_text` / `parse_tool_calls` in
/// `encoding_dsv4.py`, with the same defensive contract as
/// `GemmaToolCallParser`: any deviation from the grammar throws
/// `.malformed` (fail closed), unknown tool names throw `.unknownTool`,
/// and inputs over `maximumBytes` throw `.oversized`.
///
/// Grammar (assistant turn):
///
/// ```
/// [reasoning</think>]content[\n\n<｜DSML｜tool_calls>\n
///   <｜DSML｜invoke name="FN">\n
///     <｜DSML｜parameter name="P" string="true|false">VALUE</｜DSML｜parameter>\n
///     ...
///   </｜DSML｜invoke>\n
///   ...
/// </｜DSML｜tool_calls>][<eos>]
/// ```
///
/// `string="true"` values are raw strings; `string="false"` values must be
/// valid JSON (stricter than the reference, which would splice invalid
/// JSON into the arguments string unchecked).
public struct V4ToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    private static let toolCallsStart = "\n\n<｜DSML｜tool_calls"
    private static let toolCallsEnd = "</｜DSML｜tool_calls>"
    private static let invokeStart = "<｜DSML｜invoke"
    private static let invokeEnd = "</｜DSML｜invoke"
    private static let parameterStart = "<｜DSML｜parameter"
    private static let parameterEnd = "/｜DSML｜parameter"

    /// Parses a complete assistant completion (a single turn, including the
    /// trailing EOS unless it ends with a tool-call block).
    ///
    /// - Parameters:
    ///   - text: Raw completion text. Must contain EOS (or end exactly at
    ///     the tool-call block boundary).
    ///   - thinkingMode: `.thinking` requires a `</think>`-terminated
    ///     reasoning prefix; `.chat` treats all text as content.
    ///   - allowedTools: When non-nil, every invoked name must be present.
    ///   - callID: Maps a call's index to its ID (default `call_0`, ...).
    public func parseCompletion(
        _ text: String,
        thinkingMode: V4ThinkingMode,
        allowedTools: Set<String>? = nil,
        callID: (Int) -> String = { "call_\($0)" }
    ) throws -> V4ParsedAssistantMessage {
        guard text.utf8.count <= Self.maximumBytes else {
            throw V4ToolCallParserError.oversized
        }

        var index = text.startIndex
        var reasoning = ""

        if thinkingMode == .thinking {
            let (next, delta, stop) = Self.readUntilStop(
                text, from: index, stops: [V4SpecialToken.thinkEnd, Self.toolCallsStart])
            reasoning = String(delta)
            index = next
            guard stop == V4SpecialToken.thinkEnd else {
                throw V4ToolCallParserError.malformed
            }
        }

        let (afterContent, contentDelta, contentStop) = Self.readUntilStop(
            text, from: index, stops: [V4SpecialToken.eos, Self.toolCallsStart])
        let content = String(contentDelta)
        index = afterContent

        let isToolCalling = contentStop == Self.toolCallsStart
        if !isToolCalling {
            guard contentStop == V4SpecialToken.eos else {
                throw V4ToolCallParserError.malformed
            }
        }

        var toolCalls: [ParsedToolCall] = []
        var finalStop = contentStop
        if isToolCalling {
            let (afterCalls, calls) = try parseToolCalls(
                text, from: index, allowedTools: allowedTools, callID: callID)
            index = afterCalls
            toolCalls = calls

            let (afterTail, tail, tailStop) = Self.readUntilStop(
                text, from: index, stops: [V4SpecialToken.eos])
            guard tail.isEmpty else { throw V4ToolCallParserError.malformed }
            index = afterTail
            finalStop = tailStop
        }

        guard index == text.endIndex,
              finalStop == V4SpecialToken.eos || finalStop == nil else {
            throw V4ToolCallParserError.malformed
        }

        for special in V4SpecialToken.contentForbidden {
            guard !content.contains(special), !reasoning.contains(special) else {
                throw V4ToolCallParserError.malformed
            }
        }

        return V4ParsedAssistantMessage(
            content: content,
            reasoningContent: reasoning,
            toolCalls: toolCalls)
    }

    /// Parses a standalone, complete DSML tool-call block (text starting
    /// with `<｜DSML｜tool_calls>` and ending after
    /// `</｜DSML｜tool_calls>`; leading `\n\n` optional).
    public func parseToolCallBlock(
        _ text: String,
        allowedTools: Set<String>? = nil,
        callID: (Int) -> String = { "call_\($0)" }
    ) throws -> [ParsedToolCall] {
        guard text.utf8.count <= Self.maximumBytes else {
            throw V4ToolCallParserError.oversized
        }
        var index = text.startIndex
        if text.hasPrefix("\n\n") {
            index = text.index(index, offsetBy: 2)
        }
        guard text[index...].hasPrefix("<｜DSML｜tool_calls") else {
            throw V4ToolCallParserError.malformed
        }
        index = text.index(index, offsetBy: "<｜DSML｜tool_calls".count)
        let (end, calls) = try parseToolCalls(
            text, from: index, allowedTools: allowedTools, callID: callID)
        guard end == text.endIndex else { throw V4ToolCallParserError.malformed }
        return calls
    }

    // MARK: - parse_tool_calls

    private func parseToolCalls(
        _ text: String,
        from start: String.Index,
        allowedTools: Set<String>?,
        callID: (Int) -> String
    ) throws -> (String.Index, [ParsedToolCall]) {
        var index = start
        var calls: [ParsedToolCall] = []

        while index < text.endIndex {
            let (afterGap, gap, gapStop) = Self.readUntilStop(
                text, from: index, stops: [Self.invokeStart, Self.toolCallsEnd])
            guard gap == ">\n" else { throw V4ToolCallParserError.malformed }
            index = afterGap

            if gapStop == Self.toolCallsEnd { break }
            guard gapStop != nil else { throw V4ToolCallParserError.malformed }

            let (afterName, nameContent, nameStop) = Self.readUntilStop(
                text, from: index, stops: [Self.parameterStart, Self.invokeEnd])
            index = afterName
            let name = try Self.parseInvokeName(nameContent)
            if let allowedTools, !allowedTools.contains(name) {
                throw V4ToolCallParserError.unknownTool(name)
            }

            var arguments: [(name: String, value: String, isString: Bool)] = []
            var stop = nameStop
            while stop == Self.parameterStart {
                let (afterParam, paramContent, _) = Self.readUntilStop(
                    text, from: index, stops: [Self.parameterEnd])
                index = afterParam
                let parameter = try Self.parseParameter(paramContent)
                guard !arguments.contains(where: { $0.name == parameter.name }) else {
                    throw V4ToolCallParserError.malformed
                }
                arguments.append(parameter)

                let (afterSeparator, separator, separatorStop) = Self.readUntilStop(
                    text, from: index, stops: [Self.parameterStart, Self.invokeEnd])
                guard separator == ">\n" else { throw V4ToolCallParserError.malformed }
                index = afterSeparator
                stop = separatorStop
            }

            let call = try Self.buildCall(
                index: calls.count,
                name: name,
                arguments: arguments,
                callID: callID)
            calls.append(call)
        }

        return (index, calls)
    }

    /// Matches the reference regex `^\s*name="(.*?)">\n$` (DOTALL): the
    /// content after `<｜DSML｜invoke` must be optional whitespace,
    /// `name="`, the tool name, and a closing `">\n`.
    private static func parseInvokeName(_ content: Substring) throws -> String {
        var text = String(content)
        guard text.hasSuffix("\">\n") else { throw V4ToolCallParserError.malformed }
        text = String(text.dropLast(3))
        let trimmed = text.drop(while: { $0.isWhitespace })
        guard trimmed.hasPrefix("name=\"") else { throw V4ToolCallParserError.malformed }
        return String(trimmed.dropFirst("name=\"".count))
    }

    /// Matches the reference regex
    /// `^ name="(.*?)" string="(true|false)">(.*?)<$` (DOTALL) against the
    /// content between `<｜DSML｜parameter` and `/｜DSML｜parameter`.
    private static func parseParameter(
        _ content: Substring
    ) throws -> (name: String, value: String, isString: Bool) {
        let text = String(content)
        guard text.hasPrefix(" name=\"") else { throw V4ToolCallParserError.malformed }
        let rest = text.dropFirst(" name=\"".count)
        guard let marker = rest.range(of: "\" string=\"") else {
            throw V4ToolCallParserError.malformed
        }
        let name = String(rest[..<marker.lowerBound])
        let afterMarker = rest[marker.upperBound...]
        let isString: Bool
        let afterFlag: Substring
        if afterMarker.hasPrefix("true\">") {
            isString = true
            afterFlag = afterMarker.dropFirst("true\">".count)
        } else if afterMarker.hasPrefix("false\">") {
            isString = false
            afterFlag = afterMarker.dropFirst("false\">".count)
        } else {
            throw V4ToolCallParserError.malformed
        }
        guard afterFlag.hasSuffix("<") else { throw V4ToolCallParserError.malformed }
        return (name, String(afterFlag.dropLast()), isString)
    }

    /// Builds a `ParsedToolCall` the way the reference's
    /// `decode_dsml_to_arguments` does: `{"key": value, ...}` with
    /// `json.dumps`-formatted keys, raw values for `string="false"`, and
    /// JSON-quoted values for `string="true"`. The assembled arguments
    /// string must decode as valid JSON (fail closed on malformed
    /// `string="false"` values).
    private static func buildCall(
        index: Int,
        name: String,
        arguments: [(name: String, value: String, isString: Bool)],
        callID: (Int) -> String
    ) throws -> ParsedToolCall {
        let body = arguments.map { argument -> String in
            let value = argument.isString
                ? V4JSONTerm.dumpString(argument.value)
                : argument.value
            return "\(V4JSONTerm.dumpString(argument.name)): \(value)"
        }.joined(separator: ", ")
        let argumentsJSON = "{" + body + "}"

        let parsed: JSONValue
        do {
            parsed = try JSONDecoder().decode(JSONValue.self, from: Data(argumentsJSON.utf8))
        } catch {
            throw V4ToolCallParserError.malformed
        }
        guard case .object = parsed else { throw V4ToolCallParserError.malformed }

        return ParsedToolCall(
            id: callID(index),
            name: name,
            arguments: parsed,
            argumentsJSON: argumentsJSON)
    }

    // MARK: - _read_until_stop

    /// Reads from `index` until the earliest occurrence of any stop string.
    /// Returns the index after the matched stop (or end of text), the
    /// content before it, and the matched stop (or nil).
    private static func readUntilStop(
        _ text: String,
        from index: String.Index,
        stops: [String]
    ) -> (String.Index, Substring, String?) {
        var best: (range: Range<String.Index>, stop: String)?
        for stop in stops {
            if let range = text.range(of: stop, range: index..<text.endIndex) {
                if best == nil || range.lowerBound < best!.range.lowerBound {
                    best = (range, stop)
                }
            }
        }
        if let best {
            return (best.range.upperBound, text[index..<best.range.lowerBound], best.stop)
        }
        return (text.endIndex, text[index...], nil)
    }
}
