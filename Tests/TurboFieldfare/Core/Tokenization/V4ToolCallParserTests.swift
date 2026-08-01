import Foundation
import Testing
@testable import TurboFieldfare

/// Golden and fail-closed tests for the DeepSeek V4 DSML tool-call parser.
/// Expected values come from running the reference
/// `parse_message_from_completion_text` in `encoding_dsv4.py`.
@Suite("V4ToolCallParser")
struct V4ToolCallParserTests {
    private static let eos = "<｜end▁of▁sentence｜>"
    private static let d = "｜DSML｜"

    private let parser = V4ToolCallParser()

    @Test("Plain chat completion without tool calls")
    func plainCompletion() throws {
        let message = try parser.parseCompletion(
            "The answer is 4.\(Self.eos)", thinkingMode: .chat)
        #expect(message.content == "The answer is 4.")
        #expect(message.reasoningContent == "")
        #expect(message.toolCalls.isEmpty)
    }

    @Test("Thinking completion splits reasoning from content")
    func thinkingCompletion() throws {
        let message = try parser.parseCompletion(
            "reasoning here</think>summary answer\(Self.eos)", thinkingMode: .thinking)
        #expect(message.content == "summary answer")
        #expect(message.reasoningContent == "reasoning here")
        #expect(message.toolCalls.isEmpty)
    }

    @Test("Single invoke with a string parameter")
    func singleInvoke() throws {
        let d = Self.d
        let text = "calling\n\n<\(d)tool_calls>\n<\(d)invoke name=\"get_weather\">\n"
            + "<\(d)parameter name=\"city\" string=\"true\">Paris</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>\(Self.eos)"
        let message = try parser.parseCompletion(text, thinkingMode: .chat)
        #expect(message.content == "calling")
        #expect(message.toolCalls.count == 1)
        let call = try #require(message.toolCalls.first)
        #expect(call.name == "get_weather")
        #expect(call.argumentsJSON == #"{"city": "Paris"}"#)
        #expect(call.arguments == .object(["city": .string("Paris")]))
        #expect(call.id == "call_0")
    }

    @Test("Multiple invokes with typed parameters and thinking")
    func multipleInvokesTypedParameters() throws {
        let d = Self.d
        let text = "think first</think>result\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"n\" string=\"false\">42</\(d)parameter>\n"
            + "<\(d)parameter name=\"s\" string=\"true\">hello</\(d)parameter>\n"
            + "<\(d)parameter name=\"obj\" string=\"false\">{\"k\": [1, 2], \"u\": null}</\(d)parameter>\n"
            + "</\(d)invoke>\n<\(d)invoke name=\"g\">\n</\(d)invoke>\n"
            + "</\(d)tool_calls>\(Self.eos)"
        let message = try parser.parseCompletion(text, thinkingMode: .thinking)
        #expect(message.reasoningContent == "think first")
        #expect(message.content == "result")
        #expect(message.toolCalls.count == 2)
        #expect(message.toolCalls[0].name == "f")
        #expect(message.toolCalls[0].argumentsJSON == #"{"n": 42, "s": "hello", "obj": {"k": [1, 2], "u": null}}"#)
        #expect(message.toolCalls[0].arguments == .object([
            "n": .integer(42),
            "s": .string("hello"),
            "obj": .object(["k": .array([.integer(1), .integer(2)]), "u": .null]),
        ]))
        #expect(message.toolCalls[1].name == "g")
        #expect(message.toolCalls[1].argumentsJSON == "{}")
        #expect(message.toolCalls[1].arguments == .object([:]))
        #expect(message.toolCalls[1].id == "call_1")
    }

    @Test("Parameter values may contain > < and newlines")
    func parameterValueWithMarkupCharacters() throws {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"code\" string=\"true\">a > b\nif x < 3</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>\(Self.eos)"
        let message = try parser.parseCompletion(text, thinkingMode: .chat)
        #expect(message.toolCalls.first?.argumentsJSON == #"{"code": "a > b\nif x < 3"}"#)
    }

    @Test("Unicode string parameters pass through unescaped")
    func unicodeParameter() throws {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"q\" string=\"true\">café “quote”</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>\(Self.eos)"
        let message = try parser.parseCompletion(text, thinkingMode: .chat)
        #expect(message.toolCalls.first?.argumentsJSON == #"{"q": "café “quote”"}"#)
    }

    @Test("Tool-call block without trailing EOS is accepted")
    func toolCallBlockWithoutEOS() throws {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n</\(d)invoke>\n</\(d)tool_calls>"
        let message = try parser.parseCompletion(text, thinkingMode: .chat)
        #expect(message.toolCalls.count == 1)
    }

    @Test("Standalone block parse")
    func standaloneBlock() throws {
        let d = Self.d
        let text = "<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"a\" string=\"false\">1</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>"
        let calls = try parser.parseToolCallBlock(text)
        #expect(calls.count == 1)
        #expect(calls.first?.argumentsJSON == #"{"a": 1}"#)
    }

    @Test("Allowed-tools gate accepts listed names and rejects others")
    func allowedTools() throws {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n</\(d)invoke>\n</\(d)tool_calls>\(Self.eos)"
        #expect(throws: Never.self) {
            try parser.parseCompletion(text, thinkingMode: .chat, allowedTools: ["f"])
        }
        #expect(throws: V4ToolCallParserError.unknownTool("f")) {
            try parser.parseCompletion(text, thinkingMode: .chat, allowedTools: ["g"])
        }
    }

    // MARK: - Fail-closed cases (reference raises ValueError/AssertionError)

    @Test("Missing EOS fails closed")
    func missingEOS() {
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion("no eos at all", thinkingMode: .chat)
        }
    }

    @Test("Duplicate parameter name fails closed")
    func duplicateParameter() {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"a\" string=\"true\">1</\(d)parameter>\n"
            + "<\(d)parameter name=\"a\" string=\"true\">2</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>\(Self.eos)"
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion(text, thinkingMode: .chat)
        }
    }

    @Test("Invalid string flag fails closed")
    func invalidStringFlag() {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"a\" string=\"yes\">1</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>\(Self.eos)"
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion(text, thinkingMode: .chat)
        }
    }

    @Test("Content after the tool-call block fails closed")
    func trailingContentAfterToolCalls() {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n</\(d)invoke>\n</\(d)tool_calls>trailing\(Self.eos)"
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion(text, thinkingMode: .chat)
        }
    }

    @Test("Thinking mode without </think> fails closed")
    func thinkingMissingClose() {
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion("no close tag\(Self.eos)", thinkingMode: .thinking)
        }
    }

    @Test("Missing invoke close fails closed")
    func missingInvokeClose() {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"a\" string=\"true\">1</\(d)parameter>\n"
            + "</\(d)tool_calls>\(Self.eos)"
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion(text, thinkingMode: .chat)
        }
    }

    @Test("string=\"false\" with invalid JSON fails closed")
    func invalidJSONValue() {
        let d = Self.d
        let text = "x\n\n<\(d)tool_calls>\n<\(d)invoke name=\"f\">\n"
            + "<\(d)parameter name=\"a\" string=\"false\">not json</\(d)parameter>\n"
            + "</\(d)invoke>\n</\(d)tool_calls>\(Self.eos)"
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion(text, thinkingMode: .chat)
        }
    }

    @Test("Special tokens inside content fail closed")
    func specialTokenInContent() {
        let d = Self.d
        let text = "sneaky \(d) token\(Self.eos)"
        #expect(throws: V4ToolCallParserError.malformed) {
            try parser.parseCompletion(text, thinkingMode: .chat)
        }
    }

    @Test("Oversized input fails closed")
    func oversized() {
        let big = String(repeating: "a", count: V4ToolCallParser.maximumBytes + 1)
        #expect(throws: V4ToolCallParserError.oversized) {
            try parser.parseCompletion(big, thinkingMode: .chat)
        }
    }

    @Test("Round trip: framed assistant tool call parses back to identical arguments")
    func frameThenParse() throws {
        let framed = try V4ChatFormat.encodeMessages(
            [V4Message(role: .user, content: "Q"),
             V4Message(role: .assistant, content: "checking",
                       toolCalls: [V4ToolCallSpec(
                        name: "get_weather",
                        arguments: #"{"city": "Paris", "days": 3}"#)])],
            thinkingMode: .chat)
        // Extract the assistant turn: after the final "<｜Assistant｜></think>".
        let marker = "<｜Assistant｜></think>"
        let start = try #require(framed.range(of: marker, options: .backwards)).upperBound
        let completion = String(framed[start...])
        let message = try parser.parseCompletion(completion, thinkingMode: .chat)
        #expect(message.content == "checking")
        #expect(message.toolCalls.count == 1)
        #expect(message.toolCalls.first?.name == "get_weather")
        #expect(message.toolCalls.first?.argumentsJSON == #"{"city": "Paris", "days": 3}"#)
    }
}
