# V4F-07 design note: OpenAI-compatible server for DeepSeek V4-Flash

Status: design, 2026-08-02. Implement after V4F-05 holdouts. This is the
bridge from the port to fleet agentic work: Jcode spawns need a
loopback OpenAI endpoint with reliable tool calling.

## What exists today

- `TurboFieldfareServer` (Gemma): loopback-only OpenAI-compatible
  server, one warm model, serialized generation, one retained KV
  prefix, tool loops with fail-closed parsing. See
  docs/OPENAI_SERVER.md.
- V4 pieces already landed: `V4ChatFormat` (DSML framing, tool
  schemas, tool-result merging), `V4ToolCallParser` (fail-closed DSML
  parse), `V4LogitProducer` + `runRawCompletion` (the generation
  loop), pipelined decode, DSML `--messages-file` path.

## Shape of the work

1. **Family dispatch at startup.** Probe the model directory with
   `ManifestReader.probeModelFamily`; Gemma keeps the current server,
   V4 takes the new path. No behavioral change for Gemma.
2. **V4 chat completions.** Requests map OpenAI messages to
   `V4Message` (system -> developer per the DSML contract, tool
   messages merged per `mergeToolMessages`), render with
   `V4ChatFormat.encodeMessages` (thinking mode from request or
   server config), generate, stop on EOS id 1 and the DSML tool-call
   boundary, parse complete tool-call blocks with `V4ToolCallParser`,
   fail closed on malformed output (same contract as the Gemma
   server).
3. **Streaming.** SSE deltas from the raw-completion progress events;
   tool-call blocks buffer until complete before emitting (never
   partial-parse, per the fail-closed rule).
4. **KV prefix retention.** The Gemma server retains one verified
   conversational prefix. V4 equivalent needs the
   `ContinuableLogitProducer` contract implemented for the V4 runner
   (reset + continuation position), or v1 ships without prefix
   retention (recorded deviation).
5. **Serialization + resource guard.** One model process at a time,
   same as Gemma. The V4 resident footprint (~8.8 GB mapped + slot
   pool) must be recorded in the server docs.

## Open questions to resolve at implementation time

- Tool-call streaming granularity: whole-block buffering vs
  argument-delta streaming (match what Jcode's provider client
  tolerates; whole-block first).
- Thinking mode default for coding agents (`.chat` until tool loops
  demand otherwise; the DSML thinking-drop behavior with tools
  present is already handled by `encodeMessages`).
- Seed/temperature pass-through mapping to the shared sampler config.
- Whether the retained-prefix KV (window ring + compressed entries)
  can be snapshotted cheaply, or continuation is recompute-only in v1.

## Gates

- OpenAI client (openai python SDK) round-trip: chat completion,
  streaming, one tool loop (two calls with a result round-trip), and
  a malformed-output fail-closed case.
- Loopback only (127.0.0.1), matching the security posture; never
  proxied or exposed.
- Server lifecycle from the process rules: only a server this
  session launched, and no second model process.
