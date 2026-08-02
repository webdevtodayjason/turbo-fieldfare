# V4F-05 holdout benchmark plan for DeepSeek V4-Flash

Status: plan, 2026-08-02. Run AFTER V4F-06c prefill lands (prefill cost
dominates these numbers until then) and after a full clean release build
from a committed tree.

## Protocol deviations from COMMUNITY_BENCHMARKS.md

The community protocol targets the Gemma install. For V4-Flash the same
shape applies with these declared deviations (to be listed in every
result):

1. Model: `scratch/deepseek-v4-flash.gturbo` (145 GB), family
   `deepseek-v4-flash`, manifest minor 1. Record its manifest SHA-256
   instead of the Gemma one.
2. Sampling: protocol defaults (temperature 0.2, Top-K 64, Top-P 0.95,
   seed per case). The V4 sampler path must be verified to honor these
   before runs; record the sampler configuration in the result.
3. Chat framing: DSML via `V4ChatFormat` (`--messages-file`), not the
   Gemma template. The frozen prompts in
   `docs/benchmark-prompts/real-generation-v1/` are reused as
   user-message bodies so results are comparable in shape.
4. Context: 4,096 tokens per protocol. HCA compressed path is active
   at this depth; record that fact.

## Pre-run gates (must all pass)

- Full serial suite green (637+ tests) from the exact commit measured.
- Greedy determinism: two back-to-back 259-token runs byte-identical
  (the 06a race regression test).
- Ladder through level 5 with zero failures
  (`Scripts/v4f/longctx_ladder.py --max-level 5`).
- The model-process check from the protocol prints nothing.

## Measurements to record (per case)

- TTFT / prefill seconds and prefill tokens (this is the 06c payoff
  number; record explicitly).
- Decode seconds, new tokens, tok/s, stop reason.
- I/O per token if the runner exposes counters (record if absent).
- Process RSS and physical footprint before/after (the 128 GB Mac's
  expert-cache residency claim).
- Output text saved for coherence review (no repeating calibration
  prompt; protocol rule).

## What gets reported as results

- Commit, hardware, RAM, macOS, Swift version, exact commands, exit
  codes, complete timing footers, every protocol deviation above.
- Explicitly separated: memory-funded configuration (expert slot
  count) vs memory-free numbers, per CACHE-03's standing rule.

## Not in scope for the first holdout

- 2K+ context cases (run separately after 06c makes prefill
  affordable; the ladder already covers up to ~1.2K in principle).
- Tool-call loops (DSML parse coverage exists in tests; live server
  runs come with the server integration work).
