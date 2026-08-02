# V4F-06c-B worker brief: the chunked prefill runner

Spawn this worker AFTER 06c-A1 (batched boundary: mHC/norms/RoPE) and
06c-A2 (batched projections/window attention/o-proj) are integrated.
Fill in their exact call signatures from their reports before spawning.

## Mission

Compose the chunked prefill runner for DeepSeek V4-Flash: up to 128
prompt rows through the 43-layer stack layer-major, then hand off to
the pipelined decode runner. Wire it into the CLI behind
`TURBO_V4_CHUNKED_PREFILL=1` (exact toggle name is the acceptance
contract — the gate script uses it).

## Inputs you compose

- 06c-A1: V4PrefillBoundary (batched mHC params/pre/post, batched
  fp32-in/fp16-out RMSNorm, batched interleaved RoPE forward+inverse).
- 06c-A2: V4PrefillProj (batched FP8/BF16 GEMMs, window-KV ring writes,
  causal window MQA prefill attention, grouped o-projection).
- 3db4098: PrefillGroupedRoutedMoEV4 (grouped FP4 MoE with 8-expert
  tiles + scheduler; integration seam in its header docs and the 06b
  worker report).
- Committed: V4Model (resident accessors, hashExpertIDs,
  planRoutedExperts, fetchRoutedExperts), CompressedKVCacheManager,
  MoEV4 router (per-token; prefill may need a batched router — if A2's
  BF16 GEMM covers the gate, selection can run per-row on GPU or CPU).
- Decode runner: V4ForwardRunner (the state handoff target).

## Hard gates (the acceptance contract)

1. `python3 Scripts/v4f/prefill_parity_gates.py` prints 0 parity
   failures: token-exact greedy match between serial and chunked
   prefill on france32, story57, long231 (CSA), hca296 (HCA).
2. Speedup printed by the same script; target >= 5x wall-time on the
   231-token case. Record the number.
3. Full serial suite green (637+ tests).
4. Chunk sizes 64 and 128 agree.

## Pitfalls (already cost debugging days — see design note)

- RoPE pairing: interleaved (2i, 2i+1) everywhere.
- Output de-rotation: inverse=true, POSITIVE position.
- O-projection: 8 group slices, not one GEMM.
- Norm gammas BF16 -> fp32 widening (share the runner's cache).
- head.weight BF16 -> widened once; head GEMV for the LAST row only.
- Indexer compressor rows are 256-dim.
- CSA group-0 overlap zero-init and the indexer QAT-sim are recorded
  divergences from the reference; do not change them in this task.
- Slot ownership: tiles schedule with avoidingSlots for the in-flight
  tile; never reuse a slot queued GPU work references.

## State handoff

CompressedKVCacheManager.position after the last chunk == prompt
token count. Decode starts at that position. The window ring and
compressed entries must match the serial path bit-for-bit (the parity
gate checks this through outputs).
