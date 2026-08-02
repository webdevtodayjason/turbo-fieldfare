# V4F-06c design note: chunked prefill runner for DeepSeek V4-Flash

Status: design, 2026-08-02. Depends on V4F-06a (pipelined decode
execution) and V4F-06b (grouped prefill MoE kernels + tile scheduler).
Write the runner only after both land and their reports are read.

## Goal

Replace serial per-token prompt feeding (~0.7 s/token) with chunked
prefill: up to 128 prompt tokens per chunk moving through the
43-layer stack layer-major, matching the Gemma prefill architecture
(layer-major bounded groups, no full-prompt expert activations held).

## Composition map

Per chunk, per layer, in order:

1. **mHC attn boundary (batched).** `v4b_hc_params` is per-token;
   prefill needs the batched variants. If 06b did not cover these,
   they are small kernels: params (24-mix from [rows, 4*dim] state),
   pre gather, post merge — all row-parallel.
2. **attn_norm (batched RMSNorm, fp32 in / fp16 out).** Exists in
   Gemma prefill primitives; adapt shapes.
3. **QKV projections (batched).** FP8 block GEMM (MPP staged or the
   grouped pattern from 06b): wq_a [1024,4096], wq_b [32768,1024],
   wkv [512,4096] per row, plus compressor projections (BF16: CSA
   1024-dim, HCA 512-dim rows) and indexer wq_b (FP8 [8192,1024],
   CSA only). Per-head norms + interleaved RoPE per row (batched
   variants of v4b kernels; convention: adjacent pairs (2i, 2i+1),
   compress theta 160000 + YaRN on ratio!=0 layers).
4. **Window KV writes** into the ring rows for the chunk (batched).
5. **Compressor group flushes.** Groups completing inside the chunk
   flush via the existing v4_csa_compress_group /
   v4b_hca_compress_group / v4c_indexer_compress_group kernels, one
   dispatch per completed group, reading the chunk's staged
   projection rows. Groups are 4 (CSA) or 128 (HCA) tokens; a
   128-token chunk completes up to 32 CSA groups and 1 HCA group.
   The CSA overlap needs the previous group's rows: retain the last
   4 rows of the previous chunk's projections (carry-over buffer).
6. **Attention.** Window MQA per row against the ring (batched
   prefill attention exists for Gemma; adapt to MQA 64x512 with
   sinks). CSA/HCA: sparse/dense attention over compressed entries +
   window, per row. The CSA indexer top-512 runs per row. This is
   the schedule-risk item from recon; decode-first ordering already
   absorbed the worst of it, but budget time here.
7. **Output de-rotation** (inverse=true, position = row's absolute
   position — NOT negated; see 8fffb1a) + **grouped o-projection**
   (8 groups, per-group slices; see 676731b) batched.
8. **mHC post (batched).**
9. **FFN boundary** (batched) + **router** for all rows (BF16 GEMM,
   sqrtsoftplus, static bias, norm-by-sum, x1.5, top-6).
10. **Grouped routed MoE** via the 06b driver: group token/expert
    pairs, stream expert tiles of <=8 with next-tile prefetch, fused
    FP4 SwiGLU GEMM, weighted reduce. Hash layers (0-2): ids from the
    resident tid2eid table rows for the chunk's tokens (one CPU
    gather), weights via the score-based kernel — no router.
11. **Shared expert** (FP8 GEMM batched) added as the phase-2
    residual, as in decode.
12. **mHC ffn post (batched).**

After the last chunk: the head boundary (pre-only hc_head), final
norm, FP32 head GEMV for the LAST row only (matches Gemma prefill).

## State handoff to decode

The CompressedKVCacheManager position after the chunk == number of
prompt tokens consumed; decode of the first new token continues with
position = prompt count. The window ring and compressed entries must
be identical to the serial path's state — the parity gate checks this
implicitly through token-exact outputs.

## Gates

1. **Parity (hard):** greedy outputs token-identical to the serial
   runner for: the 6-token France prompt, the 57-token story QA, and
   the 231-token story QA (exercises CSA), plus a >260-token prompt
   (exercises HCA). Chunk sizes 64 and 128 must agree.
2. **Perf:** prefill seconds for the 231-token prompt before/after;
   target >= 5x. Report the exact numbers.
3. **Regression:** full serial suite (637 tests) green.

## Recorded pitfalls (from the debug dossier)

- RoPE pairing is interleaved everywhere (2i, 2i+1). Any new RoPE
  call site must follow it; convention errors are invisible to
  same-position tests.
- Output de-rotation: inverse=true with POSITIVE position.
- O-projection is 8 separate group slices, not one GEMV.
- Norm gammas are BF16 and the norm kernels want fp32: reuse the
  runner's widening cache (gammaF32) or share an equivalent.
- head.weight is BF16, widened once — reuse the runner's headF32.
- Indexer compressor rows are 256-dim (two 128-dim series).
- The CSA group-0 overlap and indexer QAT-sim divergences from the
  reference are documented as minor; do not "fix" them silently —
  measure first.
