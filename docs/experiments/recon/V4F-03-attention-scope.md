# V4F-03 work order: attention and residual stack for DeepSeek V4-Flash

Status: recon complete, ready for implementation planning. Read-only survey of
the current attention/KV/RoPE stack against the published V4-Flash attention
facts (PORTING_PLAYBOOK.md, V4F-00 gate table). No runtime behavior recorded;
no source changed by this document.

V4-Flash attention recap (from V4F-00): hybrid CSA layers (4:1 KV compression
with learned softmax compression weights, lightning-indexer sparse top-512
block selection, shared-KV MQA, head_dim 512, 64 query heads, q_lora_rank
1024) interleaved 1:1 with HCA layers (128:1 compression, dense over
compressed entries, no indexer). Both add a 128-token sliding-window branch of
uncompressed KV merged into the same softmax, per-head attention sinks,
partial RoPE on the last 64 dims (queries/KV/outputs, outputs rotated at
negative position), per-head RMSNorm before core attention, and grouped output
projection (8 groups through o_lora_rank 1024). Residual stream is mHC with 4x
width. Layers 0, 1, 42 have compress_ratio 0 (variant pending sibling worker).

## 1. Reusable as-is (or with trivial re-specialization)

| Asset | File / symbol | Fit for V4-Flash |
| --- | --- | --- |
| Split-KV Flash-Decoding structure | `Kernels/Attention/Attention.swift` (`Attention.encodeSplit`, `splitGeometry`), `Metal/Attention/attention.metal` (`attention_decode_partial`, `attention_decode_combine`) | The two-pass online-softmax partial/combine structure is algorithm-exact for CSA/HCA decode: 640 attendable entries (512 sparse + 128 window + sinks) is a short, fixed-size range. Reuse the pass structure and the combine kernel nearly verbatim. |
| Online-softmax numerics | `attention.metal` header comments + `block_reduce_sum`, `attn_softmax_exp` | FP32 accumulators, Milakov-Gimelshein recurrence. Directly correct; only the gather source changes. |
| Combine pass | `attention_decode_combine` | Merge of (m, d, o) partials is agnostic to how partials were produced. Reusable unchanged once `numQHeads` ceilings move (see section 2). |
| SWA ring machinery | `KVCacheManager` (`fp16RingEnabled`, `ringCapacity`, `physicalSlot`, `ringStartSlot`), `attn_ring_slot` (function constant 69) | V4's 128-token uncompressed window branch IS the existing SWA ring with window=128. The ring-index mapping and wrap-aware kernel path port directly as the window sub-branch of the merged kernel. |
| Function-constant specialization pattern | `Attention.swift` `specializedPipeline` (indices 60-69), `RoPE.swift` (50-53), `FusedQKVEpilogue.swift` (82-86) | The per-shape PSO specialization idiom carries over; V4 needs new shapes (512/64/1), not a new mechanism. |
| Per-head RMSNorm primitives | `Kernels/Primitives/RMSNorm.swift`, `Metal/Primitives/rmsnorm.metal`; per-head usage inside `fused_qkv_epilogue` (rmsnorm_bf16w_perhead) | V4's per-head RMSNorm before core attention is the same primitive. Reusable as a standalone encode step or fused into the new epilogue. |
| Partial-RoPE skeleton | `Metal/Primitives/rope.metal` (`rope_proportional_neox`, `apply_neox_pair`), `RoPE.encodeProportionalNeox` | Already supports rotating a subset of pairs. Needs a variant (see section 2) but the NeoX pair math and dispatch structure are reusable. |
| Fused QKV GEMV pattern | `Kernels/Fusions/FusedQKVGEMV.swift`, `Metal/Quant/dequant_int4.metal` (`dequant_int4_qkv_gemv_simd`) | The pattern (one kernel, concatenated Q/K/V row ranges, shared x) survives, but the V4 projections are FP8 with LoRA structure (q_lora_rank 1024), so this is a template, not a drop-in. |
| Prefill tiled kernel structure | `Kernels/Attention/PrefillAttention.swift`, `attention_prefill_causal_tiled` | HCA dense-over-compressed prefill can reuse the tiled causal structure with new geometry (64 Q heads, 1 shared KV head, compressed-entry mask). CSA prefill cannot (sparse per-row gather). |
| KVView read contract | `KVCacheManager.swift` `KVView` | The "buffer + stride + validTokenCount" view the kernels bind is the right abstraction to keep for the compressed stores. |
| cb1/io/cb2 phase pipeline | `RealForwardRunner.swift` (decode layer loop, L1357-1492) | The phase split survives unchanged; see section 3. |

Not reusable: `KVCacheManager`'s core assumption of one K-slot and one V-slot
per token per layer (`kSlot`/`vSlot`, per-token `strides`). CSA writes one
compressed entry per 4 tokens; HCA one per 128. The whole allocation/stride
model changes. `maxQHeads = 16` and the partial-scratch sizing
(`Attention.maxQHeads * maxChunks * maxHeadDim` = 2 MB) must move to 64 heads
(8 MB at the same chunk cap). `kAttnMaxQPerKV = 2` GQA-grouped partial variant
is dead code for MQA (q_per_kv = 64); only the plain partial path applies.

## 2. New work

Ordered by dependency. Day estimates in section 5.

### 2.1 CompressedKVCacheManager (new, replaces KVCacheManager role for V4)

Per-layer stores:

- CSA layers: compressed-entry buffer, one entry per 4 tokens (entry = shared
  KV rep, head_dim 512, FP8/BF16 mixed per the checkpoint), plus a
  write-side accumulator gathering the 4-token group through the learned
  softmax compression weights (a tiny GEMV + softmax encode each token,
  flush on group completion).
- HCA layers: same shape, one entry per 128 tokens.
- All layers: a 128-token uncompressed FP16 window ring (reuse the ring
  indexing logic from `KVCacheManager` wholesale) and a per-head sink
  scalar/vector store.
- compress_ratio-0 layers (0, 1, 42): needs the sibling worker's variant
  read; design the manager with a `LayerKind` enum (`.csa`, `.hca`,
  `.passthrough`) so the third kind plugs in without reshaping the API.

Keep the `KVView` contract and the `reset()`/`MADV_DONTNEED` discipline.
Memory is a non-issue per V4F-00 (~2% of GQA-8 baseline), so allocate for
maxContext up front as today.

### 2.2 CSA lightning indexer

- Decode: indexer-score kernel (query vs all compressed block reps, ReLU
  activation, per-head weights aggregated), then a top-512 selection kernel
  writing a compacted index list + scores into a shared MTLBuffer. Entry
  count is bounded (maxContext/4), so single-kernel bitonic or
  per-simdgroup top-k with a second merge pass is fine; no unbounded scratch.
- Prefill: per query row in the chunk, scores against available blocks and a
  per-row top-512. Bounded [chunk x blocks] score scratch; chunked like the
  existing prefill. This is the largest prefill-side item.
- Indexer weights arrive FP4 (V4F-02 dequant dependency).

### 2.3 Sparse shared-KV MQA core attention (decode + chunked prefill)

- Decode: new `attention_csa_partial` / reuse `attention_decode_combine`.
  Partial pass iterates a gather list (index buffer from 2.2) instead of a
  contiguous range; same online softmax. A second sub-range covers the
  128-token window ring (reuse `attn_ring_slot` logic). Sinks fold in as
  extra entries at the head of the list (or analytically in the combine:
  add sink logit to `m`/`d` accumulation, mirroring how `d_run`/`m_run`
  fold in). One combined kernel that walks [sinks | sparse blocks | window]
  in one recurrence is preferable to three kernels: one pass, one softmax,
  no merge kernel.
- Grid: 64 Q heads x chunks over a 640-entry range. Bump `maxQHeads` to 64
  and resize partial scratch accordingly. MQA: numKVHeads = 1, head_dim 512
  (already the kernel's scratch ceiling).
- HCA decode: same kernel with the gather list = all compressed entries
  (dense); can be a function-constant specialization.
- Chunked prefill: new sparse prefill kernel (per-row gather list from the
  prefill indexer). HCA prefill can adapt `attention_prefill_causal_tiled`
  with a compressed-entry visibility mask. Group boundaries (compression
  flushes mid-chunk) must be handled in the cache write path, not the
  attention kernel.

### 2.4 Sink/window merge

Covered inside 2.3 if the single-recurrence design is taken. The window
boundary is the classic off-by-one site: window = last 128 *uncompressed*
tokens, disjoint from what the compressed entries already cover. The cache
manager must expose the exact logical-position coverage of compressed
entries so the kernel never double-counts a position.

### 2.5 RoPE variants

- Last-64-dims partial RoPE: `rope_proportional_neox` rotates pairs at
  `[pair, half_dim + pair)` measured from dim 0. V4 rotates a dedicated
  trailing 64-dim slice (32 pairs internal to that slice). New variant
  kernel with a dim-offset parameter, or a function constant. Small.
- Negative-position output de-rotation: `apply_neox_pair` already takes a
  `float position`, but both kernels bind `constant uint& position`. New
  variant taking a signed/float position applied to the attention output's
  rope slice before the grouped o-proj. Small but a high-risk correctness
  point (sign/order); needs a dedicated two-token delta test against
  reference.

### 2.6 mHC layer-boundary kernels

- Residual stream becomes 4 x 4096. `hidden`, `normed`, and every
  residual-add site in the layer loop change shape.
- New kernels: dynamic Sinkhorn (20 iterations on per-layer 4x4 matrices,
  trivially one threadgroup), and the 4-way residual mix at each sublayer
  boundary (attention out, FFN out). Cheap math but touches every layer
  boundary, so it lands inside `RealForwardRunner`'s layer loop and the
  fused post-attention / layer-tail kernels.

### 2.7 Grouped output projection

8 groups through o_lora_rank 1024 then to hidden 4096. A grouped GEMV with
FP8 weights (V4F-02 dependency). Structurally a new shape on the existing
GEMV pattern; small once FP8 dequant exists.

### 2.8 Q/KV LoRA projection path

q_lora_rank 1024 down-projection + up-projection, shared-KV projection with
the compression-weight path. Replaces `FusedQKVGEMV`/`FusedQKVEpilogue`
roles for V4 layers: a new fused epilogue doing per-head RMSNorm + trailing
-slice RoPE + cache write in one kernel, mirroring `fused_qkv_epilogue`'s
structure.

## 3. Decode-phase structure: does CSA change the cb1/cb2 handoff?

No. The split exists for exactly one reason, stated in
`RealForwardRunner.swift` L1358-1360: "the only reason to break is the CPU
readback of router indices needed to issue I/O for the routed-expert blobs."
The CPU must see expert IDs because they drive `pread` file I/O.

CSA top-512 block selection drives no I/O. It selects among GPU-resident
compressed cache entries. The indexer output (index list + scores) stays in
a shared MTLBuffer written by the indexer kernel and consumed by the sparse
attention kernel inside the same `cb1` command buffer; Metal hazard-tracking
orders them. The gather list has a fixed maximum size (512 blocks + 128
window + sinks = bounded entries), so split-KV chunk geometry is static —
unlike the router, not even a count needs to come back to the CPU.

Consequences:

- The indexer and sparse attention slot into `cb1` between the QKV epilogue
  and the router, adding two kernels but no synchronization point.
- The cb1/io/cb2 overlap model (SYSTEM_DESIGN.md Decode table) is preserved
  verbatim: expert streaming remains the only mid-layer CPU handoff.
- CSA decode is *cheaper* than Gemma full attention on the same context:
  ~640 shared-KV entries vs seq_len GQA entries. Attention stops being the
  long-pole it can be at long context, which strengthens the
  SSD-bound-decode thesis from V4F-00.
- One caveat: if a future optimization wants the CPU to prune the index
  list (e.g., to skip uploading KV pages), that would introduce a second
  readback and a second handoff. Do not do this; the playbook's
  demand-page principle applies to ourselves.

## 4. Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Indexer in FP4: score error flips top-512 membership; a selection mismatch is a discrete divergence that no output-tolerance gate can absorb smoothly | High | Validate selection recall vs the reference implementation (fraction of matching blocks on fixed vectors) BEFORE any output-tolerance gate. If recall is low, keep indexer scores in FP8/BF16 even if weights stay FP4. |
| Numerical tolerances in compressed KV: FP8/BF16 mixed entries feed 512-wide dot products; error compounds through the online softmax | High | Per-layer activation comparison against reference (V4F-03 gate) at multiple context lengths, not just short prompts. Keep FP32 accumulators everywhere (current kernels already do). `fast::exp` in `attention.metal` may need `precise::exp` for the gate; measure both. |
| Sink behavior: per-head sink logits merged into the same softmax; wrong dtype or wrong merge point shifts every output | Medium | Fold sinks into the (m, d) recurrence analytically; unit-test against reference with sink-only and window-only configurations. |
| Window/compressed coverage overlap: double-counting a logical position (compressed entry + window both covering it) is a silent correctness bug | Medium | Cache manager exposes exact coverage; kernel asserts disjointness in debug builds. |
| Negative-position output RoPE: sign/order error relativizes outputs wrong; only visible as subtle quality loss | Medium | Dedicated two-token delta test against reference activations before integration. |
| compress_ratio-0 layers 0/1/42 unknown variant | Blocker (external) | Sibling worker must read the official inference code; cache manager's `.passthrough` kind is the integration seam. |
| Scratch ceilings: 64 Q heads breaks `maxQHeads = 16` partial scratch and any kernel authored around 16-head dispatch | Low | Mechanical constant bump + scratch realloc (2 MB -> 8 MB), called out so it is not discovered mid-integration. |
| Prefill compression-group boundaries mid-chunk | Medium | Handle in cache write path (flush partial groups at chunk edges); add a boundary-case test at chunk edges. |

## 5. Rough effort estimate (serial days, one engineer)

| Item | Days |
| --- | --- |
| 2.1 CompressedKVCacheManager (CSA + HCA stores, window ring, sinks, reset discipline) | 3 |
| 2.2 CSA lightning indexer (decode scores + top-512; prefill per-row top-k) | 4 |
| 2.3 Sparse MQA core attention (decode gather kernel + combine reuse; chunked sparse prefill) | 6 |
| 2.4 Sink/window merge (folded into 2.3 design + tests) | 1 |
| 2.5 RoPE variants (trailing-slice offset, negative-position output) | 2 |
| 2.6 mHC layer-boundary kernels (Sinkhorn, 4x stream mix, runner surgery) | 3 |
| 2.7 Grouped o-projection (after V4F-02 FP8 GEMV) | 1.5 |
| 2.8 Q/KV LoRA projection + fused epilogue (RMSNorm + rope + cache write) | 2.5 |
| Integration + layer-level validation vs reference activations (V4F-03 gate) | 5 |
| **Total** | **~28 days** |

Dependencies: V4F-02 (FP8/FP4 dequant) must land before 2.2 integration and
2.7; the sibling worker's layer-0/1/42 read must land before 2.1 finalizes.
Prefill-side sparse work (parts of 2.2 and 2.3) is the schedule risk; decode
-first sequencing gets an end-to-end single-token path roughly 3 weeks in.

## Summary (10 lines)

1. The split-KV online-softmax decode structure (partial + combine) is reusable nearly verbatim for CSA/HCA; only the KV source changes from contiguous range to gather list.
2. The existing SWA ring (KVCacheManager + attn_ring_slot) maps directly onto V4's 128-token uncompressed window branch.
3. KVCacheManager itself is not reusable: compressed entries are fractional-per-token, so a new CompressedKVCacheManager (CSA/HCA stores + window ring + sinks) is required.
4. RoPE needs two small variants (trailing-64-dim offset, negative-position output de-rotation); the pair math carries over.
5. Per-head RMSNorm primitives already exist and are reusable; the fused QKV GEMV/epilogue survive only as structural templates (V4 is FP8 LoRA, not int4).
6. cb1/cb2 is unchanged: CSA top-512 selection consumes GPU-resident cache, drives no I/O, and needs no CPU readback, unlike the router.
7. Biggest new kernels: lightning indexer (decode + prefill top-512), sparse shared-KV MQA attention with sink/window folded into one softmax recurrence, and mHC 4x-residual boundary kernels.
8. Top risks: FP4 indexer flipping discrete block selection (gate on selection recall first), FP8/BF16 compressed-KV numerics, and window/compressed coverage double-counting.
9. Rough effort: ~28 serial days, decode-first sequencing; prefill sparse path is the schedule risk; layers 0/1/42 blocked on sibling recon.
10. V4F-03 is the largest work item in the series as predicted, but the current stack's structure (not its shapes) is a genuine asset: pass structure, ring logic, specialization idiom, and phase pipeline all carry.
