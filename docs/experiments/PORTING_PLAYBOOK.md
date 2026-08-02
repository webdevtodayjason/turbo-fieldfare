# Porting playbook: streaming-MoE experiments beyond Gemma 4

Status: proposal. This document records no runtime behavior and changes no
defaults. It distills the methodology behind the
[103-experiment inventory](EXPERIMENT_INVENTORY.md) into a reusable screen and
an experiment series for porting the streaming-expert architecture to other
models. Each port attempt follows the same evidence rules as the original
inventory: hypothesis, variants, measured evidence, holdout gate, disposition,
lesson.

## Recorded priority decision (2026-08-01)

PORT-V4F is the lead thread. The operator's goal is V4-Flash running in this
runtime on the 128 GB Mac, independent of swarm economics. FLEET-SWARM's
FS-01 quality gate therefore does **not** gate V4F kernel work; it gates only
which swarm roles a ported model may hold. The port stands alone as the
objective; swarms are a downstream consumer of it.

## The reusable methodology

Six principles carried the Gemma 4 port. They are model-independent and are
the actual asset being reused.

1. **Exploit sparsity, not size.** Total parameters are marketing. Activated
   parameters per token set the I/O floor. The runtime pays for
   `active_params x bytes_per_param x miss_rate` per token, streamed from SSD.
2. **Split weights into resident and streamed sets.** Map the small common set
   read-only and wrap it in Metal buffers without copies. Pack the large
   routed set as fixed-stride, page-aligned blobs in per-layer files so any
   expert is one bounded `pread` away.
3. **Demand-page, do not predict.** CACHE-05 and CACHE-06 showed cross-layer
   prediction (7% hit) and Markov prefetch (throughput collapse) both lose to a
   plain LFU slot cache. Routing is close enough to unpredictable that
   speculative I/O only contends with the demand path.
4. **Overlap I/O with resident compute.** The shared expert and cached routed
   experts run on the GPU while misses stream in. The cb1/io/cb2 phase split
   exists to create that overlap, and it survives independent of model family.
5. **Gate every policy on holdouts, end to end.** Trace-trained layouts
   (CACHE-04, CACHE-08) won on training-like traces and lost on holdouts.
   Every cache, layout, or prefetch decision needs a multi-workload,
   end-to-end tok/s and I/O gate, not a simulation win.
6. **Report memory-funded speed separately.** CACHE-03: 24/32 slots buy real
   hit-rate points with real memory. A result that spends more RAM is a
   different result, not a better one.

## The feasibility screen

Apply before writing any code. A model is a streaming port candidate when:

| Criterion | Why it binds | Gemma 4 reference |
| --- | --- | --- |
| Activated params per token <= ~10-15B at 4-bit | Sets worst-case GB/token read from SSD against ~5-7 GB/s NVMe | 4B active |
| Routed experts per layer <= ~256 | A 16-32 slot cache must hold a meaningful fraction of the pool | 128 |
| Expert blob is a fixed-size contiguous layout | Enables fixed-stride packing and one-`pread` fetch | 3.36 MB blobs |
| Existing lossless 4-bit-class quantization | The repacker copies quantized values; it never requantizes | MLX affine int4 |
| Attention is a known family (GQA, SWA, MLA) | Attention kernels are the largest per-model cost | hybrid SWA/full GQA |
| Resident common set fits the RAM budget | Common weights are always live | 1.35 GB file |

## Screen outcomes recorded

| Model | Active / total | Experts per layer | Screen result | Deciding factor |
| --- | --- | --- | --- | --- |
| Gemma 4 26B-A4B | 4B / 26B | 128 | Ported (this repo) | Reference implementation |
| DeepSeek V4-Flash | 13B / 284B | fine-grained, V3-style | Candidate; see PORT-V4F below | Sparsity 4.6% is excellent; CSA/HCA attention and FP4/FP8 quants are new kernel work |
| DeepSeek V4-Pro | 49B / 1.6T | fine-grained | Rejected at screen | ~25 GB/token cold-miss traffic; sub-1 tok/s before compute |
| Kimi K3 | 104B / 2.8T | 896 | Rejected at screen | 104B active is 26x reference; 16 slots cover 1.8% of the expert pool, collapsing hit rate; KDA + Gated MLA + AttnRes + LatentMoE are four new mechanisms |
| Qwen3-30B-A3B class | 3B / 30B | 128 | Candidate; lowest-effort port | SwiGLU MoE, standard attention, existing 4-bit quants |

A screen rejection is a measurement-based disposition, not a verdict on the
model. Re-screen if a smaller variant, a better quant, or faster storage
changes the inputs.

## PORT-V4F: DeepSeek V4-Flash experiment series

Target: V4-Flash (284B total, 13B active, 1M context published, FP4 experts +
FP8 mixed, MIT license) decoding interactively (>= 2 tok/s) on Apple Silicon
with routed experts streamed from SSD and a resident budget a 16-24 GB machine
can hold. Each stage is gated; a failed gate stops the series.

### V4F-00: Architecture recon and screen confirmation

- **Hypothesis:** Published V4-Flash architecture details (expert count,
  expert FFN width, CSA/HCA structure, mHC residual form, quant scheme) map
  onto the fixed-stride streaming layout without per-expert size variance.
- **Method:** Read the technical report and weight index; compute per-expert
  blob size, per-token cold-miss traffic, and resident-set estimate.
- **Gate:** Cold-miss traffic <= ~6 GB/token and resident set <= ~8 GB at
  published precision. Record actual expert count and geometry in this doc.
- **Failure action:** If expert blobs vary in size or routing is not
  per-token top-k, revise the layout plan before any kernel work.

**Result (2026-08-01): GATE PASSED.** Sources: arXiv 2606.19348v1 and
`deepseek-ai/DeepSeek-V4-Flash` config.json.

Measured geometry (V4-Flash):

| Property | Value | Consequence for the port |
| --- | --- | --- |
| Layers | 43 (+1 MTP) | Deeper than Gemma's 30; layer loop unchanged |
| Hidden size | 4096 | Kernel shape change, no structural change |
| Routed experts | 256 per layer, uniform | Fixed-stride packing works |
| Experts per token | 6 (+1 shared) | Fewer blobs per token than Gemma's 8 |
| Expert FFN | SwiGLU (silu), intermediate 2048 | Simpler than Gemma's GeGLU; per-expert blob = 3 x 4096 x 2048 params |
| Expert precision | FP4 native (quant-aware trained) | ~12.6 MB blob; no requantization needed, FP4 dequant kernel required |
| Non-expert precision | FP8 e4m3, 128x128 blocks, ue8m0 scales | New dequant path; ~8 GB resident common set |
| Routed expert total | ~139 GB on disk | Exceeds 128 GB RAM with common set: streaming is required, and a ~100 GB expert cache makes most of the pool resident |
| Cold-miss traffic | 6 x 43 x 12.6 MB = 3.25 GB/token | Within the 6 GB gate |
| First 3 MoE layers | Hash routing by token ID | Deterministic: perfectly prefetchable, free cache hits |
| Attention | Hybrid CSA (compress 4:1 + sparse top-512 via lightning indexer) interleaved 1:1 with HCA (compress 128:1 dense) | The real kernel work; MQA shared-KV, head_dim 512, 64 query heads, q_lora_rank 1024 |
| Attention extras | Sliding-window branch (128), attention sinks, partial RoPE (last 64 dims, incl. negative-position output de-rotation), grouped output projection (8 groups, rank 1024) | All small kernels, all required for correctness |
| Residuals | mHC: 4x residual stream, dynamic Sinkhorn (20 iters) on 4x4 matrices | Cheap per-layer math, but touches every layer boundary |
| KV cache | Compressed + FP8/BF16 mixed; ~2% of GQA-8 BF16 baseline at 1M | KV memory is a non-issue at our context targets |
| Vocab / embeddings | 129280, untied | ~1 GB at FP8; head runs as GEMV |
| Reference implementation | Official PyTorch inference code on the HF repo | Validation oracle for V4F-02/03 |

Open questions resolved by the reference-implementation recon
([notes](recon/V4F-reference-notes.md)): `compress_ratios` 0 for layers 0, 1,
and 42 means those layers run a pure 128-token sliding-window MQA (base theta
10000, no YaRN), which maps directly onto this runtime's existing SWA ring
machinery. The FP4 expert format is e2m1 packed low-nibble-first along K with
e8m0 scales per 32 elements; FP8 is e4m3 with ue8m0 scales per 128x128 block.
The noaux_tc router bias is fully static at inference (no update code exists
in the reference), and hash routing is a fixed [129280, 6] lookup table, so
the first three layers need no router kernel at all.

Disposition notes for later stages:

- The 128 GB Mac plan is "mostly resident + streamed tail": ~8 GB common,
  ~100 GB expert cache (~70% pool coverage), ~1 GB/token SSD traffic at
  steady state, estimating 5-7 tok/s decode before compute overlap.
- A 16-slot-per-layer cache covers only 6.25% of this 256-expert pool
  (versus 12.5% on Gemma). V4F-04 should evaluate a global expert pool
  rather than per-layer slots, since this host's budget is capacity-rich.
- The MTP module is ignorable for v1 and is a future speculative-decode
  lever (V4F-06 candidate).

### V4F-01: Repacker and layout

- **Hypothesis:** The bounded-range streaming repack generalizes to the V4
  weight index with a new tensor-name mapping and a new quant metadata
  descriptor.
- **Method:** Extend the range planner; copy FP4/FP8 payloads unchanged;
  write per-layer expert files at a page-aligned fixed stride.
- **Gate:** Byte-identical payload round-trip on a sample of ranges, verified
  by digest, with the same 512 KB heap bound as IO-10.

### V4F-02: Quant kernels

- **Hypothesis:** FP4-expert and FP8-shared dequant can feed the existing
  GEMV/QMM structure at parity with reference outputs.
- **Method:** New Metal dequant paths only; validate against the reference
  implementation per tensor before any end-to-end run.
- **Gate:** Per-tensor output within reference tolerance on a fixed test
  vector set. Exact identity not required, per the existing invariants.

### V4F-03: Attention and residual stack

- **Hypothesis:** CSA/HCA and mHC can be implemented as bounded prefill and
  single-token decode kernels without unbounded scratch.
- **Method:** Largest work item in the series. Implement, then validate layer
  outputs against reference activations.
- **Gate:** Layer-level output tolerance; KV/state memory bounded at the
  target context lengths.

### V4F-04: Streaming integration

- **Hypothesis:** The 16-slot LFU demand-paged cache with cb1/io/cb2 overlap
  achieves >= 60% expert hit rate on V4-Flash routing without modification.
- **Method:** Wire `PreadExpertStreamer` and the phase pipeline unchanged;
  measure hit rate, I/O ms/token, tok/s on the canonical prompts.
- **Gate:** >= 2 tok/s decode on the target host with the resident budget
  from V4F-00.
- **Note:** Do not tune slot count or prefetch before the baseline gate.
  CACHE-01 through CACHE-08 are the prior art for every tuning idea; rerun
  them against V4-Flash traces only after the baseline is measured.

### V4F recon synthesis (2026-08-01, four-worker swarm)

Work orders delivered: [V4F-01 repack](recon/V4F-01-repack-workorder.md),
[V4F-02 quant kernels](recon/V4F-02-quant-kernels.md),
[V4F-03 attention](recon/V4F-03-attention-scope.md), plus the
[reference-implementation notes](recon/V4F-reference-notes.md) they share.

| Stage | Estimate | Largest items |
| --- | --- | --- |
| V4F-01 repacker | 7-11.5 days | Planner fork (2-3d), schema versioning (1-2d) |
| V4F-02 quant kernels | 18-19 days | Fixture harness, fused SwiGLU MoE rework |
| V4F-03 attention | ~28 days | Compressed KV manager, CSA indexer, sparse prefill (schedule risk) |
| **Total serial** | **~53-59 engineer-days** | Plus V4F-04 integration and V4F-05 validation |

Implementation status (branch `v4f`, 2026-08-01):

- **V4F-01 landed** (`2e47147`): dual config-family parsing, dtype
  extension, per-expert planner fork (page-aligned 13,369,344-byte
  6-slice blobs), family-versioned manifest (Gemma byte-identical), MTP
  dropped and audited. 5/5 round-trip + 39/39 pre-existing repack tests.
- **V4F-02 decode kernels landed** (`06b84f5`): FP4 e2m1+ue8m0 GEMV, FP8
  e4m3 block GEMV, FP8 embed lookup, BF16 sqrtsoftplus router (static
  bias, k=6), fused clamped-SwiGLU MoE. 22/22 GPU fixture checks.
- **V4F-03 foundations landed** (`2aadeec`): CompressedKVCacheManager,
  merged sparse+window MQA decode, CSA compressor, lightning indexer.
  **Indexer recall gate: exact** at 600/3000/20000 blocks; CSA decode
  chain matches CPU reference. 49/49 V4 tests pass.
- Follow-up queue for wave 2: MetalContext registration of the new
  shader modules, RoPE variants, mHC boundary kernels, grouped
  o-projection, Q/KV LoRA epilogue, HCA compressor kernel, prefill
  paths, tokenizer/DSML chat format, then V4F-04 runtime wiring.

Wave 2+ status update (2026-08-01/02, branch `v4f`):

- Wave 2 landed: RoPE/mHC/HCA/o-proj kernels (`2e26285`), tokenizer and
  DSML chat format (`4a5f161`), MetalContext registration and the V4
  family arch gate + V4Model (`a4b23fa`, `2208ab9`).
- **V4F-04 decode runner landed** (`c418240`): `V4ForwardRunner`
  composes all kernels into the per-token graph with cb1/io/cb2 phasing.
  v1 serializes layers to preserve slot ownership; cross-layer
  pipelining is the recorded follow-up. Coordinator-written after three
  worker deaths (Kimi K3 API stream timeouts, not task failures).
- **Option C gate PASSED** (`scripts/v4f/make_v4f_golden.py` +
  `Tests/.../V4RealCheckpointValidationTests.swift`): GPU kernels vs
  independent numpy goldens on 61.6 MB of real shard-7 bytes. FP4/FP8
  GEMVs at the fp16 output floor (max rel err 4e-4), router top-6
  selection exact, fused MoE within 4e-4, embed lookup exact. Format
  interpretation (low-nibble e2m1, ue8m0 scales, OCP e4m3) confirmed
  against real weights. Full report: `scratch/v4f-recon/validation-report.md`.
- First real-download attempt caught a planner-adjacent bug: the
  published config carries 44 compress_ratios (43 layers + 1 MTP).
  Fixed in `6dd6b66`.
- Full ~155 GB repack launched 2026-08-02 (background, resumable).
  Next gates: first-token validation vs the reference, then V4F-05
  holdouts. Prefill (V4F-06) and cross-layer pipelining remain open.

First-token debugging dossier (2026-08-02, branch `v4f`):

- Repack completed in 75 minutes; install verified.
- First-run catches fixed en route: layout.json 16 MB metadata cap
  (V4 layout is 17.1 MB; raised to 64 MB), tokenizer wrapper requiring
  Gemma-only specials (added BOS/EOS-only V4 init + `loadV4`), chunked
  prefill assumption (serial `.off` for the decode-only runner),
  compress_ratios 44-entry config (43 layers + 1 MTP; layer 42 is CSA,
  correcting a recon-era mask), resident 16-byte alignment for the
  fp32 GEMV (staged widening + planner 16-pad).
- Five real-checkpoint dtype corrections (verified against shard
  headers): head.weight BF16 (widened at init), compressor wkv/wgate
  BF16 (runner-owned BF16 projections), indexer compressor rows
  256-dim, norm gammas BF16 (widened, cached), wo_a F8_E4M3 (two-stage
  FP8 GEMV).
- **First end-to-end generation achieved** (`1b2589e`): coherent
  English for the first several tokens, then drift into repetition.
- MoE exonerated by offline ground truth: dumped live xNorm/blobs/acts
  and recomputed in numpy — CPU act rms 4.6171 vs kernel 4.6174,
  row-level match 3e-5. Two probe-side bugs caused a false "29x
  anomaly": a scale-grouping error in the Swift probe and a
  synthetic-random-x numpy ground truth. Real xNorm drives real
  experts much harder than random x.
- Open: generation drift. Suspects in order: (1) window-KV
  composition (ring slot, position, sinks) for generated tokens,
  (2) RoPE/output de-rotation composition, (3) logits head path,
  (4) mHC boundary details. Evidence: BOS stream explodes (4 -> 8000
  rms) while normal tokens stay small (~0.17), consistent with
  massive-activation literature and possibly all correct.
- **Drift RESOLVED (2026-08-02, diagnosis worker):** two runner
  composition bugs, both in `V4ForwardRunner.forward`, both
  per-token uniform (kernels and cache manager exonerated):
  (a) output de-rotation passed `position: -position` AND
  `inverse: true` to `v4b_rope_trailing`, whose inverse flag already
  conjugates — the double negation re-rotated the attention output
  at +p instead of de-rotating (reference:
  `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` at the query
  position). Fixed to positive position + inverse.
  (b) the grouped o-projection ran wo_a F8 [8192, 4096] as ONE plain
  GEMV over `attnOut[0..4096]`; the reference views o as
  [8, 4096] and dots group g's 1024 rows with slice g
  (`einsum("bsgd,grd->bsgr")`). Rows 1024..8191 (7/8 of the low-rank
  stage) consumed the wrong input slice. Fixed by looping 8 groups
  with per-group weight/scale/x/y offsets (kernel unchanged).
  Also fixed a pre-existing TURBO_V4_DEBUG probe that encoded into
  an already-committed command buffer (crashed debug runs at L00).
  Verification: greedy "The capital of France is" went from
  " the capital of the... ... ... ...package ..." to
  " Paris, the city of love, the city of lights, ..."; a 25-token
  prompt yields " Paris. The capital of France is Paris. ... The
  capital of Germany is Berlin." (no position-dependent breakage).
  122/122 filtered tests green. Short-context note: compressed
  entries are never attended below 129 tokens (visibleGroupCount
  gates on windowStart > 0), so CSA/HCA compressor details cannot
  cause short-generation drift; the group-0 overlap zero-vs-(-inf)
  gate-score difference vs the reference remains a long-context
  follow-up.
- Debug instrumentation is env-gated (TURBO_V4_DEBUG) in
  V4ForwardRunner.swift (`65a264c`); remove with the debug session.
- `65a264c` also notes the debug dumps in `scratch/v4f-recon/`
  (dbg-xNorm.bin, dbg-acts.bin, dbg-blob0/1.bin) for continuation.

De-risking findings:

- **cb1/cb2 survives.** CSA top-512 selection touches only GPU-resident
  cache state, so no new GPU-CPU-GPU handoff; the streaming phase split is
  untouched.
- **Reusable as-is:** split-KV decode attention structure, the SWA ring
  (1:1 onto the 128-token window branch and the ratio-0 layers), per-head
  RMSNorm, function-constant specialization idiom.
- **Repacker mostly transfers.** Everything below the source/planner seams
  never inspects tensor semantics; the work concentrates in four
  Gemma-shaped seams (source gating, MLX quant metadata, arch gate, tensor
  naming).
- **Alignment contract carried forward:** pad FP4 sub-tensors to 4 bytes so
  packed expert blobs keep uint-load eligibility.

Recorded top risks, in order:

1. **FP4 indexer selection recall.** FP4 lightning-indexer math can flip
   discrete top-512 block membership. Gate on selection recall against the
   reference before any output-tolerance comparison.
2. **Sparse prefill schedule risk** in V4F-03; decode-first sequencing
   mitigates.
3. ~~**FP4 on-disk container dtype and stacked-vs-per-expert source layout**~~
   RESOLVED 2026-08-01 from the live HF index and shard-7 header
   (`scratch/v4f-recon/`): experts ship as **per-expert tensors**
   `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}`; weights are **I8
   containers** packing e2m1 pairs along K (w1/w3: [2048, 2048], w2:
   [4096, 1024]) with **F8_E8M0** scales per 32 elements (w1/w3:
   [2048, 128]). Router `gate.weight` is BF16 with an F32 static bias;
   dense projections are F8_E4M3 with F8_E8M0 scales per 128x128 block.
   Checkpoint total 159.6 GB across 46 shards. Per-expert blob with
   scales is ~13.4 MB.
4. **MoE plumbing generalization:** `MoE.swift` hardcodes topK=8 and a
   9-field `MoEExpertOffsets`; V4 needs 6 slots, 6 fields, no biases.

### V4F-05: Holdout validation

- **Hypothesis:** Baseline results generalize beyond the tuning prompts.
- **Method:** Multi-workload holdout (short, medium, long-context, tool-use)
  per the community benchmark protocol.
- **Disposition rule:** Any optimization that wins on tuning prompts and loses
  on holdouts is recorded Rejected, following CACHE-04 precedent.

## PORT-SPARK: same architecture on DGX Spark (Grace Blackwell)

Target: carry the streaming-MoE architecture to NVIDIA DGX Spark (GB10 Grace
Blackwell, 128 GB unified LPDDR5x, about 273 GB/s memory bandwidth, NVMe
SSD, CUDA-native, FP4 tensor cores). Same shape, different platform API.

### What the platform change does to the thesis

The technique transfers, but the constraint that justifies it moves:

| Factor | Apple Silicon 8 GB Mac | DGX Spark |
| --- | --- | --- |
| Binding constraint | RAM capacity (model is 14.3 GB, RAM is 8 GB) | Memory bandwidth (about 273 GB/s shared) |
| Unified memory access | `makeBuffer(bytesNoCopy:)` on shared pages | Pinned/mapped host memory over C2C, GPU reads directly |
| Streaming API | `pread` into aligned slots | `pread`/`io_uring` into pinned buffers, or CUDA managed memory |
| Decode ceiling for 13B active at 4-bit | SSD-bound, a few tok/s | Bandwidth-bound, ~40 tok/s theoretical, less in practice |
| Value band | Models that do not fit in 8 GB | Models that do not fit in 128 GB |

Two consequences:

1. **Gemma-4-class models need no port.** The whole 14.3 GB `.gturbo` fits
   resident in 128 GB. Stock vLLM or llama.cpp on Spark already wins. The
   streaming architecture only earns its complexity above the resident
   budget.
2. **The interesting band on Spark is ~130 GB to ~1 TB artifacts:**
   V4-Flash (about 140 GB at FP4/FP8, barely over budget, so a small
   streamed tail on a mostly resident model), V4-Pro, K2/K3-class quants.
   The feasibility screen still applies, and it still rejects the
   high-active-parameter end: K3 at 104B active has a bandwidth ceiling of
   about 5 tok/s on Spark even fully resident, before SSD streaming makes it
   worse. V4-Flash at 13B active is the band's best citizen again.

### What transfers unchanged

- The six methodology principles above, verbatim.
- The `.gturbo`-style layout: fixed-stride page-aligned expert blobs,
  per-layer files, manifest-verified install. Filesystem truths are
  platform-independent.
- The LFU slot cache and cb1/io/cb2 overlap structure, re-expressed as
  pinned host buffers plus async CUDA streams instead of Metal command
  buffers.
- The experiment discipline and gates.

### What deliberately does not transfer

- The Metal kernels. Do not port them. The CUDA ecosystem (vLLM,
  TensorRT-LLM, llama.cpp) already has better Blackwell kernels, including
  native FP4 paths, for the attention and MoE families that matter. The port
  contributes the streaming layer those systems lack, not another kernel
  stack. vLLM today assumes resident weights; the contribution is a
  slot-cached demand-paged expert provider underneath an existing CUDA
  backend.

### Experiment series

- **SPARK-00: Recon.** Hypothesis: pinned-host expert reads over C2C sustain
  enough bandwidth that SSD, not the interconnect, remains the bottleneck.
  Method: microbenchmark `pread` into pinned buffers plus concurrent GPU
  reads of those buffers. Gate: streaming path sustains >= 80% of raw NVMe
  read bandwidth while GPU compute overlaps. Also measure CUDA managed
  memory oversubscription as the zero-code baseline; if UVM paging alone
  meets the gate, record that and descope SPARK-01.
- **SPARK-01: Streaming core.** Hypothesis: the layout + LFU slot cache +
  parallel bounded reads port to C++/CUDA in days, since they contain no
  Metal. Gate: replay Gemma 4 and V4-Flash routing traces at parity hit
  rates with the Swift implementation.
- **SPARK-02: Backend integration.** Hypothesis: a CUDA MoE backend can
  consume slot buffers directly with zero copy, preserving the cb1/io/cb2
  overlap via CUDA streams and events. Gate: V4-Flash decode at >= 10 tok/s
  with <= 100 GB resident.
- **SPARK-03: Holdout validation.** Same multi-workload gate as V4F-05,
  plus the standing rule: memory-funded speed (bigger slot pools toward
  128 GB) is reported separately from memory-free speed.

### Recorded risks

- CUDA managed-memory paging on Grace Blackwell may already approximate the
  slot cache for some workloads; SPARK-00 must measure it before building
  anything, per the demand-page-do-not-predict principle applied to
  ourselves.
- vLLM integration depth may force kernel-level surgery; llama.cpp's CUDA
  backend is the fallback integration target.
- Spark's 273 GB/s makes every high-active-parameter rejection in the
  screen table harsher, not milder. The platform change enlarges capacity,
  not per-token weight bandwidth.

## Recorded hardware fleet and per-host screen

Fleet as reported 2026-08-01. Bandwidth figures are published chip specs;
the 128 GB machine's chip is unconfirmed, so its ceiling is a range.

| Host | RAM | Approx. bandwidth | Role from the screen |
| --- | ---: | ---: | --- |
| This Mac | 128 GB | ~410-546 GB/s if M3/M4 Max class | Streaming flagship: runs > 100 GB models that just miss resident |
| M3 Ultra Mac Studio | 256 GB | ~800 GB/s | Fully-resident host for the <= ~200 GB band |
| M3 Pro | 48 GB | ~150 GB/s | Fully-resident host for the <= ~30 GB band |
| M3 Pro (second) | 48 GB | ~150 GB/s | Duplicate of the above; A/B and holdout runs |

Per-host dispositions, applying the screen with real numbers:

| Model | 48 GB M3 Pro | 128 GB Mac | 256 GB M3 Ultra |
| --- | --- | --- | --- |
| Gemma 4 26B-A4B (~14.3 GB at 4-bit) | Fully resident; stock tooling wins, streaming unneeded | Fully resident | Fully resident |
| Qwen3-30B-A3B class (~17 GB at 4-bit) | Fully resident | Fully resident | Fully resident |
| DeepSeek V4-Flash (~140 GB at FP4/FP8) | Does not fit | **Streaming port target: mostly resident, small expert tail streamed** | Fully resident; stock vLLM/MLX wins |
| DeepSeek V4-Pro (~800 GB) | No | No | Streaming still rejected: 49B active puts I/O floor at ~25 GB/token cold |
| Kimi K3 (~500 GB at ~1.5-bit) | No | No | Still rejected; see below |

### K3 on the 256 GB Studio, quantified

The rejection survives the bigger host, and now has numbers. K3's routed
experts total roughly 1.25 TB at MXFP4 (about 15 MB per expert blob, 896
experts across 93 MoE layers). Dedicating ~200 GB to expert slots caches
about 16% of the pool. At 16 active of 896 per layer, miss rates stay high;
expect roughly 10-15 GB/token of SSD traffic, or 2-3 seconds per token at
NVMe speed, before compute. The bandwidth ceiling if it *were* resident
(800 GB/s / ~52 GB active bytes) is about 15 tok/s, but residency is the
thing 256 GB cannot provide. Recorded as rejected on every host in the
fleet. Re-screen only if a materially smaller K3 variant ships.

### What the fleet changes strategically

1. **The 128 GB Mac is the streaming sweet spot.** V4-Flash at ~140 GB is
   the ideal shape for it: the resident budget covers common weights plus
   most of the expert pool, and streaming covers the tail. Expected result
   is far better than the 8 GB Gemma case because the miss rate collapses
   when most experts fit. PORT-V4F gates should target this host.
2. **The M3 Ultra makes V4-Flash boring, in a good way.** Fully resident at
   ~800 GB/s gives a theoretical ceiling around 120 tok/s for 13B active;
   stock tooling (MLX, vLLM-class) is the right answer there. Its role in
   the experiment series is the high-end control group.
3. **The M3 Pros are validation workhorses.** Two identical 48 GB hosts are
   ideal for interleaved A/B runs and holdout gates, per the measurement
   lessons in summary 09. Gemma-4-class and Qwen3-30B-class models are
   fully resident on them.
4. **DGX Spark remains the odd one out.** Same 128 GB class as this Mac,
   but bandwidth-bound rather than capacity-bound, per PORT-SPARK.

## FLEET-SWARM: local models as default agentic developers

Vision (recorded 2026-08-01): run V4-Flash on multiple owned systems as the
default model for Jcode agentic workers, coordinate Jcode sessions across
machines, and distribute large development projects to swarms of up to ~100
cooperating agents.

### Capacity math first

Swarm size is bounded by aggregate token throughput, not by agent count.
Agents also spend wall-clock time on tool calls and waiting, so effective
concurrency exceeds decode concurrency, but coding agents are
generation-heavy. Honest planning numbers:

| Host | Model residency | Single-stream decode | Batched aggregate (est.) |
| --- | --- | --- | --- |
| M3 Ultra Studio | V4-Flash resident | 30-60 tok/s (120 ceiling) | ~150-400 tok/s |
| 128 GB Mac | V4-Flash streamed tail | 10-30 tok/s | ~50-100 tok/s |
| 48 GB M3 Pro x2 | Gemma 4 / Qwen3-30B resident | 20-40 tok/s | ~40-80 tok/s each |

Fleet total: roughly 300-600 tok/s of local generation. At 20-30 tok/s per
active agent stream, that is **10-20 concurrent local agents, not 100**.
A 100-agent swarm is a scheduling abstraction: agents queue, and per-agent
throughput degrades with contention. Recorded as a constraint, not a
blocker: the fix is hierarchical routing, below.

### The architecture that makes the vision work

Four tiers, each independently testable:

1. **Inference tier.** Each host runs an OpenAI-compatible loopback server
   (TurboFieldfareServer on this repo's hosts; MLX/vLLM-class servers on
   the others). Loopback only, per the existing security posture; remote
   access goes through the coordination tier, never by exposing the port.
2. **Routing tier.** Jcode spawns already accept per-agent model routes.
   Add each local server as a provider route, then assign by role, mirroring
   the existing swarm routing policy: bulk reading/summarization and
   bounded implementation tasks go to local models; design, review,
   verification, and anything quality-critical go to frontier API models.
   A swarm of weak agents is not a strong engineer; a strong coordinator
   with cheap workers is.
3. **Coordination tier.** Git plus typed task-graph artifacts are the
   shared medium, which is already Jcode's preferred dataflow. The unbuilt
   piece is cross-machine: today a swarm lives in one Jcode instance on one
   host. Options in increasing effort: ssh-driven remote Jcode sessions
   with per-host working copies and branch-based integration; multiple
   coordinators sharing one repo through pull-based work queues; or a
   proper job broker. Start with ssh and git.
4. **Scheduling tier.** The task graph (deep mode) already handles gating,
   retries, and confidence-routed follow-up work on one host. Cross-machine
   scheduling means assigning graph nodes to hosts by model route and
   current load. This is policy, not new machinery, once tier 3 exists.

### Experiment series

- **FS-00: Serving baseline.** Hypothesis: V4-Flash serves interactively on
  the M3 Ultra via stock tooling with tool calling intact. Method: stand up
  the server, run the OpenAI tool-loop checks from this repo's server
  guide. Gate: correct tool calls over a 20-prompt suite, single-stream
  decode >= 25 tok/s.
- **FS-01: Agent quality gate.** Hypothesis: V4-Flash completes bounded
  coding tasks (single-file fix with tests) at acceptable quality. Method:
  fixed task suite, scored against a frontier API baseline, blinded rubric.
  Gate: defines which swarm roles local models may hold. **No swarm work
  before this gate.** A routing table without measured per-role quality is
  the CACHE-04 mistake at the agent level.
- **FS-02: Single-host swarm.** Hypothesis: a deep-mode task graph with
  local workers and API reviewers completes a multi-file feature faster
  than one API agent alone, measured wall-clock and dollar cost. Gate:
  faster or cheaper with equal review pass rate.
- **FS-03: Cross-machine coordination.** Hypothesis: ssh + git + task-graph
  artifacts coordinate two hosts without a broker. Method: coordinator on
  the Studio, workers on the 128 GB Mac and one M3 Pro, one shared repo
  with branch-per-node integration. Gate: a 10-node graph completes with no
  merge casualties and full artifact provenance.
- **FS-04: Fleet economics.** Method: run FS-03 at increasing agent counts,
  record throughput, queue depth, quality, and API spend per landed change.
  Disposition rule: report the agent count where contention degrades
  per-agent speed past usefulness. That number, not 100, is the fleet's
  true swarm size, and it grows every time a host or a better quant ships.

### Recorded risks

- Local-model tool-call reliability is the most likely FS-01 failure;
  Gemma-4-class models already fail closed on malformed tool output in this
  repo's server, and V4-Flash needs the same defensive parsing.
- Cross-machine Jcode coordination is the genuinely novel engineering;
  everything else is composition.
- Thermal and SSD endurance under sustained multi-agent load are
  unmeasured on every host. Add both to FS-04 measurements.

## What is deliberately out of scope

- **Kimi K3 local port.** Rejected at screen above. Aggressive requantization
  below ~1.5-2 bits average (~500-700 GB artifact) is a server-class GGUF
  problem, not a streaming-runtime problem, and it does not change the
  104B-active I/O floor.
- **V4-Pro.** Same rejection class as K3.
- **Training, fine-tuning, vision towers.** Same scope as the current repo.
