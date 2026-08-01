# V4F-02 work order: FP4/FP8 quant kernels for DeepSeek V4-Flash

Status: recon complete, ready to schedule. Read-only survey of the existing
quantization stack; no source was modified. Geometry source of truth:
[PORTING_PLAYBOOK.md](../PORTING_PLAYBOOK.md) V4F-00 result table (hidden
4096, 256 routed experts + 1 shared, 6/token, expert intermediate 2048,
SwiGLU with `swiglu_limit` 10.0, FP4 experts, FP8-e4m3 everything else).

## 1. Inventory of existing dequant kernels and their assumptions

All runtime quantization today is MLX `affine`: unsigned integer codes, one
BF16 scale **and** one BF16 bias per group of 64 elements,
`w = q * scale + bias`. `Quantization.groupSize = 64`
(`Sources/TurboFieldfare/Infrastructure/ModelIO/Quantization.swift:5`) is the
single shared constant. The affine factoring `s·Σqx + b·Σx` is the core
identity every GEMV exploits: per group, one FMA for the scale against the
code·activation dot and one FMA for the bias against the activation sum.

### 1.1 `Metal/Quant/dequant_int4.metal`

| Kernel | Wrapper | Assumptions |
| --- | --- | --- |
| `embed_lookup_int4` | `EmbedLookupInt4` | Table `[V, D/2]` packed nibbles; low nibble of byte k = element 2k, high = 2k+1. Scales/biases `[V, D/64]` BF16. `D % 64 == 0` (wrapper precondition). Fused `out_scale` (sqrt(hidden) embedding scale). One thread per output element. |
| `dequant_int4_gemv_simd` | `DequantInt4GEMV` | Row-major `[M, N/2]` nibble weights, same packing order. `N % 64 == 0` enforced at `DequantInt4GEMV.encode` (`Kernels/Quant/DequantInt4GEMV.swift:57`). Vectorized main loop consumes 4 groups (128 weight bytes) per iteration: each lane loads **two `ushort`s** and combines them, never a `uint`, because the repacker guarantees only 2-byte sub-tensor alignment in the resident common set (BF16 scale/bias regions leave 2-aligned `weightsOffset`); wrapper precondition `weightsOffset % 2 == 0` (line 61). `x` is read as `half4` pairs, so activation offsets must be 8-byte aligned. Scalar remainder path (byte per lane) covers group counts not divisible by 4. Eight rows per threadgroup, one SIMD group per row, `simd_sum`, lane 0 writes half. Function-constant specialization on (M, N) at indices 20/21/22; `realDecodeShapes` is a hardcoded Gemma list. |
| `dequant_int4_qkv_gemv_simd` | (attention path) | Three concatenated GEMVs over one shared `x`; rows partitioned q/k/v by row index. Same packing, group, and alignment assumptions. FC indices 23-26. |

### 1.2 `Metal/Quant/dequant_int8.metal`

| Kernel | Wrapper | Assumptions |
| --- | --- | --- |
| `dequant_int8_gemv_simd` | `DequantInt8GEMV` | One byte per element (no unpack), group 64, BF16 scale + bias. `N % 64 == 0`. 8 rows/TG, 2 elements per lane per group. Used for router (M=128), shared-expert projections, and lm_head (M=262144). FC indices 70-72. `max_total_threads_per_threadgroup(256)`. |
| `shared_int8_gate_up_act_simd` | `SharedExpertInt8` | Two weight tensors (gate, up) walked together per row so `x` is loaded once; fused tanh-GELU(gate)·up written to `act`. Same group/scale/bias assumptions. Rows-per-TG overridable via FC index 73. |

### 1.3 `Metal/MoE/moe.metal` — fused routed MoE (decode)

| Kernel | Assumptions |
| --- | --- |
| `router_gemv_gemma4_r4` | INT8-affine router GEMV with an extra per-element `effective_scale` multiply on the activations (Gemma-specific). 4 experts per TG. FC indices 40-43. |
| `router_topk_select_k8` | **Hardcoded top-8.** Single thread scans 256 logits, softmax over the top 8, multiplies by `per_expert_scale`. Not reusable for top-6 without a new kernel or a k-parameter. |
| `moe_phase1_gate_up_act_u16load` (+ `_subset_` variant) | The workhorse: fused INT4-affine dequant + GeGLU over up to `kMaxStreamedExperts = 8` expert blobs bound through a `RoutedBlobs` argument buffer (`device const uint8_t* blob[8]`). Per-blob sub-tensor offsets come from one `ExpertOffsets` struct (9 × uint32: gate/up/down × W/s/b). Gate and up rows walked together via `moe_int4_gate_up_rows_simd_dev_vec_u16load`, u16-pair loads (2-byte alignment contract again). Fused activation is **tanh-GELU, not silu**. Grid = `top_k * F` rows; `MoE.validate()` in `Kernels/MoE/MoE.swift:297` preconditions `topK == 8`. |
| `moe_phase2_down_reduce_k8` | One threadgroup per output element `d`; 8 SIMD groups, one per expert slot, each running `moe_int4_gemv_row_simd_dev_vec` on the down projection. Note this helper uses aligned **`uint*` loads** (line 208): the packed expert-blob layout guarantees 4-byte sub-tensor alignment, unlike the resident common set. Weighted by routing weight, summed with residual, written half. Hardcoded 8 partials. |

### 1.4 `Metal/TensorCore/tensorops.metal` — MPP prefill QMM

`mpp_prefill_affine_threadgroup_f16` (wrapper
`Kernels/TensorCore/MPPPrefillInt4QMM.swift`): tiles M=64, N=32, K=64. Per
K-group, all threads cooperatively dequantize one 32×64 int4-affine tile to
`threadgroup half`, then `matmul2d` (4 SIMD groups, fp32 accumulate) runs per
group and accumulates. Assumptions: `K % 64 == 0`, scale/bias offsets 2-byte
aligned, x/y 2-byte aligned (`MPPPrefillInt4QMM.encode` guards, lines 43-52).
Gated on `__HAVE_TENSOR__`; falls back to `.unavailable`. The dequant step is
per-element affine inside the tile fill loop (line 75), so swapping the
dequant formula is a localized change.

### 1.5 Supporting Swift infrastructure

- `Quantization.swift`: BF16 bit helpers (`bf16Bits`, `bf16ToFloat`),
  test-only quantize/dequantize reference implementations for both affine
  formats. These are the pattern for the fixture-side FP4/FP8 reference code.
- `MoE.swift`: `MoEExpertOffsets` (9 × UInt32), `maxStreamedExperts = 8`,
  argument-buffer machinery (`makeReusedRoutedArgumentBuffer`), real-decode
  shape constants hardcoded to Gemma (D=2816, F=704, topK=8, 128 experts).
- `ModelExpertIO.routedExpertOffsets` (`Runtime/Inference/ModelExpertIO.swift:20`)
  derives `MoEExpertOffsets` from `packedExpertsLayout` sub-tensor records by
  role name ("gate", "gate_scales", ...). V4 blobs will carry different
  roles (no biases; ue8m0 scales), so the layout descriptor and this mapping
  grow in parallel (V4F-01 boundary).
- `PrefillGroupedRoutedMoE.swift`: prefill-side routed MoE over 16-expert
  tiles; carries its own copy of the 9 offsets in
  `PrefillGroupedRoutedMoEStreamedParams`. Its kernels
  (`prefill_grouped_routed_moe_batched_phase1` / `_batched_down`) share the
  same affine assumptions and GeGLU activation.
- `SharedExpertInt4` / `SharedExpertInt8`: unfused three-GEMV +
  `gelu_mul_fp16` composition for the shared expert. The V4 shared expert is
  SwiGLU/FP4 and can follow the same composition pattern with a new
  `silu_mul` elementwise kernel, or reuse the fused phase1 path.

## 2. Target formats and required new kernels

### 2.1 Format deltas vs. the affine stack

| Property | Existing (Gemma) | V4-Flash routed experts | V4-Flash non-expert |
| --- | --- | --- | --- |
| Code format | uint4 / uint8 | FP4 e2m1 (sign + 2 exp + 1 mantissa) | FP8 e4m3 |
| Scale | BF16 per 64 elems | ue8m0 (power of two) per microscaling group (confirm group size, expected 32) | ue8m0 per **128×128 2-D block** |
| Bias | BF16 per group | **none** | **none** |
| Dequant | `q*s + b` | `lut[q] * 2^k` | `cvt(q) * 2^k` |
| Activation | tanh-GeGLU | **SwiGLU: silu(clamp(gate)) · clamp(up), limit 10.0** | n/a |

Two consequences simplify the kernels: no bias term means the `b·Σx` FMA and
the Σx bookkeeping disappear, and power-of-two scales mean the scale multiply
is exact in float (an exponent add), so per-group accumulate-then-scale has
no rounding hazard.

### 2.2 New kernels required

Decode (GEMV-class), one SIMD group per row, following the
`dequant_int4_gemv_simd` skeleton:

1. **`dequant_fp4_gemv_simd`** (`Metal/Quant/dequant_fp4.metal` +
   `DequantFP4GEMV.swift`). 16-entry e2m1 lookup table (exact in half/float),
   one ue8m0 scale byte per microscaling group, no bias. Same u16-pair load
   discipline as the int4 kernel unless V4F-01 guarantees 4-byte alignment for
   these tensors (recommended: pad to 4 in the repack and use uint loads).
   Rows: expert gate/up `[2048, 4096]`, expert down `[4096, 2048]`, plus any
   FP4 shared-expert projections.
2. **`dequant_fp8_e4m3_gemv_simd`** (`Metal/Quant/dequant_fp8.metal` +
   `DequantFP8GEMV.swift`). One byte per element, e4m3→float conversion
   (bit-manipulation; Metal has no native e4m3). Scale fetch is 2-D:
   `scale_grid[(row/128) * (N/128) + (col/128)]`, one byte. Per 128-column
   block: accumulate `Σ cvt(q)·x` in float, one ldexp-style scale multiply per
   block. N % 128 == 0 and M % 128 == 0 hold for all published V4-Flash dense
   shapes; enforce as preconditions and keep a scalar tail only if a future
   shape breaks it. Covers attention projections, the router, lm_head
   (M=129280, N=4096, 32× scale grid), and the grouped output projection.
3. **`embed_lookup_fp8`** (adapt `embed_lookup_int4`). Untied FP8 embedding
   table, row lookup + e4m3 convert + block scale + optional `out_scale`.
   Trivial; can also be served by a specialized `dequant_fp8_e4m3_gemv` on a
   one-hot, but the direct lookup is one small kernel.
4. **`moe_fp4_phase1_gate_up_act_swiglu`** and the subset variant (adapt
   `moe_phase1_gate_up_act_u16load`). Changes: e2m1 LUT + ue8m0 scales, drop
   biases, replace `gelu_pytorch_tanh(gu.x) * gu.y` with
   `silu(clamp(g,-10,10)) * clamp(u,-10,10)`, top_k 6 (six blobs instead of
   eight; `RoutedBlobs` shrinks or becomes a template/FC parameter).
   The exact clamp semantics (which operands are clamped, where the limit
   applies) must be transcribed from the official PyTorch reference before
   coding; treat as a correctness-critical one-liner.
5. **`moe_fp4_phase2_down_reduce_k6`** (adapt `moe_phase2_down_reduce_k8`):
   six SIMD groups/partials, FP4 down projection, weighted reduce + residual.
6. **`router_fp8_gemv` + `router_topk_select_k6`**: FP8 block-scaled router
   GEMV (item 2 covers the math; a dedicated entry point avoids the
   `effective_scale` Gemma-ism) and a k-parameterized top-k select. V3/V4
   routing uses sigmoid scoring with a correction bias — confirm against the
   reference config; the softmax-over-topk in `router_topk_select_k8` is not
   assumed reusable.

Prefill (MPP staged), following `mpp_prefill_affine_threadgroup_f16`:

7. **`mpp_prefill_fp4_threadgroup_f16`**: same tiling; the tile-fill loop
   swaps the affine formula for LUT + ue8m0. Because e2m1 values and powers
   of two are exact in half, the dequantized tile stays exact unless scale
   magnitudes push outside half range; fold the scale in float during tile
   fill (cast to half after scaling, matching the current kernel's
   `half(fma(...))` pattern) and validate range against real checkpoint
   scales. Wrapper mirrors `MPPPrefillInt4QMM` with a new
   `MPPPrefillFP4QMM`/`Path` case.
8. **`mpp_prefill_fp8_e4m3_threadgroup_f16`**: same, for the dense FP8
   weights. Scale fetch is per (tile-row-block, tile-col-block); K-tiles of
   64 split a 128-block, so the block scale is fetched once per K-pair-of-
   tiles — simplest correct form fetches per element `scale[(gN/128)][(gK/128)]`
   and relies on the cache; optimize only if the profile says so.
9. **Prefill grouped routed MoE for FP4/SwiGLU**: the
   `prefill_grouped_routed_moe_batched_phase1` / `_batched_down` pair needs
   FP4 + SwiGLU variants, or the prefill path routes expert GEMMs through
   items 7 and a grouped reduction. This is the least-scoped piece; size it
   during implementation against the actual prefill MoE call graph.

## 3. Adapt vs. write fresh (with citations)

**Adapt (skeleton, dispatch geometry, alignment discipline all carry over):**

- `dequant_fp4_gemv_simd` from `dequant_int4_gemv_simd_body`
  (`dequant_int4.metal:93-172`): keep the 8-rows/TG structure, 4-group main
  loop with remainder, `simd_sum` finish. Replace nibble decode + affine FMA
  with LUT + scale; delete the bias accumulator. Wrapper copies
  `DequantInt4GEMV` including the function-constant specialization table —
  new `realDecodeShapes`: (2048, 4096), (4096, 2048), router/head shapes as
  needed.
- `moe_fp4_phase1/phase2` from `moe.metal:248-482`: the gate/up shared-x
  walk, `RoutedBlobs` argument buffer, and slot-subset variant are all
  directly reusable in structure. `MoE.swift` needs `maxStreamedExperts` /
  `validate()` generalized off the hardcoded 8 (line 32, 297) and a V4
  offsets struct with 6 fields (no biases) next to `MoEExpertOffsets`;
  `ModelExpertIO.routedExpertOffsets` gains a parallel role mapping.
- MPP prefill kernels from `tensorops.metal:13-105`: tiling, tensor slicing,
  and accumulation are untouched; only the tile-fill dequant line changes.
- `SharedExpertInt4`'s composition (`SharedExpertInt4.swift`) with a new
  `silu_mul_fp16` elementwise kernel replacing `gelu_mul_fp16`.
- `Quantization.swift`: add e2m1 LUT, e4m3 decode, and ue8m0 helpers
  alongside `bf16Bits`/`bf16ToFloat` for fixture generation and CPU reference.

**Write fresh:**

- `dequant_fp8_e4m3_gemv_simd` + wrapper: the 2-D 128×128 block-scale
  indexing has no precedent in the stack; the int8 kernel is only a loose
  template.
- `embed_lookup_fp8`, `router_topk_select_k6` (parameterize k and scoring),
  `silu_mul_fp16`.
- The manifest/layout descriptor for the new sub-tensor roles
  (`gate_scales_ue8m0`, FP8 `weight_scale_inv`-style grids) — V4F-01 owns the
  repack side, V4F-02 owns the reader contract.

**Open questions to resolve against the checkpoint/reference before coding
(verify in V4F-01 fixture extraction):**

1. FP4 microscaling group size (expected 32) and whether scales are stored
   per row-of-groups or interleaved.
2. Nibble packing order of e2m1 pairs on disk (lo-first vs hi-first) and
   tensor endianness of the ue8m0 scale arrays.
3. FP8 block-scale grid layout (row-major over `(M/128, N/128)` is expected;
   V3 used a transposed scale tensor in some checkpoints).
4. Exact `swiglu_limit` application point and router scoring (sigmoid +
   correction bias vs softmax).
5. Whether the shared expert and the first-3 hash-routed layers share the
   routed FP4 format exactly (affects only descriptor plumbing).

## 4. Validation strategy per kernel

Oracle: the official PyTorch inference code on `deepseek-ai/DeepSeek-V4-Flash`
(V4F-00 recorded it as the reference for V4F-02/03). Gate per the playbook:
per-tensor tolerance on a fixed test vector set; bit identity not required.

1. **Fixture extraction (shared).** Script pulls a fixed sample of real
   checkpoint tensors via safetensors: 2-3 expert gate/up/down row-blocks, one
   dense attention projection tile, router rows, embedding rows, and their
   scale tensors. Emits raw payload bytes (as the repacker will deliver them)
   plus golden float32 dequantized weights and golden GEMV/GEMM outputs for
   fixed-seed input vectors (deterministic RNG, saved seeds). Fixture bytes
   must match what V4F-01's layout emits, so the harness consumes the same
   sub-tensor offsets the runtime will.
2. **CPU reference in Swift first.** Extend `Quantization.swift` with
   e2m1/e4m3/ue8m0 decode. Dequantized-weight comparison against golden must
   be **exact** (LUT values and powers of two are exactly representable);
   any mismatch here is a packing/layout bug, not tolerance. This is the
   cheapest place to resolve open questions 1-3.
3. **GPU GEMV kernels** (`dequant_fp4_gemv_simd`,
   `dequant_fp8_e4m3_gemv_simd`, `embed_lookup_fp8`): compare against golden
   GEMV outputs. Tolerance: FP4 path ~1e-2 relative on the output (half
   output, fp32 accumulate); FP8 path ~5e-3 (larger dynamic range, exact
   scales). Include shapes that hit the remainder path (group count not
   divisible by 4) and misalignment-adjacent offsets (2-aligned but not
   4-aligned) as negative-path tests mirroring the existing preconditions.
4. **Fused MoE** (`moe_fp4_phase1/phase2`): golden phase-1 activations and
   phase-2 outputs from the reference SwiGLU expert forward on 6 fixed expert
   blobs with fixed routing weights. This is where the `swiglu_limit`
   transcription is proven; include activation magnitudes that exercise the
   ±10 clamp. Tolerance 1e-2 relative; also test the subset variant with 1,
   3, and 6 active slots.
5. **MPP prefill variants**: golden GEMM outputs at M = 64 and 128 tokens,
   same fixtures. Tolerance 5e-3 relative; gate on `isAvailable` skip
   behavior unchanged. Verify the half-tile range question (2.2 item 7) by
   asserting max |scaled tile value| from real checkpoint scales before
   trusting tolerance results.
6. **Router**: golden logits + top-6 indices/weights from the reference
   routing function, including tie cases and the correction-bias path.
7. **Harness placement**: follow the existing `Tests/` pattern (focused
   public tests, one model-using test at a time); quant-kernel tests are
   GPU-but-not-model tests and can share one fixture bundle on disk.

## 5. Rough effort estimates (engineer-days)

| Item | Days | Notes |
| --- | ---: | --- |
| Fixture extraction + golden generation from PyTorch reference | 3 | Includes resolving the five open questions; gates everything below |
| `Quantization.swift` e2m1/e4m3/ue8m0 helpers + CPU reference | 1 | Exact-match validation falls out of this |
| `dequant_fp4_gemv_simd` + wrapper + tests | 2 | Adapt-heavy |
| `dequant_fp8_e4m3_gemv_simd` + `embed_lookup_fp8` + wrapper + tests | 2 | Fresh 2-D block-scale indexing |
| Fused MoE phase1/phase2 FP4-SwiGLU (k6, subset variant, offsets plumbing) | 3 | Largest decode item; touches `MoE.swift`, `MoEExpertOffsets`, `ModelExpertIO` |
| Router FP8 GEMV + top-6 select | 1 | Small once scoring rule is confirmed |
| MPP prefill FP4 variant | 2 | Tile-fill swap + range validation |
| MPP prefill FP8 variant | 2 | Same, plus block-scale fetch |
| Prefill grouped routed MoE FP4/SwiGLU | 2-3 | Least scoped; may share kernels with decode |
| **Total** | **18-19** | Serial; some parallelism possible across Metal vs. harness work |

## 10-line summary

1. The entire existing stack is MLX-affine (group-64 codes, BF16 scale+bias); nothing in the tree speaks FP4, FP8, or power-of-two scales today.
2. Three decode GEMV families exist (int4, int8, embed lookup) plus a fused int4+GeGLU+weighted-reduce routed MoE and an MPP tiled prefill QMM, all sharing the same one-SIMD-per-row structure.
3. Alignment contract is explicit: resident tensors guarantee only 2-byte sub-tensor alignment (u16-pair loads), packed expert blobs guarantee 4 (uint loads); V4F-01 should pad FP4 sub-tensors to 4 bytes to unlock the faster path.
4. FP4 experts need a new e2m1-LUT + ue8m0 dequant GEMV, a fused SwiGLU (`silu(clamp(g,±10))·clamp(u,±10)`) phase-1, and a k=6 phase-2 reduce; all are structural adaptations of `moe.metal` with biases deleted.
5. FP8-e4m3 dense weights need a fresh GEMV with 2-D 128×128 block-scale indexing; it also covers router, lm_head (129280×4096), and an FP8 embed lookup.
6. Prefill needs two MPP variants (FP4, FP8); the change is localized to the tile-fill dequant line, with a half-range check on real checkpoint scales.
7. `MoE.swift`'s hardcoded topK=8/`maxStreamedExperts=8` and the 9-field `MoEExpertOffsets` (with biases) must be generalized to 6 slots and 6 fields.
8. Validation runs per kernel against golden tensors extracted from the official PyTorch reference: exact match on CPU dequant, ~1e-2/5e-3 relative on GPU outputs, clamp-exercising SwiGLU cases.
9. Five open questions (FP4 group size, nibble order, scale grid layout, swiglu clamp point, router scoring) must be resolved from the checkpoint before coding; they are cheap to settle in the fixture step.
10. Total estimate ~18-19 engineer-days, roughly one-third of it the fixture/validation harness and the fused MoE rework.
