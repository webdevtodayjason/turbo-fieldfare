# V4F-01 work order: repacker and layout for DeepSeek V4-Flash

Status: work order (recon complete). No code written. Prepared by recon worker,
2026-08-01, against commit on `main`.

Playbook reference: `docs/experiments/PORTING_PLAYBOOK.md`, V4F-00 result table
(gate passed) and V4F-01 stage definition. Gate for this stage: byte-identical
payload round-trip on a sample of ranges, verified by digest, with the same
512 KB heap bound as IO-10.

Target geometry (from V4F-00): 43 layers, hidden 4096, 256 uniform routed
experts/layer, SwiGLU intermediate 2048, FP4 expert weights (~12.6 MB blob),
6+1 experts per token, FP8 e4m3 non-expert with 128x128 blocks and ue8m0
scales, vocab 129280 untied, source `deepseek-ai/DeepSeek-V4-Flash`.

## 1. What transfers unchanged

These components are model-agnostic. They see only planned byte ranges and
planned output files; none of them parse tensor semantics.

- `Core/Remote/HuggingFaceRemote.swift` (`HuggingFaceRemoteSource`,
  `resolveFileInfo`, `pinned`, `downloadRangeToTempFile`, `fetchSmallFile`):
  pure HF resolve/Range plumbing, keyed on repoID + revision strings only.
- `Core/Remote/RemoteDownloadSession.swift`, `RemoteRetry.swift`
  (`RemoteRetryPolicy`), `RemoteChunkPolicy.swift`, `RemoteRangeTransfer.swift`,
  `SourceByteProvider.swift` (`HTTPRangeSourceByteProvider.copyBatch`,
  `destinationDigest`): bounded-range download, retry, digest verification.
- `Core/Planning/RangeCopyPlanner.swift`: coalescing, splitting, destination
  interval validation, canonical fingerprint. It consumes a `RepackPlan`; it
  never inspects dtype or names. Transfer requires only that the planner
  above it produces valid plans.
- `Core/Writing/ResidentWriter.swift`, `WriterCore.swift`,
  `BoundedScratch.swift`: tile-bounded pwrite machinery, index encoder
  (`GTurboBinary.writeIndexHeader` / `writeIndexEntry` in
  `Core/Format/GTurboEncoders.swift`), 512 KB scratch bound (the IO-10 bound
  the V4F-01 gate references).
- `Core/Remote/RemoteInstallCheckpoint.swift`, `Core/System/InstallLock.swift`,
  `Core/System/Posix.swift`, `Core/System/DiskSpaceChecker.swift`:
  resume/checkpoint/locking. Checkpoint validation is fingerprint-based, so
  a V4 plan change correctly invalidates old partials.
- `Core/Verification/Sha256Stream.swift`, `RepackAudit.swift`,
  `VerifiedInstallReceiptWriter.swift`, `VerifiedInstallTool.swift`:
  hashing, audit, receipt. The audit records `tensorsDroppedMultimodal` and
  `packedExpertLayoutMode` as strings, so new values need no schema change.
- `Core/Remote/RemoteStreamingRepacker.swift` orchestration skeleton
  (`run`/`runPrepared`): lock, snapshot, plan, range plan, checkpoint,
  copy, layout.json write, `GTurboLayoutValidator.validate`, sidecar copy,
  manifest write, atomic rename. The control flow is correct for V4; the
  Gemma assumptions it embeds are itemized in section 2.
- `Core/Format/Safetensors.swift` header-parsing structure (8-byte length,
  JSON header, `data_offsets` -> absolute offsets). Only the dtype switch
  is Gemma-shaped.
- `Core/Format/GTurboJSON.swift` encoder skeleton and
  `Core/Format/GTurboLayoutValidator.swift`: layout.json is already written
  per-expert from the plan; the validator checks page alignment,
  `offset == expert * stride`, full expert coverage. All still the right
  invariants for 256-expert fixed-stride V4 layers.
- Runtime `ResidentIndexReader` (binary index decode) and
  `PackedExpertsLayoutReader` (layout.json decode): both are schema-driven
  and name-agnostic. `Quantization.bf16Bits`/`bf16ToFloat` helpers remain
  useful regardless of model.

Net: the entire streaming, resume, verification, and I/O-boundedness
machinery transfers. The work is concentrated in four Gemma-shaped seams:
source gating, metadata parsing, tensor classification/layout planning, and
the manifest/quant schema.

## 2. Where Gemma-specific assumptions live

### 2.1 Source gating (repo identity)

- `Core/Remote/SupportedModelSource.swift`: hardcoded
  `repoID = "mlx-community/gemma-4-26b-a4b-it-4bit"`, pinned `revision`,
  `sourceIndexSHA256`, `approximateDownloadBytes` (~14.6 GB),
  `installedBytes`, `reserveBytes`. The CLI (`Command/main.swift`, `run`)
  calls `SupportedModelSource.installOptions(...)` unconditionally and prints
  `displayName` on success; usage text says "the supported Gemma 4
  checkpoint".
- `Core/Verification/SourceFingerprint.swift`: `knownFingerprints` has one
  entry; `requireKnownSource: true` (set by `SupportedModelSource
  .installOptions`) makes `RemoteSnapshotLoader.load` throw
  `sourceFingerprintRejected` for any other weight index.
- `RemoteStreamingRepacker.writeManifest`: `modelID: plan.matchedModelID ??
  "unknown/snapshot"` degrades gracefully, but the install path still hard
  gates on the fingerprint.

### 2.2 MLX affine quant metadata

- `Core/Format/IndexLoader.swift.load`: requires `config.json ->
  quantization` with MLX keys `bits` / `group_size` / `mode` plus per-tensor
  overrides keyed by name-without-`.weight`; throws
  `configJsonInvalid("no quantization slot")` otherwise. DeepSeek V4-Flash
  has no such block; its FP8/FP4 scheme is native (e4m3 with 128x128 blocks
  and ue8m0 scales for non-experts, FP4 expert weights), described by
  DeepSeek-style `quantization_config` fields (`quant_method`, `fmt`,
  `weight_block_size`), not MLX affine.
- `IndexLoader.quantSpec(forTensor:)`: returns bits only; group size is
  assumed uniform and validated against `baseGroupSize`. V4 has two distinct
  quant families with different blocking, so `QuantSpec` (`bits: Int` in
  `Core/Format/TensorMetadata.swift`) cannot express either.
- `RepackPlanner.planResidentFile`: quantized detection is
  `weight.dtype == .u32 && name.hasSuffix(".weight")` — the MLX convention of
  packing int4 into U32 with sibling `base + ".scales"` / `base + ".biases"`
  BF16 tensors. It then hard-requires those companions
  (`missingScalesCompanion` / `missingBiasesCompanion`) and requires
  `scales.dtype == .bf16`. V4 non-expert weights are `F8_E4M3` with F32
  `weight_scale_inv` block scales and no biases; expert weights are FP4 with
  their own scale sidecar naming. None of this matches.
- `RepackPlanner.logicalShape(forPackedSource:bits:)`: assumes
  `last_dim * 32/bits` unpacking from a U32 container. FP8 is one byte per
  element (no unpacking); FP4 packing is 2 elements/byte but the container
  dtype and order are source-defined (see risks).
- `RemoteStreamingRepacker.writeManifest`: default
  `GTurboJSON.QuantBitWidths(embedding: 4, attention: 4, router: 8,
  sharedExpert: 8, routedExpert: 4)` with Gemma-name probes
  (`".self_attn.q_proj.weight"`, `".router.proj.weight"`,
  `".mlp.gate_proj.weight"`,
  `"language_model.model.embed_tokens.weight"`) to refine them.

### 2.3 Manifest / arch gate

- `Core/Format/ArchInfo.swift.load`: requires `config.json -> text_config`
  and Gemma keys (`moe_intermediate_size`, `num_experts`, `top_k_experts`,
  `num_global_key_value_heads`, `global_head_dim`, `layer_types`,
  `rope_parameters.full_attention.partial_rotary_factor`,
  `attention_k_eq_v`, `final_logit_softcapping`, ...). DeepSeek config.json
  is flat (no `text_config`) and uses `n_routed_experts`,
  `num_experts_per_tok`, `q_lora_rank`, etc. This loader throws before any
  planning.
- `ArchInfo` struct itself: the field set is the Gemma schema (dual
  RoPE thetas, `fullAttentionLayerMask`, `attentionKEqV`,
  `finalLogitSoftcap`). CSA/HCA, mHC, partial RoPE with output de-rotation,
  and grouped output projection have no fields.
- `Core/Format/GTurboJSON.encodeManifest`: emits that same arch dict plus a
  fixed five-slot quant table with hardcoded `"scheme": plan.baseMode`
  (`"affine"`), `"scaleType": "BF16"`, `"biasType": "BF16"`, `"groupSize":
  plan.baseGroupSize`.
- Runtime `ManifestReader.validateQuant`: allows only `embedding [4]`,
  `attention [4]`, `router [8]`, `sharedExpert [4,8]`, `routedExpert [4]`,
  all with `scheme == "affine"`, BF16 scale/bias, and `groupSize ==
  Quantization.groupSize` (64, hardcoded in
  `Infrastructure/ModelIO/Quantization.swift`). Every V4 slot fails this.
- Runtime `ManifestReader.validateArch`: field-by-field equality against
  `ArchConfig.gemma4_26B_A4B` (hardcoded in
  `Infrastructure/ModelIO/ModelTypes.swift`, used by `Model.load` and the
  app probe). Also `ManifestArch` has no optional fields, so the manifest
  JSON schema itself is Gemma-shaped.
- `ManifestReader.validate`: requires `packed_experts/layer_%02d.bin` for
  every L in `0..<numLayers`. Fine for V4 (all 43 layers are MoE), but note
  layers 0-2 are hash-routed MoE — same file treatment, routing policy is a
  runtime matter, not a repack matter.

### 2.4 Tensor naming / classification and layout.json schema

- `RepackPlanner.classify`: requires the `language_model.` prefix; V4 names
  are `model.layers.<N>...`. Every V4 tensor lands in `.unknown` and throws
  `unknownTensorPrefix`.
- `RepackPlanner.routedExpertRole`: expects
  `.experts.switch_glu.{gate,up,down}_proj.` (one stacked rank-3 tensor per
  role per layer, `[experts, out, in]`). V4 almost certainly stores experts
  either as per-expert 2D tensors (`model.layers.N.mlp.experts.E.gate_proj
  .weight`, the DeepSeek convention) or as a stacked tensor under a
  different name; and roles are `gate_proj`/`up_proj`/`down_proj` with SwiGLU
  but possibly with FP4 scale sidecars (`weight_scale_inv` or similar).
- `RepackPlanner.planLayerFile`: assumes exactly 3 rank-3 U32 weight
  tensors with BF16 `.scales`/`.biases` siblings, 9 slices per blob ordered
  weights->scales->biases per role. FP4 experts with per-block scales need a
  different slice set (weights + scales only, scale dtype F32/FP8, count 6,
  not 9).
- `RepackPlanner.lmResidentOrdering` / `slotRank`: Gemma names
  (`self_attn.q_proj`, `router.proj`, `router.scale`,
  `router.per_expert_scale`, `mlp.{gate,up,down}_proj` for the shared
  expert, the six layernorm spellings, `.layer_scalar`). Unknown names sort
  at rank 100 — deterministic, so ordering still works, but the V4 layer
  (q_a/q_b LoRA projections, `kv_a_proj_with_mqa`, `kv_b_proj`, `mlp.gate`
  router, `mlp.shared_experts.*`, mHC tensors) needs explicit ranks for a
  readable, stable index.
- `RepackPlanner.isMultimodalTensorName`: harmless no-op for V4 (text-only).
- `GTurboJSON.encodeLayout` + runtime `PackedExpertsLayoutReader`: the
  layout.json schema is name- and dtype-tolerant on read (reader keeps only
  offsets), so 6-slice V4 blobs fit the existing schema with new role keys
  (e.g. `gate`/`gate_scales`, no `_biases`). Size check: 43 layers x 256
  experts x ~(6 tensors x ~60 B + overhead) is roughly 4-8 MB, under both
  the writer's implicit budget and the reader's
  `PackedExpertsLayoutReader.defaultMaxBytes` 16 MB cap, but within 2x of it
  — worth an explicit assertion in tests.
- Runtime `ResidentIndex` 72-byte entry format: weight + scale + bias
  offsets with dtype byte 0-3 (`U32/BF16/FP16/FP32`). It can physically hold
  FP8 weights + F32 scales + zero biases, but dtype codes for `F8_E4M3`,
  `F32` scales-as-F32, and FP4 need new enum values on both writer
  (`SourceTensor.Dtype` in `TensorMetadata.swift`, `Safetensors` parser
  switch) and reader (`ResidentBuffer`/whoever interprets `dtype`).

## 3. Concrete change list

Ordered. Each item is independently testable; items 1-5 are repacker-only,
item 6 touches the runtime schema, item 7 is the gate.

1. **Source descriptor for V4-Flash.** Generalize `SupportedModelSource`
   into a small descriptor type (repoID, revision, index SHA-256,
   download/installed/reserve bytes, display name) with two instances:
   Gemma (unchanged values) and `deepseek-ai/DeepSeek-V4-Flash`
   (approximate download ~147 GB: ~139 GB experts + ~8 GB resident + index;
   reserve stays 1 GB). Add a `--model v4flash` (or `--repo`) flag in
   `Command/main.swift`; default remains Gemma. Add the V4 index SHA-256 to
   `SourceFingerprint.knownFingerprints` after validating against a fresh
   upload. Update usage text.
2. **Quant metadata loader for the DeepSeek scheme.** Extend
   `IndexLoader.load` to accept either MLX `quantization` or DeepSeek
   `quantization_config` (`quant_method: "fp8"`, `fmt: "e4m3"`,
   `weight_block_size: [128, 128]`, plus the FP4 expert descriptor, exact
   keys TBD from the real config). Replace `QuantSpec(bits:)` with an enum:
   `.mlxAffine(bits:)` vs `.fp8BlockwiseE4M3(blockRows: 128, blockCols: 128,
   scaleType: .ue8m0)` vs `.fp4Expert(scaleLayout: TBD)`. Thread it through
   `IndexLoader.quantSpec`, `RepackPlan.baseMode`, and the manifest writer.
3. **Arch loader fork.** Add a DeepSeek branch to `ArchInfo.load` (or a
   sibling `ArchInfoV4`) reading the flat config.json keys:
   `num_hidden_layers=43`, `hidden_size=4096`, `n_routed_experts=256`,
   `num_experts_per_tok=6`, `moe_intermediate_size=2048`,
   `vocab_size=129280`, plus attention/residual fields needed later by
   V4F-03. Keep the Gemma `text_config` path untouched.
4. **Safetensors dtype extension.** Add `F8_E4M3` (1 byte), `F32` scale
   tensors, and whatever the FP4 container dtype is (`U8` packed or a
   safetensors F4 type — confirm from the actual shard headers) to
   `SourceTensor.Dtype` and the `Safetensors.parseHeaderBytes` switch, with
   correct `elementBytes` and a shape*elementBytes check that tolerates
   sub-byte packing.
5. **Planner fork for classification and slicing.**
   - Generalize `RepackPlanner.classify` to accept the `model.` prefix and
     the V4 routed-expert name pattern; keep `.unknown` throwing.
   - Support per-expert 2D source tensors (if that is what the index shows):
     group `model.layers.N.mlp.experts.E.{gate,up,down}_proj.*` by (layer,
     role, expert) and emit one `PerExpertTensorSlice` per expert from the
     per-expert `SourceTensor` (`sourceOffsetPerExpert` becomes per-tensor
     absolute offsets; the slice struct needs a `[UInt64]` offsets array
     instead of a single stride, or keep stride when the source is a
     stacked rank-3 tensor — decide after reading the real index).
   - Blob layout for V4: per role, `weights` (FP4 packed) + `scales`
     (FP4/FP8 block scales), no biases: 6 slices, then
     `roundUpToPage(blobCursor)`. Expected blob ~12.6 MB + scales; layer
     file = 256 x stride (~3.3 GB). `GTurboLayoutValidator` already enforces
     the invariants.
   - Resident file: all non-expert tensors (attention, router, shared
     expert, norms, embed_tokens, lm_head — untied) as FP8 weights with F32
     `weight_scale_inv` block scales and `biasSize = 0`. Extend
     `planResidentFile` to pair `base + ".weight_scale_inv"` instead of
     `.scales`/`.biases` when the quant spec is `.fp8BlockwiseE4M3`, and to
     record logical shape without the `32/bits` unpack for 8-bit elements.
   - Extend `slotRank`/`lmResidentOrdering` with V4 names and update
     `writeManifest`'s bit-width probes (embedding 8, attention 8, router
     TBD, sharedExpert 8, routedExpert 4) — or better, derive the quant
     table from the per-tensor specs instead of name probes.
6. **Manifest + runtime schema versioning.** This is the one item that
   crosses into `Sources/TurboFieldfare/`, and it should stay minimal for
   V4F-01 (load-ability, not execution):
   - Add a `manifest.modelFamily` or bump `versionMinor` to 1 so V4 manifests
     can carry a DeepSeek arch dict and new quant slots (`fp4mx`-style
     routedExpert, `fp8e4m3-blockwise` others) without breaking Gemma
     validation.
   - Extend `ManifestArch` with optional fields (or a second decodable
     struct selected by family), relax `validateQuant` per family, and add
     `ArchConfig.deepseekV4Flash` alongside `gemma4_26B_A4B`. `Model.load`'s
     `expecting:` parameter already exists; the app default stays Gemma.
   - Add dtype codes for FP8/FP4/F32-scales to the resident index contract
     (writer `GTurboBinary`, reader `ResidentIndexReader`) — codes only;
     dequant kernels are V4F-02.
7. **Gate verification.** Round-trip test: repack, then for a sample of
   coalesced ranges (plus one full expert blob and one resident FP8 tensor),
   digest the installed bytes against independently fetched source bytes.
   Reuse `HTTPRangeSourceByteProvider.destinationDigest` and the existing
   audit. Add a scratch-bound assertion (`BoundedScratch.defaultLimitBytes`,
   512 KB) mirroring IO-10. Do this against a synthetic fixture first
   (small fake V4 index + shards served by the existing local test HTTP
   machinery, if present) before any 147 GB live run; the live run is the
   final gate, not the dev loop.

Also mechanical: `RemoteStreamingRepacker` passes `layoutMode: "identity"`
and the audit field `packedExpertLayoutMode`; keep identity. Layer file
names `layer_%02d.bin` satisfy 43 layers unchanged. Tokenizer sidecar list
(`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
`chat_template.jinja/json`) already covers V4's files; confirm V4's chat
template location (likely inside `tokenizer_config.json`).

## 4. Risks and unknowns

1. **FP4 on-disk encoding (highest risk).** How V4-Flash stores FP4 expert
   weights in safetensors — container dtype, element order within a byte,
   scale dtype and blocking — is not yet confirmed from real shard headers.
   The repacker copies bytes unchanged, so the blob layout only needs the
   container's byte size per expert, but `elementBytes * shape == size`
   validation and the layout.json `dtype` field both need the truth. Read
   two shard headers before writing item 4/5 code.
2. **Stacked vs per-expert source tensors.** DeepSeek convention is
   per-expert 2D tensors. If so, the plan emits 256 x 3 x 2 = 1536 source
   tensors per layer instead of 9, and `RangeCopyPlanner.coalesce` sees
   ~230k small copies. Coalescing handles this, but range-request count and
   `remoteGapBytesDownloaded` (over-fetch inside 512 KB chunks covering
   interleaved non-expert data) need measurement; worst case the chunk
   policy or copy order needs a V4 tweak. Shard layout of expert tensors
   (contiguous per expert vs interleaved) decides this.
3. **FP8 scale tensor naming/shape.** `weight_scale_inv` naming and its
   `[rows/128, cols/128]` shape are the DeepSeek V3 convention; V4-Flash
   with ue8m0 scales may differ. Confirm from the index.
4. **Index size.** A ~230k-entry `weight_map` JSON may exceed the current
   4 MB cap in `RemoteSnapshotLoader.fetchSmallFile` for
   `model.safetensors.index.json`. Raise the cap deliberately (bounded,
   e.g. 64 MB) rather than discovering it at runtime.
5. **layout.json near its reader cap.** ~4-8 MB estimate vs 16 MB
   `PackedExpertsLayoutReader.defaultMaxBytes`. Fine today; assert in the
   validator so a schema fattening fails loudly.
6. **Resume/checkpoint interplay.** A V4 plan fingerprint differs from any
   Gemma one, so old partials are correctly rejected; but the output
   directory lock is per-directory, so nothing prevents pointing V4 at the
   Gemma output path. The `--model` flag must feed
   `inspectPersistentInstall`'s repoID check consistently (it already
   compares `repoID`, so this works if the flag is threaded everywhere).
7. **Disk budget.** ~147 GB install plus ~0.5 GB chunk scratch on top of
   whatever the host holds; `DiskSpaceChecker` handles it, but the operator
   facing docs (README, playbook) still say ~15 GB. Doc update belongs to
   this stage's PR.
8. **First-3-layers hash routing.** Repack treats them identically to other
   MoE layers (correct), but the runtime later needs to know which layers
   are hash-routed. The manifest/arch dict should carry that fact now
   (cheap) rather than retrofit it during V4F-04.
9. **MTP module tensors.** The +1 MTP layer's tensors will appear in the
   index. They must be classified and either dropped explicitly (recorded
   in the audit like multimodal drops) or packed; decide deliberately, do
   not let them fall into `.unknown` and abort the install.
10. **Layer-0/1/42 attention anomaly** (`compress_ratios = 0`, carried to
    V4F-03) does not affect repack bytes but affects which tensors exist per
    layer; if those layers have different tensor sets, the resident planner
    must not assume uniform per-layer tensor names. Confirm from the index.

## 5. Effort estimate (days per work item)

| # | Item | Days | Notes |
| --- | --- | ---: | --- |
| 0 | Read real V4 index + 2 shard headers + config.json (no download of payloads) | 0.5 | Resolves risks 1-3, 9; prerequisite for everything |
| 1 | Source descriptor + `--model` flag + fingerprint entry | 0.5 | Mechanical; fingerprint waits on a validated upload |
| 2 | DeepSeek quant metadata parsing (`IndexLoader`, `QuantSpec` enum) | 1 | Needs item 0's facts |
| 3 | DeepSeek arch loader branch | 0.5 | Flat keys, well documented by V4F-00 |
| 4 | Safetensors dtype extension (F8_E4M3, F32, FP4 container) | 0.5-1 | Depends on risk 1 resolution |
| 5 | Planner fork: classification, per-expert grouping, 6-slice blob, resident FP8 pairing, ordering | 2-3 | Largest repacker item; per-expert-vs-stacked fork decides the low/high end |
| 6 | Manifest/runtime schema versioning (family, quant slots, arch config, dtype codes) | 1-2 | Crosses into runtime; keep to load-ability |
| 7 | Gate: synthetic fixture round-trip + digest verification + scratch bound, then live 147 GB run | 1-2 | Live run is wall-clock dominated (download time), not engineering time |
| | **Total** | **7-11.5** | Excludes V4F-02/03 kernel work entirely |

## 6. Ten-line summary of the change list

1. Generalize `SupportedModelSource` into per-model descriptors; add `deepseek-ai/DeepSeek-V4-Flash` (~147 GB) and a `--model` CLI flag, default Gemma.
2. Add the V4 weight-index SHA-256 to `SourceFingerprint.knownFingerprints` after validating a fresh upload.
3. Teach `IndexLoader` the DeepSeek `quantization_config` (FP8 e4m3 128x128 + FP4 experts); replace `QuantSpec(bits:)` with a per-scheme enum.
4. Add a flat-config DeepSeek branch to `ArchInfo.load` (43 layers, hidden 4096, 256 experts, top-6, intermediate 2048, vocab 129280).
5. Extend `SourceTensor.Dtype` and the `Safetensors` parser for `F8_E4M3`, `F32` scales, and the real FP4 container dtype.
6. Fork `RepackPlanner.classify` for `model.` prefixes and V4 expert names; support per-expert 2D source tensors grouped by (layer, role, expert).
7. Emit 6-slice FP4 expert blobs (weights + scales, no biases) at a page-aligned fixed stride, 256 per layer, 43 layer files; validator unchanged.
8. Plan the resident FP8 common file by pairing `.weight` with `.weight_scale_inv` (F32, no biases), including untied `lm_head`; extend `slotRank` for V4 tensor names.
9. Version the manifest/layout schema by model family; relax `ManifestReader.validateQuant` per family and add `ArchConfig.deepseekV4Flash`; add FP8/FP4 dtype codes to the resident index contract.
10. Verify the gate on a synthetic fixture first (digest round-trip, 512 KB scratch bound), then run the live ~147 GB install; classify and explicitly drop the MTP tensors, recorded in the audit.
