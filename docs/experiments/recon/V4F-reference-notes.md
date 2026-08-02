# DeepSeek-V4 Reference Inference Notes (recon)

Source: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
(files: `model.py`, `kernel.py`, `convert.py`, `generate.py`, `config.json`, `README.md`, `requirements.txt`)
plus `../encoding/encoding_dsv4.py` + `encoding/README.md`, repo-root `tokenizer_config.json`,
`generation_config.json`, and `DeepSeek-V4-Flash/raw/main/config.json`.

Primary target is **Flash** (43 layers, 256 experts, dim 4096). Where Pro (61 layers, 384 experts,
dim 7168) differs, it is called out. The reference `inference/` tree ships only in the Pro repo; it is
written generically off `ModelArgs`, so it describes both.

Two configs seen:
- `inference/config.json` (Pro): n_layers 61, dim 7168, moe_inter_dim 3072, 384 routed experts, top-6,
  route_scale **2.5**, index_topk **1024**, n_heads 128, q_lora_rank 1536, o_groups 16,
  compress_ratios = `[128, 128, 4, 128, 4, ..., 4, 0]` (layers 0,1 are ratio 128; last layer 0).
  No `n_hash_layers` key -> defaults to 0 in this file.
- `DeepSeek-V4-Flash/config.json`: 43 layers, dim 4096, moe_intermediate 2048, 256 experts, top-6,
  routed_scaling_factor **1.5**, index_topk **512**, n_heads 64, q_lora_rank 1024, o_groups 8,
  o_lora_rank 1024, num_hash_layers **3**, compress_ratios = `[0, 0, 4, 128, 4, ..., 4, 0]`
  (0 at layers 0, 1, 42; ratio 4 on even layers 2..40, ratio 128 on odd layers 3..41).
  Shared: head_dim 512, rope_head_dim 64, window 128, swiglu_limit 10.0, hc_mult 4, sinkhorn 20,
  hc_eps 1e-6, norm eps 1e-6, index_n_heads 64, index_head_dim 128, rope_theta 10000,
  compress_rope_theta 160000, YaRN factor 16, original_seq_len 65536, beta_fast 32, beta_slow 1,
  vocab 129280, 1 MTP layer.

---

## 1. compress_ratio = 0 (Flash layers 0, 1, 42)

In `Attention.__init__`, `self.compress_ratio = args.compress_ratios[layer_id]`; `0` is falsy, so:
- No `Compressor`, no `Indexer` are built.
- `kv_cache` size = `window_size` only (128 entries x head_dim 512).
- Attention runs **pure sliding-window MQA**: topk_idxs = the 128 ring-buffer window slots only
  (`get_window_topk_idxs`), one shared 512-dim KV vector per token, all 64 q heads attend to it.
- **RoPE config changes**: `original_seq_len = 0` (YaRN disabled) and `rope_theta = 10000`
  (base theta, NOT compress_rope_theta). Comment: "disable YaRN and use base rope_theta in pure
  sliding-window attention".
- Everything else (q lora + per-head RMS norm, partial RoPE on last 64 dims, attn_sink, output
  de-rotation, grouped O projection) is identical to other layers.

So layers 0/1/42 are local-only layers with a 128-token receptive field plus the attention sink.

## 2. CSA forward path (compress_ratio = 4, Flash even layers 2..40)

Per-token flow in `Attention.forward`:

**Q path**
- `qr = q_norm(wq_a(x))` — wq_a: dim->q_lora_rank (fp8), RMSNorm(1024). `qr` is reused by the indexer.
- `q = wq_b(qr)` -> [b,s,64,512]; then **weight-free per-head RMS renorm**:
  `q *= rsqrt(mean(q^2, -1) + eps)` over the 512 head dim.
- Partial RoPE on `q[..., -64:]` with the layer freqs_cis (see RoPE note below).

**Shared-KV MQA path (window branch)**
- `kv = kv_norm(wkv(x))` — wkv: dim->512 (single shared vector for all heads: MQA, K==V).
- Partial RoPE on `kv[..., -64:]`.
- **QAT simulation**: `act_quant(kv[..., :-64], block=64, ue8m0, inplace=True)` — non-rope 448 dims
  are quant-dequant simulated to FP8 e4m3 with 64-wide blocks; rope dims stay bf16 "for positional
  precision". Cache stays bf16.

**Compression weights (`Compressor`, overlap mode because ratio==4)**
- `coff = 2`. `wkv`, `wgate`: dim -> 2*512 = 1024, **fp32 Linears**; compression runs entirely in fp32
  (`x = x.float()`).
- `ape`: learned absolute positional embedding [4, 1024] fp32, added to gate scores per position in
  group.
- Pooling: group of 4 tokens -> overlapping 8-token window via `overlap_transform`: the 1024-dim
  projection is split into two 512 halves; the compressed token for group g pools
  `[prev group's 4 tokens (first 512 channels), current group's 4 tokens (second 512 channels)]`,
  i.e. window dim doubled to 8 while channels return to 512.
- `kv_out = sum_{8} softmax(wgate_scores + ape) * wkv_values` (softmax over the 8 window positions).
- Then `RMSNorm(512)`, cast back to bf16.
- RoPE on the last 64 dims at the **group-start position**: prefill `freqs_cis[:cutoff:4]`
  (positions 0, 4, 8, ...), decode `freqs_cis[start_pos + 1 - 4]`. So compressed slot i carries
  RoPE phase of absolute position `4*i`, not `4*i+3`.
- QAT sim on non-rope dims: `act_quant(kv[..., :-64], block=64, inplace=True)`.
- Written into `kv_cache[:, win:]` (compressed region after the 128 window slots).
- Decode state: `kv_state`/`score_state` [b, 8, 1024] fp32 ring; a compressed token is emitted only
  when `(start_pos + 1) % 4 == 0`, then state shifts (current group becomes next overlap group).
  At prefill, the final partial group (< 4 tokens) is stashed in state and the last full group is
  stashed as the next overlap half.

**Lightning indexer (`Indexer`, only exists when ratio == 4)**
- q: `wq_b(qr)` — ColumnParallel q_lora_rank -> 64 heads * 128, from the **shared** `qr` (same
  q-lora as main attention). Partial RoPE on last 64 dims. Then **randomized Hadamard rotation**
  (`hadamard_transform(x, scale=d^-0.5)`, requires bf16) and **FP4 QAT sim**
  (`fp4_act_quant(block=32, inplace=True)`).
- Own `Compressor(ratio=4, head_dim=128, rotate=True)`: same overlap machinery but 128-dim; after
  norm+RoPE it applies Hadamard + FP4 sim (`rotate=True` branch) instead of FP8-sim. Indexer
  compressed KV cache: [b, max_seq_len/4, 128] bf16.
- Per-head weights: `weights_proj(x)` — bf16 Linear dim->64, scaled by
  `softmax_scale * n_heads^-0.5` where `softmax_scale = 128^-0.5`.
- Score: `index_score = einsum("bshd,btd->bsht", q, kv_cache)`; `relu_()`; multiply by
  per-head weights; **sum over heads** -> [b, s, t_compressed]. (ReLU-then-weighted-sum, not
  softmax — this is the "lightning" scoring.)
- Causal mask on prefill: compressed slot j visible to token t iff `j < (t+1)//4`... precisely
  `j >= ceil((t+1)/4)` masked; implemented as `arange(s//4) >= arange(1,s+1).unsqueeze(1)//4`.
- `topk(min(index_topk=512, end_pos//4))` indices. Indices are offset by `offset`
  (= seqlen at prefill, = win=128 at decode) so they address the compressed region of the
  concatenated KV layout.

**Sliding-window branch merge**
- `topk_idxs = cat([window_topk (128 entries, ring positions), compress_topk (<=512)], -1)`, int32.
- Prefill attends directly over `kv_cat = [full window kv (seqlen), compressed kv]`; the ring buffer
  is only written (in rotated layout: `cache[cutoff:win], cache[:cutoff] = kv[-win:].split(...)` with
  `cutoff = seqlen % win`) for later decode steps. Decode attends over
  `kv_cache = [ring window (128), compressed region]`.
- **One** `sparse_attn` call handles both branches: gather rows by index (-1 -> zero KV and -inf
  score), online-softmax FlashAttention-style, `scale = 512^-0.5` applied after the QK gemm.

**Attention sink**
- Learnable per-head fp32 vector `attn_sink [n_heads]`. In the kernel it enters only the
  **denominator**: `sum_exp[h] += exp(attn_sink[h] - running_max[h])`. Equivalent to a virtual key
  with score `attn_sink[h]/scale`... precisely with logit `attn_sink[h]` (after the softmax_scale is
  applied to QK scores; the sink term is NOT multiplied by scale) and value 0. Note the running max
  never includes the sink.

**Partial RoPE and output de-rotation**
- Only the last 64 of 512 dims are rotated, for q and the shared kv.
- Because V == K here, the attention output's last 64 dims are a convex mix of key-position-rotated
  vectors. The reference then applies `apply_rotary_emb(o[..., -64:], freqs_cis_of_query, inverse=True)`
  — complex-conjugate (de-rotation) at the **query position's** phase. This is exact only if all
  selected keys shared the query's position; in general it leaves a residual rotation of
  `key_pos - query_pos` per component, which the model learns through. Implement exactly as written:
  conjugate-multiply with the current token's freqs.

**Grouped output projection**
- `o` reshaped [b, s, o_groups=8, (64*512)/8 = 4096].
- `wo_a` weight [8*1024, 4096] viewed [8, 1024, 4096]; `o = einsum("bsgd,grd->bsgr", o, wo_a)` —
  per-group low-rank down-projection to o_lora_rank 1024. **wo_a is bf16** (convert.py dequantizes
  its FP8 checkpoint form by folding the 128x128 block scales into the weights).
- `wo_b`: RowParallel 8*1024 -> dim (fp8), sums the group low-rank outputs.

## 3. HCA forward path (compress_ratio = 128, Flash odd layers 3..41)

Differences vs CSA:
- `overlap = False` (coff = 1): wkv/wgate are dim->512 fp32; non-overlapping groups of 128 tokens,
  softmax-pooled with `ape [128, 512]`. One compressed token per 128 tokens.
- **No indexer** (`indexer = None`). The compressed top-k is `get_compress_topk_idxs` = **all**
  compressed slots up to the causal bound — i.e. dense attention over the entire compressed stream,
  not sparse selection. (topk = window 128 + seqlen/128 compressed entries.)
- Compressed RoPE positions stride by 128 (group-start position, same rule as CSA).
- RoPE: same as CSA — whole layer (window KV included) uses compress_rope_theta 160000 + YaRN.
- Same QAT sim (FP8 block-64) on compressed kv non-rope dims, same sink, de-rotation, O projection.

RoPE note for both CSA/HCA: because `compress_ratio` is truthy, `Attention.freqs_cis` is built with
`rope_theta = compress_rope_theta (160000)` **and YaRN** (factor 16, original_seq_len 65536,
beta_fast 32 / beta_slow 1, smooth ramp `freqs/factor*(1-smooth) + freqs*smooth`). This freqs table
is shared by q, window kv, compressed kv, and the indexer. Only ratio-0 layers use theta 10000
without YaRN.

## 4. mHC (manifold Hyper-Connections) exact computation

State: hidden is [b, s, hc=4, dim] throughout the trunk; the embedding output is broadcast to all 4
copies at entry.

Per sublayer (attn and ffn each have their own params), all in fp32:
- Params: `hc_fn [(2+hc)*hc = 24, hc*dim]`, `hc_base [24]`, `hc_scale [3]`.
- Flatten x to [b,s,4*dim]; compute `rsqrt = rsqrt(mean(x^2, -1) + norm_eps)` (norm_eps = 1e-6,
  **not** hc_eps).
- `mixes = (x @ hc_fn^T) * rsqrt` -> [b,s,24].
- Split (kernel `hc_split_sinkhorn`):
  - `pre[j]  = sigmoid(mixes[j]      * scale[0] + base[j])      + hc_eps`   (j in 0..3)
  - `post[j] = 2 * sigmoid(mixes[4+j] * scale[1] + base[4+j])`              (no eps, factor 2)
  - `comb[j,k] = mixes[8 + j*4+k] * scale[2] + base[8 + j*4+k]` -> [4,4]
  - Sinkhorn on comb: (1) row-softmax (max-subtracted exp, normalized) `+ hc_eps`;
    (2) column normalize `comb / (col_sum + hc_eps)`;
    (3) repeat 19 more times: row normalize `comb / (row_sum + hc_eps)`, then column normalize.
    Total: 1 softmax + 20 column normalizations + 19 row normalizations, hc_eps = 1e-6 in every
    denominator. Not a textbook doubly-stochastic Sinkhorn (eps-biased).
- `hc_pre`: branch input `y = sum_j pre[j] * x[j]` -> [b,s,dim]; then RMSNorm; then the sublayer.
- `hc_post`: `out[k] = post[k] * sublayer_out + sum_j comb[j,k] * residual[j]` for k in 0..3.
  (Note the index order: output copy k gathers comb[:, k] over residual copies j.)

Final head (`ParallelHead.hc_head`): same mixes machinery with its own [4, 4*dim] fn / [4] base /
[1] scale, but **pre-only**: `pre = sigmoid(mixes*scale + base) + hc_eps`, `y = sum pre*x`, then
final RMSNorm, then lm_head (fp32 weight, logits from last position in fp32).

MTP block (1 layer): `x = e_proj(enorm(embed(ids))).unsqueeze(2) + h_proj(hnorm(x))`
(embedding projected and broadcast-added to each of the 4 hc copies), then a normal Block, then its
own hc_head -> shared lm_head. Shares embed/head with the main model.

## 5. MoE router

`Gate.forward(x, input_ids)`:
- `scores = x.float() @ W_router.float()^T` (router weight stored bf16, computed fp32), [tokens, 256].
- sqrtsoftplus: `scores = sqrt(softplus(scores))`. (softmax / sigmoid variants exist but unused.)
- `original_scores = scores` saved.
- **noaux_tc bias at inference is fully static**: `scores = scores + bias` for selection only. The
  reference inference code contains **no bias update logic whatsoever** (no EMA, no load
  balancing) — the bias is a frozen checkpoint parameter (`e_score_correction_bias` renamed to
  `bias` by convert.py).
- Selection: `indices = topk(scores + bias, k=6)` (or hash table, below).
- Weights: `weights = original_scores.gather(indices)` — bias does **not** affect weights.
- norm_topk_prob: since score_func != softmax, `weights /= weights.sum(-1, keepdim=True)`
  (matches Flash `norm_topk_prob: true`).
- `weights *= route_scale` (1.5 Flash / 2.5 Pro).
- Shared expert: exactly 1, always active, output added **unweighted**.
- Expert FFN (SwiGLU, fp32 internal): `gate = w1(x).float()`, `up = w3(x).float()`;
  with swiglu_limit = 10.0: `up = clamp(up, -10, 10)`, `gate = clamp(gate, max=10)` (note: gate has
  no lower clamp); `h = silu(gate) * up`; routing weight multiplied in before `w2`
  (`w2((weight * h).to(dtype))`).
- Reference dispatch loops experts in Python with `bincount` — a Metal port should use a real
  gathered/scatter MoE kernel; semantics are plain token-choice top-6.

## 6. Hash routing (Flash layers 0, 1, 2)

There is **no computed hash function**. `Gate` for `layer_id < n_hash_layers` holds
`tid2eid`: an int32 parameter of shape **[vocab_size=129280, n_activated_experts=6]** loaded from
the checkpoint — a fixed per-token-ID expert assignment table.
- `indices = tid2eid[input_ids]` (token id -> its 6 experts).
- Routing **weights are still score-based**: router logits -> sqrtsoftplus -> gather at the hash
  indices -> normalize by sum -> * route_scale. Hash layers have **no bias parameter**.
- Note: hash MoE layers coincide with compress_ratio 0 (layers 0,1) and the first CSA layer (2).
- convert.py quirk: the no-`.weight` key check lists `"tie2eid"` (typo of `tid2eid`); harmless either
  way (falls through to the `split(".")[-2]` branch and passes the tensor through unchanged), but
  grep for both spellings when parsing checkpoints.

## 7. Quantization formats on disk

**FP8 (everything except experts)** — HF checkpoint + converted form:
- Weights: e4m3 [out, in]; scales named `weight_scale_inv` -> renamed `scale`,
  shape `[ceil(out/128), ceil(in/128)]`, dtype **e8m0 (ue8m0: power-of-2 only)**. One scale per
  128x128 block.
- Activations: dynamic per-128-block-along-K quantization at every `linear()` call:
  `scale = amax/448` floored at amax >= 1e-4, rounded **up to a power of 2** when ue8m0
  (`fast_round_scale` = bit-twiddled ceil(log2)). GEMM accumulates FP32 with a separate
  scale-corrected accumulator per 128 K-block.
- `wo_a` is dequantized to bf16 at convert time (scales folded in). Compressor wkv/wgate, RMSNorm
  gammas, hc_*, attn_sink, ape, lm_head are fp32; embed + weights_proj bf16; router weight bf16.

**FP4 experts** (Flash `expert_dtype: fp4`):
- Storage: `[out, in//2]` of `float4_e2m1fn_x2` — 2 e2m1 values packed per byte along K.
  In the HF safetensors this is raw **int8**; convert.py just reinterprets
  (`.view(torch.float4_e2m1fn_x2)`). Nibble order: **low nibble = element 2i, high nibble =
  element 2i+1** (from `cast_e2m1fn_to_e4m3fn`: `low = x & 0x0F; high = x >> 4`, stacked low-first).
- e2m1fn value table: `{0, .5, 1, 1.5, 2, 3, 4, 6}` and negatives (sign bit 8).
- Scales: `[out, in//32]` **e8m0**, one power-of-2 scale per 32 elements along K.
- GEMM (`fp4_gemm`): activations quantized to FP8 e4m3 per-128-block (same act_quant), FP4 weights
  cast FP4->FP32->FP8 on the fly, FP8xFP8 gemm per 32-wide K block, accumulator scaled by
  `act_scale[k//4] * weight_scale[k]` (act scale group = 4 weight blocks).
- FP4 act QAT sim floors amax at `6 * 2^-126` before power-of-2 rounding.
- Optional FP8 expert mode: `cast_e2m1fn_to_e4m3fn` folds the per-32 e8m0 scales into e4m3 values
  via a per-128x128-block offset (`MAX_OFFSET_BITS = 6`, since 6.0 * 2^6 = 384 < 448), producing
  standard FP8 128x128-block weights. Not needed if native FP4 is implemented.

**KV QAT sims to match (bf16 storage, quant-dequant only):**
- window/compressed kv non-rope dims: FP8 e4m3, block 64, ue8m0.
- indexer q and indexer compressed kv: Hadamard transform (scale d^-1/2) then FP4 e2m1, block 32.
  Hadamard requires a `fast_hadamard_transform`-equivalent; a normalized Walsh-Hadamard matmul is
  fine in Metal.

## 8. Tokenizer / chat template / tool calls

- Repo root: `tokenizer.json` (6.4 MB, PreTrainedTokenizerFast BPE) + `tokenizer_config.json`.
  **No `chat_template`** in tokenizer_config; `add_bos_token: false`, bos id 0
  `<｜begin▁of▁sentence｜>`, eos id 1 `<｜end▁of▁sentence｜>` (also pad). `generation_config.json`:
  do_sample, temperature 1.0, top_p 1.0 (generate.py CLI default 0.6; sampling = softmax then
  exponential-race Gumbel-max argmax).
- Chat format lives entirely in `encoding/encoding_dsv4.py` (`encode_messages`,
  `parse_message_from_completion_text`):
  - `<bos>{system}<｜User｜>{user}<｜Assistant｜></think>{response}<eos>` in chat mode —
    note the literal `</think>` immediately after the Assistant prefix to suppress thinking.
    Thinking mode uses `<think>` instead and strips prior-turn reasoning unless tools are present.
  - Roles: system (raw content, tools/response_format appended), developer (rendered as user with
    `<｜User｜>` prefix; internal search pipeline only), user, assistant, latest_reminder
    (`<｜latest_reminder｜>`), and **no tool role** — tool results are merged into user messages as
    `<tool_result>...</tool_result>` content blocks, sorted by call order.
  - Tool calls use **DSML**: assistant emits `\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke
    name="FN">\n<｜DSML｜parameter name="P" string="true|false">VALUE</｜DSML｜parameter>\n...
    </｜DSML｜invoke>\n</｜DSML｜tool_calls>` then EOS. `string="true"` => raw string;
    `string="false"` => JSON. Tool schemas are injected into the system prompt under `## Tools`.
  - Quick-instruction task tokens (`<｜action｜>`, `<｜query｜>`, `<｜title｜>`, `<｜authority｜>`,
    `<｜domain｜>`, `<｜read_url｜>`) appended after messages for internal classification tasks.
  - `reasoning_effort="max"` prepends a fixed "Reasoning Effort: Absolute maximum..." prefix.

## Ambiguities / surprises worth knowing before a Metal port

1. `inference/` exists only in the **Pro** repo; Flash config values must be merged in by hand (done
   above). The two repos disagree on which layers are "special": Pro has ratio 128 on layers 0,1 and
   0 only on the last layer; Flash has 0 on 0,1,42. The reference code handles all three ratio cases
   uniformly, so this is just config.
2. `n_hash_layers` is absent from the shipped Pro inference config.json (defaults 0); Flash needs 3
   and requires the `tid2eid` checkpoint tensor, whose exact safetensors key spelling should be
   verified (convert.py contains the `tie2eid` typo).
3. Attention sink is denominator-only and is **not** multiplied by softmax_scale and not included in
   the running max — subtle but must match.
4. Compressed tokens take the RoPE phase of their group's **first** token position (stride ratio,
   offset 0), and de-rotation of the output uses the **query** position — an approximation the model
   was trained with; do not "fix" it.
5. All compressed-attention layers (ratio 4 **and** 128) use compress_rope_theta 160000 + YaRN for
   the *entire* layer including the 128-token sliding window; ratio-0 layers use plain theta 10000.
6. Q has an extra weight-free RMS renorm after wq_b; indexer q does not.
7. Ratio-4 compressed vectors pool an 8-token overlapping window but split channels: previous group
   contributes the first 512 channels of wkv/wgate output, current group the second 512. Do not
   pool 8 tokens of a single 1024-dim projection.
8. `get_window_topk_idxs` ring-buffer order and the prefill rotated write must match exactly, or
   decode will attend to permuted window entries.
9. Reference MoE/attention are QAT-simulated (quant-dequant) in bf16 in several places
   (kv, indexer) — for numerical parity replicate the sims; for pure throughput they can be dropped
   at a small quality risk.
10. mHC Sinkhorn normalization order is softmax -> col -> 19x(row, col), all eps-biased; and
    `hc_post` gathers comb[:, k] (column) per output copy. pre/post/comb come from one 24-wide
    projection of the RMS-normalized flattened hc state.
11. requirements: torch>=2.10, transformers>=5.0, tilelang==0.1.8, fast_hadamard_transform —
    reference targets CUDA + tilelang; nothing here is portable as-is, but all math is simple.
