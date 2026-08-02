# DeepSeek V4-Flash experimental validation

Date: 2026-08-02

## Status

DeepSeek V4-Flash now repacks, loads, and generates coherently through the
TurboFieldfare runtime on Apple Silicon. The implementation is suitable for an
**experimental** contribution. It is not ready to claim complete long-context
support.

Accurate contribution summary:

> Adds experimental DeepSeek V4-Flash support on Apple Silicon. The 145 GB
> repacked model loads and generates coherently on a 128 GB Mac. On the tested
> M5 Max, clean greedy short-context decode measured 5.755 tok/s over 48 new
> tokens, with a 24-token run measuring 6.022 tok/s. Focused GPU correctness
> tests and short-context generation pass. Compressed-memory long-context
> retrieval remains experimental: a natural 231-token retrieval prompt returns
> the correct answer immediately, while a reproducible synthetic 2,353-token
> retrieval prompt returns the wrong season. Long-context inference is not
> represented as fully supported.

## Validated system

- Branch: `v4f`
- Validation base commit: `036ae3e7f746d238e09b1212530292b8c32fcc05`
- Hardware: Apple M5 Max, 128 GB unified memory
- macOS: 26.5.2
- Swift: 6.3.2
- Model source revision: `60d8d70770c6776ff598c94bb586a859a38244f1`
- Installed artifact: `scratch/deepseek-v4-flash.gturbo`, 145 GB
- Model receipt, manifest, and packed-expert layout present
- Disk available during final runs: 798-804 GiB
- Memory free before final runs: 56-65%

## Reference-alignment work in the final validation diff

The final correctness pass closed four differences from the official reference:

1. Every completed compressed group is visible to attention. Compressed groups
   intentionally overlap the 128-token window and must not be filtered as
   duplicate coverage.
2. CSA previous gate accumulators begin at negative infinity, matching the
   overlap compressor's initial score state.
3. Lightning-indexer queries and compressed entries use the normalized
   Hadamard transform plus block-32 FP4 quantize-dequantize simulation.
4. Window KV uses the reference block-64 FP8 activation
   quantize-dequantize simulation on its first 448 channels. Compressor pooled,
   normalized, and rotated values are rounded through BF16 at the same points as
   the reference.

The window-KV RMSNorm, trailing RoPE, and FP8 simulation are fused into one
Metal dispatch for decode. The fused kernel retains the FP16 storage boundary
between normalization and the dependent stages. The existing CPU golden covers
that complete chain. This recovered the reference-faithful 24-token decode rate
from 4.109 tok/s to 6.022 tok/s on the validation host.

## Automated validation

### Complete package

Command:

```bash
Scripts/test.sh
```

Result: **709 tests in 134 suites passed** in 314.542 seconds on the
exact final kernel tree. This includes the Gemma path and all V4 coverage.

### Complete V4 filter

Command:

```bash
Scripts/test.sh --filter V4
```

Result: **162 tests in 23 suites passed** in 255.304 seconds.

This includes:

- repack payload and manifest round trips
- FP4 expert and FP8 resident kernels
- real checkpoint byte/golden tests
- router and routed-MoE tests
- CSA/HCA compressor and attention tests
- mHC, RoPE, window-ring, and chunked-prefill tests
- tokenizer, chat formatting, and tool-call parsing
- the fused window norm/RoPE/FP8 QAT decode chain

Release build:

```bash
swift build -c release
```

Result: exit 0.

## Real-checkpoint generation

All reported runs were greedy (`--temperature 0`), used one model process at a
time, and ran with the trusted verified-install receipt and the validated
256-token chunked-prefill path:

```bash
TURBO_V4_TRUSTED_INSTALL=1 \
TURBO_V4_CHUNKED_PREFILL=1 \
TURBO_V4_PREFILL_CHUNK_TOKENS=256 \
.build/release/TurboFieldfareCLI \
  --model scratch/deepseek-v4-flash.gturbo \
  --prompt "$PROMPT" \
  --max-new "$MAX_NEW" \
  --temperature 0
```

### Short context

Prompt: `The capital of France is`

Output, 48 new tokens:

```text
 Paris. It is one of the most popular tourist destinations in the world. With over 30 million foreign visitors per year, it is the most visited city in the world. It has some of the largest museums and palaces: The Louvre,
[family=v4flash stop=maxTokens prefill=6tok prefill_s=2.275 new=48tok decode=8.34s tok/s=5.755]
```

A separate 24-token run produced the same prefix and measured 6.022 tok/s:

```text
[family=v4flash stop=maxTokens prefill=6tok prefill_s=2.221 new=24tok decode=3.99s tok/s=6.022]
```

These are measurements of this host and run state, not performance ceilings.

### Natural retrieval at 231 prompt tokens

Fixture: `scratch/v4f-recon/long-prompt.txt`

Eight-token output:

```text
 The festival is held in autumn. Is
[family=v4flash stop=maxTokens prefill=231tok prefill_s=12.143 new=8tok decode=2.33s tok/s=3.426]
```

Disposition: retrieval success. A 24-token continuation later contradicted its
own first answer, so this gate demonstrates access to the answer, not broad
long-answer quality.

### Synthetic retrieval at 2,353 prompt tokens

Fixture: `scratch/v4f-recon/perf/v4f-2k-fictional.txt`

The prompt states that the Bronze Rain Procession is held in autumn. `autumn`
is the only season answer in the prompt. Final-tree output:

```text
 The Bronze Rain Procession is held in the spring. The correct answer is spring. The incorrect answers are summer, autumn
[family=v4flash stop=maxTokens prefill=2353tok prefill_s=120.873 new=24tok decode=5.99s tok/s=4.010]
```

Disposition: reproducible failure. Do not advertise complete compressed-memory
or 2K-plus retrieval support.

Final-query instrumentation showed that CSA layers do select early compressed
groups, including group zero. The failure is therefore not explained by top-k
dropping the beginning of the prompt. Without an official CUDA run or saved
reference intermediates for this exact fixture, the remaining cause cannot yet
be separated confidently between another implementation mismatch and lossy
model behavior under this quantized runtime.

## Measurement and protocol deviations

- The community benchmark guide targets the pinned Gemma path. These V4 runs are
  experimental engineering validation, not community benchmark submissions.
- `TURBO_V4_TRUSTED_INSTALL=1` uses the verified receipt and size checks rather
  than re-hashing the 145 GB artifact on every run.
- `TURBO_V4_CHUNKED_PREFILL=1` and chunk size 256 select the experimental V4
  layer-major prompt path.
- Temperature zero was used for deterministic diagnosis, not the application's
  default sampled policy.
- The first attempt to launch the complete suite as a background task failed in
  the task harness before a process started. The same command passed when run
  directly, and the exact final tree later passed 709 tests in 134 suites.
- Two intermediate model-run requests were accidentally wrapped with duplicate
  parallel entries. The harness returned only one execution in each case, but
  their performance numbers are excluded above. The reported 48-token short run
  and 2,353-token failure were issued as direct sequential commands.

## Upstream disposition

Recommended next step: open an upstream issue or draft PR presenting this as an
experimental second model family. Keep the first review centered on architecture
and ownership boundaries, not a claim of production support.

Suggested scope statement:

- supported experimentally: repack, load, greedy raw completion, short-context
  coherent generation, real-byte kernel validation, chunked prompt prefill
- incomplete: production server/chat integration polish, community holdouts,
  authoritative reference parity for long-context compressed memory
- known limitation: reproducible wrong retrieval at 2,353 prompt tokens

The existing narrow `layerIndex` robustness fix remains a safer standalone first
PR if maintainers prefer incremental review before the multi-model-family branch.
