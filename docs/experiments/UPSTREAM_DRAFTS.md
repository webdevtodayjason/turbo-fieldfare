# Upstream contribution drafts

Staged candidates for upstream PRs, in submission order. Status column
tracks where each stands. See the porting playbook for the full context.

## 1. `layerIndex` anchor robustness (draft, ready to stage)

**Status:** drafted 2026-08-02; code exists on branch `v4f` intertwined
with the V4 port; needs extraction to a standalone commit on main.

**Why narrow and safe:** no behavior change for any current Gemma tensor
name; pure robustness + a regression test. Fits CONTRIBUTING's "focused
fixes" rule and serves as the good-faith first contact before the larger
multi-model-family conversation.

**Problem:** `RepackPlanner.layerIndex(in:)` anchors on the first
occurrence of the substring `.layers.` anywhere in the tensor name and
parses the digits after it. Any current or future tensor whose name
contains `.layers.` in a non-layer position (nested module names,
expert-grouped names, flat names like `layers.N.…` with no prefix) is
misparsed or silently unclassified. Found while adding a second model
family whose flat tensor names (`layers.0.…`) needed
`v4LayerIndex` prefix parsing instead.

**Patch sketch (main-branch shape):**

```swift
private static func layerIndex(in name: String) -> Int? {
    // Anchor on the layer component only: a ".layers.<N>." segment
    // whose digits run to the next dot. Reject anything else instead
    // of parsing digits out of an arbitrary substring.
    for part in name.split(separator: ".") {
        if part == "layers" { continue }
        ...
    }
}
```

(Final implementation: scan path components, find the `layers`
component, parse the following component as an integer, require the
expected Gemma prefix structure; return nil otherwise. Include a test
with adversarial names: `foo.layersx.1`, `layers.0.attn`,
`a.layers.2.layers.3.b`.)

**PR description draft:**

> Title: Make repack layerIndex parsing anchor-safe
>
> `layerIndex(in:)` currently finds the first `.layers.` substring and
> parses digits after it. Names containing that substring in a
> non-layer position misparse. This PR anchors on the path component
> structure instead and adds adversarial-name tests. No behavior change
> for the pinned Gemma checkpoint (all its names parse identically).

## 2. Experimental DeepSeek V4-Flash support (issue or draft PR)

**Status:** validated experimentally on branch `v4f`; see
[`V4F_VALIDATION.md`](V4F_VALIDATION.md). The 145 GB repacked model loads and
generates coherently on a 128 GB M5 Max. Clean greedy short-context decode
measured 5.755 tok/s over 48 new tokens and 6.022 tok/s over 24. The complete
V4 filter passes 162 tests across 23 suites, including real-checkpoint kernel
goldens.

**Claim boundary:** compressed-memory long-context retrieval is not fully
supported. A natural 231-token retrieval prompt returns the correct answer
immediately, but a synthetic 2,353-token prompt reproducibly returns the wrong
season. Present this as an experimental second model family, with the failing
fixture and result included rather than hidden.

Recommended first contact is an issue or draft PR because the arch gate is a
maintainer-owned pinned contract. Candidate review slices remain: safetensors
dtype extension -> dual-family quant parsing -> per-expert planner fork ->
kernel pack -> runtime family branch. If maintainers prefer a narrower first
change, extract the `layerIndex` robustness fix above before proposing the full
series.

**PR summary draft:**

> Adds experimental DeepSeek V4-Flash support on Apple Silicon. The 145 GB
> repacked model loads and generates coherently on a 128 GB Mac. On the tested
> M5 Max, clean greedy short-context decode measured approximately 5.8-6.0
> tok/s. Focused GPU correctness tests and short-context generation pass.
> Compressed-memory long-context retrieval remains experimental: natural
> retrieval succeeds at 231 tokens, while a reproducible synthetic retrieval
> test fails at 2,353 tokens. Long-context inference is not represented as
> fully supported.
