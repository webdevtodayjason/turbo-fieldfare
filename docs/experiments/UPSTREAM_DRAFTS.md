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

## 2. Multi-model-family support (future RFC, NOT a PR first)

The V4-Flash port (branch `v4f`) is the evidence base. Open an issue
before any PR series; the arch gate is the maintainer's pinned
contract. Candidate slice order if accepted: safetensors dtype
extension -> dual-family quant parsing -> per-expert planner fork ->
kernel pack -> runtime family branch. Do not send before V4F-05
holdout evidence exists (per the community benchmark protocol).
