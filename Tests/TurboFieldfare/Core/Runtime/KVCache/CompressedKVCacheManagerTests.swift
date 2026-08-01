import Testing
import Foundation
import Metal
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

/// V4F-03 milestone 1: CompressedKVCacheManager capacity math, group
/// bookkeeping, coverage disjointness, FP8-split round-trip, and reset
/// semantics on synthetic data. No model process required.
@Suite struct CompressedKVCacheManagerTests {

    private func makeManager(maxContext: Int = 1024,
                             ratios: [Int]? = nil)
        throws -> (MetalContext, CompressedKVCacheManager) {
        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: ratios ?? V4CacheConfig.deepSeekV4Flash.compressRatios)
        let kv = try CompressedKVCacheManager(device: ctx.device,
                                              config: config,
                                              maxContext: maxContext)
        return (ctx, kv)
    }

    // MARK: - Config / layer kinds

    @Test func flashConfig_layerKindsMatchPublishedRatios() {
        let config = V4CacheConfig.deepSeekV4Flash
        #expect(config.numLayers == 43)
        #expect(config.kind(layer: 0) == .passthrough)
        #expect(config.kind(layer: 1) == .passthrough)
        #expect(config.kind(layer: 42) == .passthrough)
        #expect(config.kind(layer: 2) == .csa)
        #expect(config.kind(layer: 3) == .hca)
        #expect(config.kind(layer: 40) == .csa)
        #expect(config.kind(layer: 41) == .hca)
    }

    // MARK: - Capacity math

    @Test func compressedCapacity_isFractionalPerToken() throws {
        let (_, kv) = try makeManager(maxContext: 1024)
        // CSA layer 2: ceil(1024/4) entries; HCA layer 3: ceil(1024/128).
        #expect(kv.compressedCapacity(layer: 2) == 256)
        #expect(kv.compressedCapacity(layer: 3) == 8)
        #expect(kv.compressedCapacity(layer: 0) == 0)
    }

    @Test func compressedCapacity_roundsUpForPartialGroups() throws {
        let (_, kv) = try makeManager(maxContext: 1030)
        #expect(kv.compressedCapacity(layer: 2) == 258)   // ceil(1030/4)
        #expect(kv.compressedCapacity(layer: 3) == 9)     // ceil(1030/128)
    }

    // MARK: - Window ring

    @Test func windowSlot_wrapsAtWindowBoundary() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        let stride = 512 * 2
        #expect(kv.windowSlot(layer: 0, position: 0).offset == 0)
        #expect(kv.windowSlot(layer: 0, position: 127).offset == 127 * stride)
        #expect(kv.windowSlot(layer: 0, position: 128).offset == 0)
        #expect(kv.windowSlot(layer: 0, position: 131).offset == 3 * stride)
    }

    @Test func windowRange_coversLastMinWindowTokens() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        #expect(kv.windowRange(tokenCount: 5) == 0..<5)
        #expect(kv.windowRange(tokenCount: 128) == 0..<128)
        #expect(kv.windowRange(tokenCount: 300) == 172..<300)
        #expect(kv.windowStartSlot(tokenCount: 128) == 0)
        #expect(kv.windowStartSlot(tokenCount: 300) == 300 % 128)
    }

    // MARK: - Group completion

    @Test func groupCompletion_firesOnGroupBoundaryOnly() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        // CSA ratio 4: completes at token positions 3, 7, 11, ...
        #expect(!kv.completesGroup(layer: 2, tokenPosition: 2))
        #expect(kv.completesGroup(layer: 2, tokenPosition: 3))
        #expect(!kv.completesGroup(layer: 2, tokenPosition: 4))
        #expect(kv.completesGroup(layer: 2, tokenPosition: 7))
        #expect(kv.groupIndex(layer: 2, tokenPosition: 3) == 0)
        #expect(kv.groupIndex(layer: 2, tokenPosition: 7) == 1)
        // HCA ratio 128: first completion at token 127.
        #expect(!kv.completesGroup(layer: 3, tokenPosition: 126))
        #expect(kv.completesGroup(layer: 3, tokenPosition: 127))
        // Passthrough never completes.
        #expect(!kv.completesGroup(layer: 0, tokenPosition: 3))
    }

    @Test func completedGroupCount_tracksAppends() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        #expect(kv.completedGroupCount(layer: 2, tokenCount: 0) == 0)
        #expect(kv.completedGroupCount(layer: 2, tokenCount: 3) == 0)
        #expect(kv.completedGroupCount(layer: 2, tokenCount: 4) == 1)
        #expect(kv.completedGroupCount(layer: 2, tokenCount: 300) == 75)
        #expect(kv.completedGroupCount(layer: 3, tokenCount: 300) == 2)
    }

    // MARK: - Coverage

    @Test func groupCoverage_identityIsOwnGroup_poolingOverlapsForCSA() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        #expect(kv.groupCoverage(layer: 2, group: 3) == 12..<16)
        // CSA pooling reads one group back (overlapped 8-token window).
        #expect(kv.poolingReadRange(layer: 2, group: 3) == 8..<16)
        #expect(kv.poolingReadRange(layer: 2, group: 0) == 0..<4)
        // HCA is non-overlapping.
        #expect(kv.groupCoverage(layer: 3, group: 1) == 128..<256)
        #expect(kv.poolingReadRange(layer: 3, group: 1) == 128..<256)
        // RoPE phase = group-start position (recon note #4).
        #expect(kv.ropePosition(layer: 2, group: 3) == 12)
        #expect(kv.ropePosition(layer: 3, group: 1) == 128)
    }

    @Test func visibleGroupCount_excludesGroupsTouchingTheWindow() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        // tokenCount 300: window covers [172, 300). CSA groups are 4 wide;
        // group g covers [4g, 4g+4); fully below 172 iff 4g+4 <= 172, so
        // g <= 42 -> 43 visible groups (of 75 completed).
        #expect(kv.visibleGroupCount(layer: 2, windowStart: 172, tokenCount: 300) == 43)
        // HCA: 128-wide groups; only group 0 ([0,128)) lies fully below 172.
        #expect(kv.visibleGroupCount(layer: 3, windowStart: 172, tokenCount: 300) == 1)
        #expect(kv.visibleGroupCount(layer: 2, windowStart: 0, tokenCount: 300) == 0)
    }

    @Test func coverageDisjointness_holdsAcrossPositions() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        var rng = SeedTree(0xA11).key("coverage")
        for _ in 0..<50 {
            let tokenCount = Int(rng.uniform(1, 512))
            let windowStart = kv.windowRange(tokenCount: tokenCount).lowerBound
            let visible = kv.visibleGroupCount(layer: 2, windowStart: windowStart,
                                               tokenCount: tokenCount)
            // The asserted invariant: last visible group's coverage ends at
            // or before the window start (no position attended twice).
            kv.assertDisjointCoverage(layer: 2, groupCount: visible,
                                      tokenCount: tokenCount)
            if visible > 0 {
                let last = kv.groupCoverage(layer: 2, group: visible - 1)
                #expect(last.upperBound <= windowStart)
            }
            // And the next group (if completed) always reaches into the window.
            let completed = kv.completedGroupCount(layer: 2, tokenCount: tokenCount)
            if visible < completed {
                let next = kv.groupCoverage(layer: 2, group: visible)
                #expect(next.upperBound > windowStart)
            }
        }
    }

    // MARK: - FP8 helpers

    @Test func e4m3_roundTripHitsExactValues() {
        // Powers of two and small integers are exact in e4m3.
        for v: Float in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
                         0.25, 16.0, 448.0, -1.0, -448.0, 0.015625] {
            let rt = V4FP8.e4m3Decode(V4FP8.e4m3Encode(v))
            #expect(rt == v, "e4m3 round-trip of \(v) gave \(rt)")
        }
    }

    @Test func e4m3_clampsTo448AndRoundsToNearestEven() {
        #expect(V4FP8.e4m3Decode(V4FP8.e4m3Encode(1000.0)) == 448.0)
        #expect(V4FP8.e4m3Decode(V4FP8.e4m3Encode(-1000.0)) == -448.0)
        // 1.05 is between 1.0 and 1.125 (ulp 0.125); rounds to 1.0.
        #expect(V4FP8.e4m3Decode(V4FP8.e4m3Encode(1.05)) == 1.0)
        // Relative error bound: 3 mantissa bits => <= 2^-4 of the binade.
        var rng = SeedTree(0xB22).key("e4m3")
        for _ in 0..<1000 {
            let v = rng.uniform(-448, 448)
            let rt = V4FP8.e4m3Decode(V4FP8.e4m3Encode(v))
            let denom = max(abs(v), 0.015625)
            #expect(abs(rt - v) / denom <= 0.0625 + 1e-6,
                    "e4m3 rel error for \(v): \(abs(rt - v) / denom)")
        }
    }

    @Test func ue8m0_roundsUpToPowerOfTwo() {
        #expect(V4FP8.ue8m0Decode(V4FP8.ue8m0Encode(1.0)) == 1.0)
        #expect(V4FP8.ue8m0Decode(V4FP8.ue8m0Encode(1.1)) == 2.0)
        #expect(V4FP8.ue8m0Decode(V4FP8.ue8m0Encode(0.003)) == 0.00390625)
        // blockScale: amax floored at 1e-4, /448, up to power of two.
        #expect(V4FP8.blockScale(amax: 448.0) == 1.0)
        #expect(V4FP8.blockScale(amax: 0.0) == V4FP8.blockScale(amax: 1e-4))
        #expect(V4FP8.blockScale(amax: 500.0) == 2.0)
    }

    // MARK: - Split entry round-trip

    @Test func quantizeEntry_roundTripsWithinFP8Bounds() throws {
        let config = V4CacheConfig(compressRatios: [4])
        var rng = SeedTree(0xC33).key("entry")
        let entry = (0..<512).map { _ in rng.uniform(-2, 2) }
        let q = CompressedKVCacheManager.quantizeEntry(entry, config: config)
        #expect(q.values.count == 448)
        #expect(q.rope.count == 64)
        let back = CompressedKVCacheManager.dequantizeEntry(
            values: q.values, scales: q.scales, rope: q.rope, config: config)
        #expect(back.count == 512)
        // Rope dims: FP16 passthrough, near-exact.
        for d in 448..<512 {
            #expect(abs(back[d] - entry[d]) <= 2e-3,
                    "rope dim \(d) error \(abs(back[d] - entry[d]))")
        }
        // Non-rope dims: bounded by the per-block ue8m0 scale (rel err
        // 2^-4 of the block's binade grid, plus the subnormal floor).
        for b in 0..<7 {
            let scale = V4FP8.ue8m0Decode(q.scales[b])
            for i in 0..<64 {
                let d = b * 64 + i
                let err = abs(back[d] - entry[d])
                #expect(err <= scale * 0.0625 * max(1.0, abs(entry[d]) / scale)
                        + scale * 0.001,
                        "dim \(d) err \(err) scale \(scale)")
            }
        }
    }

    @Test func writeThenReadEntry_throughManagerBuffers() throws {
        let ctx = try MetalContext()
        let config = V4CacheConfig(compressRatios: [4])
        let kv = try CompressedKVCacheManager(device: ctx.device,
                                              config: config,
                                              maxContext: 64)
        var rng = SeedTree(0xD44).key("rw")
        let entry = (0..<512).map { _ in rng.uniform(-1, 1) }
        kv.writeEntry(layer: 0, group: 3, entry: entry)
        let back = kv.readEntry(layer: 0, group: 3)
        let expected = CompressedKVCacheManager.dequantizeEntry(
            values: CompressedKVCacheManager.quantizeEntry(entry, config: config).values,
            scales: CompressedKVCacheManager.quantizeEntry(entry, config: config).scales,
            rope: CompressedKVCacheManager.quantizeEntry(entry, config: config).rope,
            config: config)
        #expect(back == expected)
        // Neighbouring group untouched.
        let zero = kv.readEntry(layer: 0, group: 2)
        #expect(zero.allSatisfy { $0 == 0 })
    }

    @Test func indexerStore_existsForCSAOnly() throws {
        let (_, kv) = try makeManager(maxContext: 1024)
        let slot = kv.indexerSlot(layer: 2, group: 5)
        #expect(slot.offset == 5 * 128 * 2)
        #expect(slot.buffer.length == 256 * 128 * 2)
    }

    // MARK: - Reset

    @Test func reset_dropsPositionAndAllowsReuse() throws {
        let (_, kv) = try makeManager(maxContext: 512)
        kv.advance(by: 300)
        #expect(kv.position == 300)
        kv.reset()
        #expect(kv.position == 0)
        kv.advance(by: 4)
        #expect(kv.completedGroupCount(layer: 2) == 1)
    }
}
