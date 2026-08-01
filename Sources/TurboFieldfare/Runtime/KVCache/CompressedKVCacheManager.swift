import Foundation
import Darwin
import Metal

/// Fractional-per-token compressed KV store for DeepSeek V4-Flash
/// (V4F-03, milestone 1). Replaces the `KVCacheManager` role for V4 layers;
/// the production Gemma manager is untouched.
///
/// Storage per layer:
///
/// - **Window ring** (all layers): the last `window` (128) uncompressed
///   tokens, FP16, ring-indexed exactly like `KVCacheManager`'s SWA ring.
///   The reference QAT-sims these to FP8 but stores BF16; we keep FP16
///   storage and leave the sim to the write-side epilogue (design decision
///   recorded in the V4F-03 report).
/// - **Compressed entries** (CSA/HCA layers only): one entry per
///   `compressRatio` tokens. Each entry is the split FP8/FP16 layout from
///   `V4KVLayout`: e4m3 values + ue8m0 scales for the non-rope 448 dims,
///   FP16 for the trailing 64 rope dims.
///
/// Group bookkeeping (compress-on-group-completion) lives here; the actual
/// pooling math runs in the `v4_csa_compress_group` Metal kernel. The
/// manager decides *when* a group completes and *where* the entry lands:
///
/// - CSA (ratio 4, overlap): entry `g` is emitted when token position
///   `4g+3` is appended. Its pooling window also reads group `g-1`'s
///   projections (the write-side accumulator must retain them), and it
///   carries the RoPE phase of absolute position `4g` (group start).
/// - HCA (ratio 128): entry `g` emitted when token `128g+127` is appended;
///   non-overlapped; RoPE phase of position `128g`.
///
/// Coverage invariant (the double-count guard from the work-order risk
/// table): the attention kernel must only select compressed entries whose
/// whole group lies *below* the window start. `visibleGroupCount` exposes
/// exactly that bound; `assertDisjointCoverage` checks it in debug builds.
public final class CompressedKVCacheManager {
    public let config: V4CacheConfig
    public let maxContext: Int

    /// A write/read address into one of the split buffers.
    public struct Slot: @unchecked Sendable {
        public let buffer: MTLBuffer
        public let offset: Int
    }

    /// Address of one compressed entry across the three split buffers.
    public struct CompressedSlot: @unchecked Sendable {
        public let values: Slot
        public let scales: Slot
        public let rope: Slot
    }

    /// Indexer head dim for CSA layers (index_head_dim 128 per config).
    public static let indexerHeadDim = 128

    private let windowBuffers: [MTLBuffer]        // per layer: window x headDim FP16
    private let valueBuffers: [MTLBuffer?]        // per layer: cap x nonRopeDim e4m3
    private let scaleBuffers: [MTLBuffer?]        // per layer: cap x scaleStride ue8m0
    private let ropeBuffers: [MTLBuffer?]         // per layer: cap x ropeDim FP16
    private let indexerBuffers: [MTLBuffer?]      // CSA layers: cap x 128 FP16
    private let compressedCapacities: [Int]       // per layer: entries (0 for passthrough)

    public private(set) var position: Int = 0

    public init(device: MTLDevice,
                config: V4CacheConfig,
                maxContext: Int) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        self.config = config
        self.maxContext = maxContext

        let windowBytes = config.window * config.headDim * V4KVLayout.fp16Size
        let valueStride = V4KVLayout.valueStride(config: config)
        let scaleStride = V4KVLayout.scaleStride(config: config)
        let ropeStride = V4KVLayout.ropeStride(config: config)

        var windows: [MTLBuffer] = []
        var values: [MTLBuffer?] = []
        var scales: [MTLBuffer?] = []
        var ropes: [MTLBuffer?] = []
        var indexers: [MTLBuffer?] = []
        var caps: [Int] = []
        windows.reserveCapacity(config.numLayers)

        for layer in 0..<config.numLayers {
            guard let win = device.makeBuffer(length: windowBytes,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            win.label = "v4kv.window.layer\(layer)"
            windows.append(win)

            let ratio = config.compressRatio(layer: layer)
            if ratio > 0 {
                let cap = (maxContext + ratio - 1) / ratio
                caps.append(cap)
                guard let v = device.makeBuffer(length: cap * valueStride,
                                                options: .storageModeShared),
                      let s = device.makeBuffer(length: cap * scaleStride,
                                                options: .storageModeShared),
                      let r = device.makeBuffer(length: cap * ropeStride,
                                                options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                v.label = "v4kv.values.layer\(layer)"
                s.label = "v4kv.scales.layer\(layer)"
                r.label = "v4kv.rope.layer\(layer)"
                values.append(v); scales.append(s); ropes.append(r)

                if config.kind(layer: layer) == .csa {
                    guard let ib = device.makeBuffer(
                        length: cap * Self.indexerHeadDim * V4KVLayout.fp16Size,
                        options: .storageModeShared) else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    ib.label = "v4kv.indexer.layer\(layer)"
                    indexers.append(ib)
                } else {
                    indexers.append(nil)
                }
            } else {
                caps.append(0)
                values.append(nil); scales.append(nil); ropes.append(nil)
                indexers.append(nil)
            }
        }

        self.windowBuffers = windows
        self.valueBuffers = values
        self.scaleBuffers = scales
        self.ropeBuffers = ropes
        self.indexerBuffers = indexers
        self.compressedCapacities = caps
    }

    // MARK: - Capacity math

    public func layerKind(_ layer: Int) -> V4LayerKind { config.kind(layer: layer) }

    /// Allocated compressed-entry capacity (0 for passthrough layers).
    public func compressedCapacity(layer: Int) -> Int { compressedCapacities[layer] }

    /// Total resident bytes across all layers (for budget reporting).
    public var totalBytes: Int {
        var total = 0
        for b in windowBuffers { total += b.length }
        for layer in 0..<config.numLayers {
            total += valueBuffers[layer]?.length ?? 0
            total += scaleBuffers[layer]?.length ?? 0
            total += ropeBuffers[layer]?.length ?? 0
            total += indexerBuffers[layer]?.length ?? 0
        }
        return total
    }

    // MARK: - Position cursor

    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext,
                     "advance would exceed maxContext")
        position += count
    }

    /// Drop all cached positions and return physical pages to the OS,
    /// mirroring `KVCacheManager.reset()` (no zeroing; kernels read only
    /// valid ranges).
    public func reset() {
        position = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            advise(windowBuffers[layer], pageSize: pageSize, seen: &advised)
            if let v = valueBuffers[layer] { advise(v, pageSize: pageSize, seen: &advised) }
            if let s = scaleBuffers[layer] { advise(s, pageSize: pageSize, seen: &advised) }
            if let r = ropeBuffers[layer] { advise(r, pageSize: pageSize, seen: &advised) }
            if let i = indexerBuffers[layer] { advise(i, pageSize: pageSize, seen: &advised) }
        }
    }

    // MARK: - Window ring

    /// Physical ring slot for a logical token position. The window ring is
    /// exactly `window` slots; every layer rings (decode-first: the window
    /// branch never needs more than the last 128 tokens).
    public func windowPhysicalSlot(position: Int) -> Int {
        position % config.window
    }

    /// Write target for the window-branch KV of token `position` (FP16,
    /// headDim contiguous).
    public func windowSlot(layer: Int, position: Int) -> Slot {
        validateRange(start: position, count: 1)
        return Slot(buffer: windowBuffers[layer],
                    offset: windowPhysicalSlot(position: position)
                        * config.headDim * V4KVLayout.fp16Size)
    }

    public func windowBuffer(layer: Int) -> MTLBuffer { windowBuffers[layer] }

    /// Logical positions the window branch covers after `tokenCount` tokens
    /// have been written: the last `min(window, tokenCount)` positions.
    public func windowRange(tokenCount: Int) -> Range<Int> {
        let lo = max(0, tokenCount - config.window)
        return lo..<tokenCount
    }

    /// Ring slot of the oldest window entry (0 while the ring has not
    /// wrapped), matching `KVCacheManager.ringStartSlot`.
    public func windowStartSlot(tokenCount: Int) -> Int {
        guard tokenCount > config.window else { return 0 }
        return tokenCount % config.window
    }

    // MARK: - Compressed group bookkeeping

    /// Number of completed compressed groups after `tokenCount` tokens.
    /// Entry `g` completes once token `(g+1)*ratio - 1` is appended.
    public func completedGroupCount(layer: Int, tokenCount: Int) -> Int {
        let ratio = config.compressRatio(layer: layer)
        guard ratio > 0 else { return 0 }
        return min(tokenCount / ratio, compressedCapacities[layer])
    }

    public func completedGroupCount(layer: Int) -> Int {
        completedGroupCount(layer: layer, tokenCount: position)
    }

    /// True when appending the token at `tokenPosition` (0-based) completes
    /// group `tokenPosition / ratio`, i.e. a compressed entry must be
    /// flushed after this token's projections are available.
    public func completesGroup(layer: Int, tokenPosition: Int) -> Bool {
        let ratio = config.compressRatio(layer: layer)
        guard ratio > 0 else { return false }
        return (tokenPosition + 1) % ratio == 0
    }

    /// Group index flushed when `completesGroup` returns true for
    /// `tokenPosition`.
    public func groupIndex(layer: Int, tokenPosition: Int) -> Int {
        tokenPosition / config.compressRatio(layer: layer)
    }

    /// Logical token positions that group `g` *represents* for selection
    /// and coverage purposes: `[g*ratio, (g+1)*ratio)`. (CSA pooling also
    /// *reads* group g-1's projections — see `poolingReadRange` — but the
    /// entry's coverage identity is its own group.)
    public func groupCoverage(layer: Int, group: Int) -> Range<Int> {
        let ratio = config.compressRatio(layer: layer)
        precondition(ratio > 0, "passthrough layers have no compressed groups")
        return (group * ratio)..<(group * ratio + ratio)
    }

    /// Token positions whose projections the pooling of group `g` reads.
    /// CSA overlaps one group back; HCA is non-overlapping.
    public func poolingReadRange(layer: Int, group: Int) -> Range<Int> {
        let coverage = groupCoverage(layer: layer, group: group)
        if config.kind(layer: layer) == .csa, group > 0 {
            return (coverage.lowerBound - config.compressRatio(layer: layer))..<coverage.upperBound
        }
        return coverage
    }

    /// Absolute position whose RoPE phase the compressed entry carries:
    /// the group's first token (recon note #4: stride ratio, offset 0).
    public func ropePosition(layer: Int, group: Int) -> Int {
        groupCoverage(layer: layer, group: group).lowerBound
    }

    /// Write/read address of compressed entry `group`.
    public func compressedSlot(layer: Int, group: Int) -> CompressedSlot {
        let ratio = config.compressRatio(layer: layer)
        precondition(ratio > 0, "passthrough layers have no compressed store")
        precondition(group >= 0 && group < compressedCapacities[layer],
                     "group \(group) out of capacity \(compressedCapacities[layer])")
        return CompressedSlot(
            values: Slot(buffer: valueBuffers[layer]!,
                         offset: group * V4KVLayout.valueStride(config: config)),
            scales: Slot(buffer: scaleBuffers[layer]!,
                         offset: group * V4KVLayout.scaleStride(config: config)),
            rope: Slot(buffer: ropeBuffers[layer]!,
                       offset: group * V4KVLayout.ropeStride(config: config)))
    }

    // MARK: - Coverage disjointness (double-count guard)

    /// Number of leading compressed groups whose entire coverage lies below
    /// `windowStart`. The sparse/dense attention over compressed entries
    /// must select only from `[0, visibleGroupCount)` so no logical
    /// position is attended twice (once compressed, once through the
    /// window branch).
    public func visibleGroupCount(layer: Int, windowStart: Int) -> Int {
        visibleGroupCount(layer: layer, windowStart: windowStart,
                          tokenCount: position)
    }

    /// Explicit-`tokenCount` form for call sites that track their own
    /// decode position.
    public func visibleGroupCount(layer: Int, windowStart: Int,
                                  tokenCount: Int) -> Int {
        let ratio = config.compressRatio(layer: layer)
        guard ratio > 0, windowStart > 0 else { return 0 }
        return min(windowStart / ratio,
                   completedGroupCount(layer: layer, tokenCount: tokenCount))
    }

    /// Debug-build assertion that the first `groupCount` compressed groups
    /// and the window range cover disjoint logical positions.
    public func assertDisjointCoverage(layer: Int,
                                       groupCount: Int,
                                       tokenCount: Int) {
        let window = windowRange(tokenCount: tokenCount)
        guard groupCount > 0, !window.isEmpty else { return }
        let lastCovered = groupCoverage(layer: layer, group: groupCount - 1).upperBound
        assert(lastCovered <= window.lowerBound,
               "compressed group \(groupCount - 1) covers up to \(lastCovered) "
               + "but window starts at \(window.lowerBound): double-count")
    }

    // MARK: - Indexer store (CSA layers)

    /// Whole indexer compressed cache for a CSA layer ([capacity, 128] FP16;
    /// FP4 QAT sim + Hadamard happen write-side, storage stays FP16).
    public func indexerBuffer(layer: Int) -> MTLBuffer {
        precondition(config.kind(layer: layer) == .csa,
                     "only CSA layers have an indexer store")
        return indexerBuffers[layer]!
    }

    /// Write/read address of indexer entry `group` (128 FP16 dims).
    public func indexerSlot(layer: Int, group: Int) -> Slot {
        precondition(group >= 0 && group < compressedCapacities[layer],
                     "indexer group \(group) out of capacity")
        return Slot(buffer: indexerBuffer(layer: layer),
                    offset: group * Self.indexerHeadDim * V4KVLayout.fp16Size)
    }

    // MARK: - CPU-side quantize helpers (synthetic writes, tests)

    /// Quantize one 512-dim FP32 entry into the split store layout: FP8
    /// e4m3 + ue8m0 scales for the non-rope dims (reference `act_quant`
    /// semantics), FP16 passthrough for the rope dims.
    public static func quantizeEntry(_ entry: [Float],
                                     config: V4CacheConfig)
        -> (values: [UInt8], scales: [UInt8], rope: [Float16]) {
        precondition(entry.count == config.headDim)
        let nonRope = config.nonRopeDim
        let blocks = nonRope / V4KVLayout.fp8Block
        var values = [UInt8](repeating: 0, count: nonRope)
        var scales = [UInt8](repeating: 0,
                             count: V4KVLayout.scaleStride(config: config))
        for b in 0..<blocks {
            let base = b * V4KVLayout.fp8Block
            var amax: Float = 0
            for i in 0..<V4KVLayout.fp8Block {
                amax = max(amax, abs(entry[base + i]))
            }
            let scale = V4FP8.blockScale(amax: amax)
            scales[b] = V4FP8.ue8m0Encode(scale)
            let inv = 1.0 / scale
            for i in 0..<V4KVLayout.fp8Block {
                values[base + i] = V4FP8.e4m3Encode(entry[base + i] * inv)
            }
        }
        let rope = (0..<config.ropeDim).map { Float16(entry[nonRope + $0]) }
        return (values, scales, rope)
    }

    /// Inverse of `quantizeEntry` (FP32 approximation of the stored entry).
    public static func dequantizeEntry(values: [UInt8],
                                       scales: [UInt8],
                                       rope: [Float16],
                                       config: V4CacheConfig) -> [Float] {
        var out = [Float](repeating: 0, count: config.headDim)
        let nonRope = config.nonRopeDim
        let blocks = nonRope / V4KVLayout.fp8Block
        for b in 0..<blocks {
            let scale = V4FP8.ue8m0Decode(scales[b])
            let base = b * V4KVLayout.fp8Block
            for i in 0..<V4KVLayout.fp8Block {
                out[base + i] = V4FP8.e4m3Decode(values[base + i]) * scale
            }
        }
        for i in 0..<config.ropeDim {
            out[nonRope + i] = Float(rope[i])
        }
        return out
    }

    /// Write a quantized entry into the layer's split buffers at `group`.
    public func writeEntry(layer: Int, group: Int, entry: [Float]) {
        let slot = compressedSlot(layer: layer, group: group)
        let q = Self.quantizeEntry(entry, config: config)
        q.values.withUnsafeBytes { src in
            slot.values.buffer.contents().advanced(by: slot.values.offset)
                .copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        q.scales.withUnsafeBytes { src in
            slot.scales.buffer.contents().advanced(by: slot.scales.offset)
                .copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
        q.rope.withUnsafeBytes { src in
            slot.rope.buffer.contents().advanced(by: slot.rope.offset)
                .copyMemory(from: src.baseAddress!, byteCount: src.count)
        }
    }

    /// Read back and dequantize the entry at `group`.
    public func readEntry(layer: Int, group: Int) -> [Float] {
        let slot = compressedSlot(layer: layer, group: group)
        let nonRope = config.nonRopeDim
        let scaleStride = V4KVLayout.scaleStride(config: config)
        var values = [UInt8](repeating: 0, count: nonRope)
        var scales = [UInt8](repeating: 0, count: scaleStride)
        var rope = [Float16](repeating: 0, count: config.ropeDim)
        values.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(
                from: slot.values.buffer.contents().advanced(by: slot.values.offset),
                byteCount: nonRope)
        }
        scales.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(
                from: slot.scales.buffer.contents().advanced(by: slot.scales.offset),
                byteCount: scaleStride)
        }
        rope.withUnsafeMutableBytes { dst in
            dst.baseAddress!.copyMemory(
                from: slot.rope.buffer.contents().advanced(by: slot.rope.offset),
                byteCount: config.ropeDim * V4KVLayout.fp16Size)
        }
        return Self.dequantizeEntry(values: values, scales: scales,
                                    rope: rope, config: config)
    }

    // MARK: - Internals

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int,
                        seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}
