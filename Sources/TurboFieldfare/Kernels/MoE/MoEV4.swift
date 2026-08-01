import Foundation
import Metal

/// Per-expert sub-tensor offsets for a packed V4-Flash routed-expert blob.
/// Six 4-byte-aligned regions, no biases (unlike `MoEExpertOffsets`):
/// gate/up are `[F, D/2]` FP4 pairs with `[F, D/32]` ue8m0 scales;
/// down is `[D, F/2]` with `[D, F/32]` ue8m0 scales.
@frozen
public struct V4ExpertOffsets {
    public var gateWOff: UInt32
    public var gateSOff: UInt32
    public var upWOff: UInt32
    public var upSOff: UInt32
    public var downWOff: UInt32
    public var downSOff: UInt32

    public init(gateWOff: UInt32, gateSOff: UInt32,
                upWOff: UInt32, upSOff: UInt32,
                downWOff: UInt32, downSOff: UInt32) {
        self.gateWOff = gateWOff
        self.gateSOff = gateSOff
        self.upWOff = upWOff
        self.upSOff = upSOff
        self.downWOff = downWOff
        self.downSOff = downSOff
    }
}

/// Fused routed MoE for DeepSeek V4-Flash decode (V4F-02): FP4 e2m1 + ue8m0
/// expert dequant, clamped SwiGLU phase 1, weighted top-6 reduce phase 2, and
/// the BF16 sqrt-softplus router. Structural mirror of `MoE` with
/// `maxStreamedExperts = 6`; every kernel lives in the `moe_v4` module,
/// compiled into the shared `MetalContext` library.
final class MoEV4 {
    static let maxStreamedExperts = 6

    private let device: MTLDevice
    private let routerGemvPSO: MTLComputePipelineState
    private let routerSelectPSO: MTLComputePipelineState
    private let routerLogits: MTLBuffer
    private let phase1PSO: MTLComputePipelineState
    private let phase1SubsetPSO: MTLComputePipelineState
    private let phase2ReducePSO: MTLComputePipelineState
    private let routedArgEncoder: MTLArgumentEncoder
    private let reusableRoutedArgBuffer: MTLBuffer

    init(context: MetalContext) throws {
        self.device = context.device
        let library = context.library
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            try context.pipeline(name)
        }
        self.routerGemvPSO = try pipeline("router_v4_gemv_bf16")
        self.routerSelectPSO = try pipeline("router_v4_topk_select_k6")
        self.phase1PSO = try pipeline("moe_v4_phase1_gate_up_act_swiglu")
        self.phase1SubsetPSO = try pipeline("moe_v4_phase1_gate_up_act_swiglu_subset")
        self.phase2ReducePSO = try pipeline("moe_v4_phase2_down_reduce_k6")

        guard let logits = context.device.makeBuffer(
                length: 256 * MemoryLayout<Float>.stride,
                options: .storageModeShared),
              let phase1Function = library.makeFunction(
                name: "moe_v4_phase1_gate_up_act_swiglu") else {
            throw MetalError.noDevice
        }
        self.routerLogits = logits
        self.routedArgEncoder = phase1Function.makeArgumentEncoder(bufferIndex: 0)
        guard let reusable = context.device.makeBuffer(
                length: routedArgEncoder.encodedLength,
                options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.reusableRoutedArgBuffer = reusable
    }

    /// BF16 router GEMV + sqrt-softplus top-6 selection. Weights written to
    /// `outWeights` are F32 (normalized gathered scores × `routeScale`);
    /// `bias` is the static F32 correction bias used for selection only.
    func encodeRouterV4(commandBuffer: MTLCommandBuffer,
                        weights: MTLBuffer, weightsOffset: Int = 0,
                        bias: MTLBuffer, biasOffset: Int = 0,
                        hidden: MTLBuffer,
                        outIndices: MTLBuffer,
                        outWeights: MTLBuffer,
                        numExperts: UInt32,
                        d: UInt32,
                        routeScale: Float = 1.5) {
        precondition(d.isMultiple(of: 64))
        precondition(numExperts >= UInt32(Self.maxStreamedExperts) && numExperts <= 256)

        var expertCount = numExperts
        var dimension = d
        var scale = routeScale
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(routerGemvPSO)
            encoder.setBuffer(weights, offset: weightsOffset, index: 0)
            encoder.setBuffer(hidden, offset: 0, index: 1)
            encoder.setBuffer(routerLogits, offset: 0, index: 2)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: (Int(numExperts) + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            encoder.endEncoding()
        }

        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(routerSelectPSO)
            encoder.setBuffer(routerLogits, offset: 0, index: 0)
            encoder.setBuffer(bias, offset: biasOffset, index: 1)
            encoder.setBuffer(outIndices, offset: 0, index: 2)
            encoder.setBuffer(outWeights, offset: 0, index: 3)
            encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&scale, length: MemoryLayout<Float>.stride, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encoder.endEncoding()
        }
    }

    func makeRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                  topK: UInt32) -> MTLBuffer? {
        validate(routedBlobs: routedBlobs, topK: topK)
        guard let buffer = routedBlobs.first?.device.makeBuffer(
                length: routedArgEncoder.encodedLength,
                options: .storageModeShared) else {
            return nil
        }
        encodeRoutedArgumentBuffer(buffer, routedBlobs: routedBlobs)
        return buffer
    }

    func makeReusedRoutedArgumentBuffer(routedBlobs: [MTLBuffer],
                                        topK: UInt32) -> MTLBuffer {
        validate(routedBlobs: routedBlobs, topK: topK)
        encodeRoutedArgumentBuffer(reusableRoutedArgBuffer, routedBlobs: routedBlobs)
        return reusableRoutedArgBuffer
    }

    /// Phase 1: FP4 gate/up dequant + clamped SwiGLU over all `topK` slots.
    /// Writes `acts[slot * F + f] = silu(min(gate, 10)) * clamp(up, ±10)`.
    func encodeRoutedPhase1SwiGLU(commandBuffer: MTLCommandBuffer,
                                  routedArgBuffer: MTLBuffer,
                                  routedBlobs: [MTLBuffer],
                                  routedOffsets: V4ExpertOffsets,
                                  x: MTLBuffer,
                                  acts: MTLBuffer,
                                  d: UInt32,
                                  f: UInt32,
                                  topK: UInt32) {
        validate(routedBlobs: routedBlobs, topK: topK)
        validateGeometry(routedOffsets: routedOffsets, d: d, f: f)
        var dimension = d
        var intermediate = f
        var expertCount = topK
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase1PSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<V4ExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(topK * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Phase 1 over a subset of slots (expert-cache hit path). `activeSlots`
    /// is a device UInt32 array of slot indices; rows land in the same
    /// `acts[slot * F + f]` positions as the full kernel.
    func encodeRoutedPhase1SwiGLUSubset(commandBuffer: MTLCommandBuffer,
                                        routedArgBuffer: MTLBuffer,
                                        routedBlobs: [MTLBuffer],
                                        routedOffsets: V4ExpertOffsets,
                                        x: MTLBuffer,
                                        acts: MTLBuffer,
                                        activeSlots: MTLBuffer,
                                        activeSlotIndices: [UInt32],
                                        activeCount: UInt32,
                                        d: UInt32,
                                        f: UInt32,
                                        topK: UInt32) {
        guard activeCount > 0 else { return }
        validate(routedBlobs: routedBlobs, topK: topK)
        validateGeometry(routedOffsets: routedOffsets, d: d, f: f)
        precondition(activeSlotIndices.count == Int(activeCount))
        var dimension = d
        var intermediate = f
        var expertCount = topK
        var active = activeCount
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase1SubsetPSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for slot in activeSlotIndices {
            encoder.useResource(routedBlobs[Int(slot)], usage: .read)
        }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<V4ExpertOffsets>.stride, index: 1)
        encoder.setBuffer(x, offset: 0, index: 2)
        encoder.setBuffer(acts, offset: 0, index: 3)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBuffer(activeSlots, offset: 0, index: 7)
        encoder.setBytes(&active, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: (Int(activeCount * f) + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Phase 2: FP4 down projection + routing-weighted top-6 reduce + residual.
    /// One threadgroup per output element, six SIMD groups (one per slot).
    /// `routingWeights` is F32 (the router's normalized scaled weights).
    func encodeRoutedPhase2Reduce(commandBuffer: MTLCommandBuffer,
                                  routedArgBuffer: MTLBuffer,
                                  routedBlobs: [MTLBuffer],
                                  routedOffsets: V4ExpertOffsets,
                                  acts: MTLBuffer,
                                  routingWeights: MTLBuffer,
                                  residual: MTLBuffer,
                                  y: MTLBuffer,
                                  d: UInt32,
                                  f: UInt32,
                                  topK: UInt32) {
        validate(routedBlobs: routedBlobs, topK: topK)
        validateGeometry(routedOffsets: routedOffsets, d: d, f: f)
        var dimension = d
        var intermediate = f
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(phase2ReducePSO)
        encoder.setBuffer(routedArgBuffer, offset: 0, index: 0)
        for buffer in routedBlobs { encoder.useResource(buffer, usage: .read) }
        var offsets = routedOffsets
        encoder.setBytes(&offsets, length: MemoryLayout<V4ExpertOffsets>.stride, index: 1)
        encoder.setBuffer(acts, offset: 0, index: 2)
        encoder.setBuffer(routingWeights, offset: 0, index: 3)
        encoder.setBuffer(residual, offset: 0, index: 4)
        encoder.setBuffer(y, offset: 0, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&intermediate, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(d), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * Self.maxStreamedExperts, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func validate(routedBlobs: [MTLBuffer], topK: UInt32) {
        precondition(topK == UInt32(Self.maxStreamedExperts))
        precondition(routedBlobs.count == Int(topK))
    }

    private func validateGeometry(routedOffsets: V4ExpertOffsets, d: UInt32, f: UInt32) {
        // FP4 group walk and half4 activation loads.
        precondition(d.isMultiple(of: 32), "D must be a multiple of 32")
        precondition(f.isMultiple(of: 32), "F must be a multiple of 32")
        // Blob sub-tensors are padded to 4 bytes so the kernels can use
        // aligned uint weight loads.
        for offset in [routedOffsets.gateWOff, routedOffsets.gateSOff,
                       routedOffsets.upWOff, routedOffsets.upSOff,
                       routedOffsets.downWOff, routedOffsets.downSOff] {
            precondition(offset.isMultiple(of: 4),
                         "V4 expert blob sub-tensor offset \(offset) is not 4-aligned")
        }
    }

    private func encodeRoutedArgumentBuffer(_ buffer: MTLBuffer,
                                            routedBlobs: [MTLBuffer]) {
        routedArgEncoder.setArgumentBuffer(buffer, offset: 0)
        for (index, blob) in routedBlobs.enumerated() {
            routedArgEncoder.setBuffer(blob, offset: 0, index: index)
        }
    }
}
