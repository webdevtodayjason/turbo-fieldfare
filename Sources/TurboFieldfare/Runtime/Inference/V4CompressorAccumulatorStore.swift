import Foundation
import Metal

/// Per-layer compressor accumulator buffers used by `V4ForwardRunner` while
/// staging compressed-attention groups during decode.
///
/// CSA and HCA layers cannot share these buffers because command buffers for
/// neighboring layers may overlap and because each compressor layer maintains
/// previous/current group state independently.
public final class V4CompressorAccumulatorStore {
    public struct CSAState {
        public let prevKV: MTLBuffer
        public let curKV: MTLBuffer
        public let prevGate: MTLBuffer
        public let curGate: MTLBuffer
        public let idxPrevKV: MTLBuffer
        public let idxCurKV: MTLBuffer
        public let idxPrevGate: MTLBuffer
        public let idxCurGate: MTLBuffer
    }

    public struct HCAState {
        public let ringKV: MTLBuffer
        public let ringGate: MTLBuffer
    }

    public static let csaRows = 4
    public static let csaWidth = 1024
    public static let csaIndexerWidth = 256
    public static let hcaRows = 128
    public static let hcaWidth = 512

    private let csaByLayer: [Int: CSAState]
    private let hcaByLayer: [Int: HCAState]

    public init(device: MTLDevice, layerKinds: [V4LayerKind]) throws {
        func scratch(_ floats: Int, _ label: String,
                     initialValue: Float = 0) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: floats * MemoryLayout<Float>.stride,
                                                 options: .storageModeShared) else {
                throw MetalError.missingFunction("v4 compressor accumulator: \(label)")
            }
            buffer.label = label
            if initialValue == 0 {
                memset(buffer.contents(), 0, buffer.length)
            } else {
                let values = buffer.contents().assumingMemoryBound(to: Float.self)
                values.initialize(repeating: initialValue, count: floats)
            }
            return buffer
        }

        var csa: [Int: CSAState] = [:]
        var hca: [Int: HCAState] = [:]
        for (layer, kind) in layerKinds.enumerated() {
            switch kind {
            case .csa:
                csa[layer] = CSAState(
                    prevKV: try scratch(Self.csaRows * Self.csaWidth, "layer\(layer).csaPrevKV"),
                    curKV: try scratch(Self.csaRows * Self.csaWidth, "layer\(layer).csaCurKV"),
                    prevGate: try scratch(Self.csaRows * Self.csaWidth,
                                          "layer\(layer).csaPrevGate",
                                          initialValue: -.infinity),
                    curGate: try scratch(Self.csaRows * Self.csaWidth, "layer\(layer).csaCurGate"),
                    idxPrevKV: try scratch(Self.csaRows * Self.csaIndexerWidth, "layer\(layer).idxPrevKV"),
                    idxCurKV: try scratch(Self.csaRows * Self.csaIndexerWidth, "layer\(layer).idxCurKV"),
                    idxPrevGate: try scratch(Self.csaRows * Self.csaIndexerWidth,
                                             "layer\(layer).idxPrevGate",
                                             initialValue: -.infinity),
                    idxCurGate: try scratch(Self.csaRows * Self.csaIndexerWidth, "layer\(layer).idxCurGate"))
            case .hca:
                hca[layer] = HCAState(
                    ringKV: try scratch(Self.hcaRows * Self.hcaWidth, "layer\(layer).hcaRingKV"),
                    ringGate: try scratch(Self.hcaRows * Self.hcaWidth, "layer\(layer).hcaRingGate"))
            case .passthrough:
                continue
            }
        }
        self.csaByLayer = csa
        self.hcaByLayer = hca
    }

    public func csa(layer: Int) -> CSAState {
        guard let state = csaByLayer[layer] else {
            preconditionFailure("Layer \(layer) is not a CSA compressor layer")
        }
        return state
    }

    public func hca(layer: Int) -> HCAState {
        guard let state = hcaByLayer[layer] else {
            preconditionFailure("Layer \(layer) is not an HCA compressor layer")
        }
        return state
    }

    public func rollCSA(commandBuffer: MTLCommandBuffer, layer: Int) {
        let state = csa(layer: layer)
        copy(commandBuffer: commandBuffer, from: state.curKV, to: state.prevKV,
             bytes: Self.csaRows * Self.csaWidth * MemoryLayout<Float>.stride)
        copy(commandBuffer: commandBuffer, from: state.curGate, to: state.prevGate,
             bytes: Self.csaRows * Self.csaWidth * MemoryLayout<Float>.stride)
        copy(commandBuffer: commandBuffer, from: state.idxCurKV, to: state.idxPrevKV,
             bytes: Self.csaRows * Self.csaIndexerWidth * MemoryLayout<Float>.stride)
        copy(commandBuffer: commandBuffer, from: state.idxCurGate, to: state.idxPrevGate,
             bytes: Self.csaRows * Self.csaIndexerWidth * MemoryLayout<Float>.stride)
    }

    private func copy(commandBuffer: MTLCommandBuffer, from src: MTLBuffer,
                      to dst: MTLBuffer, bytes: Int) {
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: src, sourceOffset: 0, to: dst, destinationOffset: 0, size: bytes)
        blit.endEncoding()
    }
}
