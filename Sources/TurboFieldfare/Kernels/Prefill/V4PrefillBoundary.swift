import Foundation
import Metal

/// Batched mHC / RMSNorm / RoPE layer-boundary kernels for DeepSeek V4-Flash
/// chunked prefill (V4F-06c, work-order A1). Wraps `v4pf_hc_params`,
/// `v4pf_hc_pre`, `v4pf_hc_post`, `v4pf_rmsnorm_f32f16`, and
/// `v4pf_rope_trailing` in `prefill_boundary_v4.metal`.
///
/// These are the row-parallel variants of the decode boundary kernels in
/// `V4HyperConnections` / `V4RoPE`: one dispatch moves up to `maxRows`
/// prompt tokens through a boundary. The residual stream is [rows, 4 x dim]
/// fp32; per-row dynamic parameters land in `paramsBuffer` as [rows, 24]
/// fp32 (`pre[4] | post[4] | comb[16]` per row, exact decode ordering).
///
/// RoPE uses the INTERLEAVED adjacent-pair convention (elements 2i, 2i+1 of
/// the trailing rope slice) with a per-row positions buffer; the inverse
/// mode applies the conjugate at the row's POSITIVE absolute position
/// (output de-rotation; see the design note pitfalls).
final class V4PrefillBoundary: @unchecked Sendable {
    static let hcMult = 4
    /// pre[4] | post[4] | comb[16] fp32 per row.
    static let paramsFloatsPerRow = 24

    /// Maximum rows (prompt tokens per chunk) the scratch params buffer holds.
    let maxRows: Int

    private let device: MTLDevice
    private let paramsPSO: MTLComputePipelineState
    private let prePSO: MTLComputePipelineState
    private let postPSO: MTLComputePipelineState
    private let rmsnormPSO: MTLComputePipelineState
    private let rmsnormSerialOrderPSO: MTLComputePipelineState
    private let rmsnormF16PSO: MTLComputePipelineState
    private let ropePSO: MTLComputePipelineState

    /// Scratch holding the latest `encodeParams` output, [maxRows, 24] fp32
    /// (hazard-tracked in the same command buffer; read back in tests only).
    let paramsBuffer: MTLBuffer

    // Device-keyed standalone module library. `prefill_boundary_v4` is not
    // yet registered in `MetalContext.shaderModules` (integration registers
    // it in one pass), so compile it separately, mirroring the pre-context
    // V4 wrapper pattern. Once registered this can switch to
    // `V4ShaderLibrary` / `context.pipeline(...)` with no call-site change.
    private static let libraryLock = NSLock()
    private nonisolated(unsafe) static var libraries: [ObjectIdentifier: MTLLibrary] = [:]

    private static func moduleLibrary(for device: MTLDevice) throws -> MTLLibrary {
        libraryLock.lock()
        if let cached = libraries[ObjectIdentifier(device)] {
            libraryLock.unlock()
            return cached
        }
        libraryLock.unlock()
        guard let url = Bundle.module.url(forResource: "prefill_boundary_v4",
                                          withExtension: "metal",
                                          subdirectory: "Metal/Prefill") else {
            throw MetalError.missingShaderResource("prefill_boundary_v4")
        }
        let src = try String(contentsOf: url, encoding: .utf8)
        let opts = MTLCompileOptions()
        opts.languageVersion = .version4_0
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: src, options: opts)
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
        libraryLock.lock()
        libraries[ObjectIdentifier(device)] = library
        libraryLock.unlock()
        return library
    }

    private static func pipeline(device: MTLDevice,
                                 library: MTLLibrary,
                                 name: String) throws -> MTLComputePipelineState {
        guard let fn = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        return try device.makeComputePipelineState(function: fn)
    }

    init(device: MTLDevice, maxRows: Int = 128) throws {
        precondition(maxRows > 0)
        self.device = device
        self.maxRows = maxRows
        let library = try Self.moduleLibrary(for: device)
        self.paramsPSO = try Self.pipeline(device: device, library: library,
                                           name: "v4pf_hc_params")
        self.prePSO = try Self.pipeline(device: device, library: library,
                                        name: "v4pf_hc_pre")
        self.postPSO = try Self.pipeline(device: device, library: library,
                                         name: "v4pf_hc_post")
        self.rmsnormPSO = try Self.pipeline(device: device, library: library,
                                            name: "v4pf_rmsnorm_f32f16")
        self.rmsnormSerialOrderPSO = try Self.pipeline(
            device: device, library: library,
            name: "v4pf_rmsnorm_f32f16_serial_order")
        self.rmsnormF16PSO = try Self.pipeline(device: device, library: library,
                                               name: "v4pf_rmsnorm_f16f16")
        self.ropePSO = try Self.pipeline(device: device, library: library,
                                         name: "v4pf_rope_trailing")
        guard let params = device.makeBuffer(
            length: maxRows * Self.paramsFloatsPerRow * MemoryLayout<Float>.size,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        self.paramsBuffer = params
    }

    /// Batched dynamic parameters for one sublayer boundary, all rows at once.
    ///
    /// - `x`: flattened stream, [rows, 4 * dim] fp32.
    /// - `hcFn`: [24, 4 * dim] fp32 (`hc_fn`).
    /// - `hcBase`: [24] fp32 (`hc_base`).
    /// - `hcScale`: [3] fp32 (`hc_scale`).
    /// Writes [rows, 24] fp32 into `paramsBuffer`.
    func encodeParams(commandBuffer cb: MTLCommandBuffer,
                      x: MTLBuffer, xOffset: Int = 0,
                      hcFn: MTLBuffer, hcFnOffset: Int = 0,
                      hcBase: MTLBuffer, hcBaseOffset: Int = 0,
                      hcScale: MTLBuffer, hcScaleOffset: Int = 0,
                      rows: Int,
                      dim: Int,
                      normEps: Float = 1e-6,
                      hcEps: Float = 1e-6) {
        precondition(rows > 0 && rows <= maxRows)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(paramsPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(hcFn, offset: hcFnOffset, index: 1)
        enc.setBuffer(hcBase, offset: hcBaseOffset, index: 2)
        enc.setBuffer(hcScale, offset: hcScaleOffset, index: 3)
        enc.setBuffer(paramsBuffer, offset: 0, index: 4)
        var d = UInt32(dim)
        var ne = normEps
        var he = hcEps
        enc.setBytes(&d, length: 4, index: 5)
        enc.setBytes(&ne, length: 4, index: 6)
        enc.setBytes(&he, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 768, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Batched branch input gather:
    /// `y[r*dim+d] = sum_j pre[r,j] * x[r*4*dim + j*dim + d]` (fp32).
    /// Consumes the params produced by the latest `encodeParams`.
    func encodePre(commandBuffer cb: MTLCommandBuffer,
                   x: MTLBuffer, xOffset: Int = 0,
                   out: MTLBuffer, outOffset: Int = 0,
                   rows: Int,
                   dim: Int) {
        precondition(rows > 0 && rows <= maxRows)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(prePSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(paramsBuffer, offset: 0, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var d = UInt32(dim)
        var r = UInt32(rows)
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&r, length: 4, index: 4)
        let tg = min(256, Int(prePSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: rows * dim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Batched boundary merge:
    /// `out[r, k*dim+d] = post[r,k] * sublayer[r,d] +
    ///  sum_j comb[r,j,k] * residual[r, j*dim+d]` (comb gather by column).
    /// `out` may alias `residual` (per-thread disjoint slots make the
    /// in-place stream update safe).
    func encodePost(commandBuffer cb: MTLCommandBuffer,
                    residual: MTLBuffer, residualOffset: Int = 0,
                    sublayer: MTLBuffer, sublayerOffset: Int = 0,
                    out: MTLBuffer, outOffset: Int = 0,
                    rows: Int,
                    dim: Int) {
        precondition(rows > 0 && rows <= maxRows)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(postPSO)
        enc.setBuffer(residual, offset: residualOffset, index: 0)
        enc.setBuffer(sublayer, offset: sublayerOffset, index: 1)
        enc.setBuffer(paramsBuffer, offset: 0, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var d = UInt32(dim)
        var r = UInt32(rows)
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&r, length: 4, index: 5)
        let tg = min(256, Int(postPSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: rows * dim, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Batched RMSNorm, fp32 in / fp16 out, fp32 gamma and math:
    /// `out[r, i] = x[r, i] * rsqrt(mean(x[r]^2) + eps) * gamma[i]`.
    /// One threadgroup per row.
    func encodeRMSNorm(commandBuffer cb: MTLCommandBuffer,
                       x: MTLBuffer, xOffset: Int = 0,
                       gamma: MTLBuffer, gammaOffset: Int = 0,
                       out: MTLBuffer, outOffset: Int = 0,
                       rows: Int,
                       n: Int,
                       eps: Float = 1e-6,
                       useGamma: Bool = true) {
        precondition(rows > 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(rmsnormPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(gamma, offset: gammaOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var nn = UInt32(n)
        var e = eps
        var ug = UInt32(useGamma ? 1 : 0)
        enc.setBytes(&nn, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.setBytes(&ug, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Batched FP32-input RMSNorm with the decode kernel's exact reduction and
    /// multiplication order. Use this for runtime prefill parity. The generic
    /// method above remains useful for optional gamma-free validation paths.
    func encodeRMSNormSerialOrder(commandBuffer cb: MTLCommandBuffer,
                                  x: MTLBuffer, xOffset: Int = 0,
                                  gamma: MTLBuffer, gammaOffset: Int = 0,
                                  out: MTLBuffer, outOffset: Int = 0,
                                  rows: Int,
                                  n: Int,
                                  eps: Float = 1e-6) {
        precondition(rows > 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(rmsnormSerialOrderPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(gamma, offset: gammaOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var nn = UInt32(n)
        var e = eps
        enc.setBytes(&nn, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Batched decode-parity RMSNorm, fp16 input/output with one 256-thread
    /// threadgroup per row. `out` may alias `x` when both use the same row
    /// stride, matching `v4b_rmsnorm`.
    func encodeRMSNormF16(commandBuffer cb: MTLCommandBuffer,
                          x: MTLBuffer, xOffset: Int = 0,
                          gamma: MTLBuffer, gammaOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          rows: Int,
                          n: Int,
                          eps: Float = 1e-6,
                          useGamma: Bool = true) {
        precondition(rows > 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(rmsnormF16PSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(gamma, offset: gammaOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var nn = UInt32(n)
        var e = eps
        var ug = UInt32(useGamma ? 1 : 0)
        enc.setBytes(&nn, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.setBytes(&ug, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Batched trailing-slice partial RoPE with per-row positions, in place.
    /// Interleaved adjacent-pair convention; `inverse` applies the conjugate
    /// (output de-rotation) at the row's POSITIVE absolute position.
    ///
    /// - `x`: [rows, width] FP16.
    /// - `positions`: [rows] fp32 absolute positions (signed/fractional
    ///   values are legal, matching the decode kernel).
    func encodeRoPE(commandBuffer cb: MTLCommandBuffer,
                    x: MTLBuffer, xOffset: Int = 0,
                    positions: MTLBuffer, positionsOffset: Int = 0,
                    rows: Int,
                    width: Int = 512,
                    ropeDim: Int = 64,
                    inverse: Bool = false,
                    config: V4RoPE.Config = .compressedLayer) {
        precondition(rows > 0 && width > ropeDim && ropeDim % 2 == 0)
        precondition(xOffset % MemoryLayout<Float16>.size == 0)
        precondition(positionsOffset % MemoryLayout<Float>.size == 0)
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(ropePSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(positions, offset: positionsOffset, index: 1)
        var r = UInt32(rows)
        var w = UInt32(width)
        var rd = UInt32(ropeDim)
        var inv = UInt32(inverse ? 1 : 0)
        var theta = config.theta
        var factor = config.yarnFactor
        var orig = config.originalSeqLen
        var bf = config.betaFast
        var bs = config.betaSlow
        var uy = UInt32(config.useYarn ? 1 : 0)
        enc.setBytes(&r, length: 4, index: 2)
        enc.setBytes(&w, length: 4, index: 3)
        enc.setBytes(&rd, length: 4, index: 4)
        enc.setBytes(&inv, length: 4, index: 5)
        enc.setBytes(&theta, length: 4, index: 6)
        enc.setBytes(&factor, length: 4, index: 7)
        enc.setBytes(&orig, length: 4, index: 8)
        enc.setBytes(&bf, length: 4, index: 9)
        enc.setBytes(&bs, length: 4, index: 10)
        enc.setBytes(&uy, length: 4, index: 11)
        let total = rows * (ropeDim / 2)
        let tg = min(256, Int(ropePSO.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
    }
}
