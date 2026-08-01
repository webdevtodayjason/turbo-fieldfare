import Foundation
import Metal

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh"
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }
}

/// Compile-time architecture baseline for DeepSeek V4-family installs
/// (`manifest.json -> modelFamily == "deepseek-v4-flash"`, versionMinor 1).
/// Field names mirror the flat DeepSeek `config.json` schema, which is what
/// the V4 repacker writes into `manifest.json -> arch` (V4F-01). The runtime
/// loader cross-checks every field at load time; mismatches throw
/// `ModelError.archMismatch`. Separate type from `ArchConfig` so the Gemma
/// gate stays byte-identical.
public struct V4ArchConfig: Sendable, Equatable {
    public let numLayers: Int
    public let hiddenSize: Int
    public let vocabSize: Int
    public let numExperts: Int
    public let numSharedExperts: Int
    public let topKExperts: Int
    public let moeIntermediateSize: Int
    public let numHashLayers: Int
    public let numMTPLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let qLoraRank: Int
    public let qkRopeHeadDim: Int
    public let oGroups: Int
    public let oLoraRank: Int
    public let indexNHeads: Int
    public let indexHeadDim: Int
    public let indexTopk: Int
    public let slidingWindow: Int
    public let ropeTheta: Double
    public let compressRopeTheta: Double
    public let compressRatios: [Int]
    public let routedScalingFactor: Double
    public let swigluLimit: Double
    public let normTopkProb: Bool
    public let scoringFunc: String
    public let topkMethod: String
    public let hiddenActivation: String
    public let hcMult: Int
    public let hcEps: Double
    public let hcSinkhornIters: Int
    public let rmsNormEps: Double
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int
    public let yarnFactor: Double
    public let yarnOriginalMaxPositions: Int
    public let yarnBetaFast: Double
    public let yarnBetaSlow: Double

    public init(numLayers: Int, hiddenSize: Int, vocabSize: Int,
                numExperts: Int, numSharedExperts: Int, topKExperts: Int,
                moeIntermediateSize: Int, numHashLayers: Int, numMTPLayers: Int,
                numHeads: Int, numKVHeads: Int, headDim: Int,
                qLoraRank: Int, qkRopeHeadDim: Int, oGroups: Int,
                oLoraRank: Int, indexNHeads: Int, indexHeadDim: Int,
                indexTopk: Int, slidingWindow: Int,
                ropeTheta: Double, compressRopeTheta: Double,
                compressRatios: [Int], routedScalingFactor: Double,
                swigluLimit: Double, normTopkProb: Bool,
                scoringFunc: String, topkMethod: String,
                hiddenActivation: String, hcMult: Int, hcEps: Double,
                hcSinkhornIters: Int, rmsNormEps: Double,
                tieWordEmbeddings: Bool, maxPositionEmbeddings: Int,
                yarnFactor: Double, yarnOriginalMaxPositions: Int,
                yarnBetaFast: Double, yarnBetaSlow: Double) {
        self.numLayers = numLayers
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numExperts = numExperts
        self.numSharedExperts = numSharedExperts
        self.topKExperts = topKExperts
        self.moeIntermediateSize = moeIntermediateSize
        self.numHashLayers = numHashLayers
        self.numMTPLayers = numMTPLayers
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.qLoraRank = qLoraRank
        self.qkRopeHeadDim = qkRopeHeadDim
        self.oGroups = oGroups
        self.oLoraRank = oLoraRank
        self.indexNHeads = indexNHeads
        self.indexHeadDim = indexHeadDim
        self.indexTopk = indexTopk
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.compressRopeTheta = compressRopeTheta
        self.compressRatios = compressRatios
        self.routedScalingFactor = routedScalingFactor
        self.swigluLimit = swigluLimit
        self.normTopkProb = normTopkProb
        self.scoringFunc = scoringFunc
        self.topkMethod = topkMethod
        self.hiddenActivation = hiddenActivation
        self.hcMult = hcMult
        self.hcEps = hcEps
        self.hcSinkhornIters = hcSinkhornIters
        self.rmsNormEps = rmsNormEps
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.yarnFactor = yarnFactor
        self.yarnOriginalMaxPositions = yarnOriginalMaxPositions
        self.yarnBetaFast = yarnBetaFast
        self.yarnBetaSlow = yarnBetaSlow
    }

    /// Canonical DeepSeek V4-Flash baseline (43 layers, ratio 0 on 0/1/42,
    /// CSA ratio 4 on even 2...40, HCA ratio 128 on odd 3...41).
    public static let deepSeekV4Flash = V4ArchConfig(
        numLayers: 43,
        hiddenSize: 4096,
        vocabSize: 129280,
        numExperts: 256,
        numSharedExperts: 1,
        topKExperts: 6,
        moeIntermediateSize: 2048,
        numHashLayers: 3,
        numMTPLayers: 1,
        numHeads: 64,
        numKVHeads: 1,
        headDim: 512,
        qLoraRank: 1024,
        qkRopeHeadDim: 64,
        oGroups: 8,
        oLoraRank: 1024,
        indexNHeads: 64,
        indexHeadDim: 128,
        indexTopk: 512,
        slidingWindow: 128,
        ropeTheta: 10_000.0,
        compressRopeTheta: 160_000.0,
        compressRatios: V4ArchConfig.flashCompressRatios(),
        routedScalingFactor: 1.5,
        swigluLimit: 10.0,
        normTopkProb: true,
        scoringFunc: "sqrtsoftplus",
        topkMethod: "noaux_tc",
        hiddenActivation: "silu",
        hcMult: 4,
        hcEps: 1e-6,
        hcSinkhornIters: 20,
        rmsNormEps: 1e-6,
        tieWordEmbeddings: false,
        maxPositionEmbeddings: 1_048_576,
        yarnFactor: 16.0,
        yarnOriginalMaxPositions: 65_536,
        yarnBetaFast: 32.0,
        yarnBetaSlow: 1.0)

    private static func flashCompressRatios() -> [Int] {
        (0..<43).map { layer in
            if layer == 0 || layer == 1 || layer == 42 { return 0 }
            return layer % 2 == 0 ? 4 : 128
        }
    }

    /// Per-layer attention variant from `compressRatios` (0 = passthrough
    /// sliding window, 4 = CSA, 128 = HCA), matching `V4CacheConfig.kind`.
    public func layerKind(_ layer: Int) -> V4LayerKind {
        switch compressRatios[layer] {
        case 4:   return .csa
        case 128: return .hca
        default:  return .passthrough
        }
    }

    /// True for layers 0..<numHashLayers: expert ids come from the fixed
    /// `tid2eid` table (no router selection kernel, no bias tensor).
    public func isHashRouted(layer: Int) -> Bool {
        layer < numHashLayers
    }

    /// Cache geometry for `CompressedKVCacheManager`.
    public var cacheConfig: V4CacheConfig {
        V4CacheConfig(compressRatios: compressRatios,
                      headDim: headDim,
                      ropeDim: qkRopeHeadDim,
                      window: slidingWindow,
                      numQHeads: numHeads)
    }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)
    case unsupportedModelFamily(found: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        case .unsupportedModelFamily(let found):
            return "manifest modelFamily \"\(found)\" is not supported by this loader"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
