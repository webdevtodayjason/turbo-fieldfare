import Foundation
import Metal

/// Device-keyed access to `MetalContext` pipelines for the V4 wrappers whose
/// initializers predate the context registration of the V4 shader modules
/// (V4F-02/03). The modules (`dequant_v4`, `moe_v4`, `attention_v4`,
/// `attention_v4b`, `attention_v4c`) are now compiled into the shared
/// `MetalContext` library; this class hands out context pipelines so those
/// wrappers no longer self-compile standalone libraries. New code should
/// take a `MetalContext` and call `context.pipeline(...)` directly.
///
/// `@unchecked Sendable`: the cache is lock-guarded and Metal objects are
/// thread-safe for pipeline creation.
final class V4ShaderLibrary: @unchecked Sendable {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var contexts: [ObjectIdentifier: MetalContext] = [:]

    /// Shared context for `device`, compiling the combined library on first
    /// use per device.
    static func context(for device: MTLDevice) throws -> MetalContext {
        lock.lock()
        if let cached = contexts[ObjectIdentifier(device)] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let context = try MetalContext(device: device)
        lock.lock()
        contexts[ObjectIdentifier(device)] = context
        lock.unlock()
        return context
    }

    /// The combined shared library (for wrappers that build argument
    /// encoders from kernel functions).
    func library(device: MTLDevice,
                 module _: String,
                 subdirectory _: String) throws -> MTLLibrary {
        try Self.context(for: device).library
    }

    func pipeline(device: MTLDevice,
                  module _: String,
                  subdirectory _: String,
                  name: String) throws -> MTLComputePipelineState {
        try Self.context(for: device).pipeline(name)
    }
}
