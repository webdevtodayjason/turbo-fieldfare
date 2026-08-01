import Foundation
import Metal

/// Self-contained loader for the V4 shader modules (`dequant_v4`, `moe_v4`).
///
/// `MetalContext` compiles a hardcoded module list that production code owns;
/// V4F-02 must not edit it, so the V4 wrappers compile their own library from
/// the same bundled `.metal` sources (the whole `Metal/` tree is copied into
/// the module bundle by Package.swift). Once the modules are registered in
/// `MetalContext.shaderModules`/`shaderSubdirectories`, this loader can be
/// deleted and the wrappers switched to `context.pipeline(...)`.
///
/// `@unchecked Sendable`: the caches are lock-guarded and Metal objects are
/// thread-safe for pipeline creation.
final class V4ShaderLibrary: @unchecked Sendable {
    private struct PipelineKey: Hashable {
        var module: String
        var name: String
    }

    private var libraries: [String: MTLLibrary] = [:]
    private var pipelines: [PipelineKey: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    func library(device: MTLDevice,
                 module: String,
                 subdirectory: String) throws -> MTLLibrary {
        lock.lock()
        if let cached = libraries[module] {
            lock.unlock()
            return cached
        }
        lock.unlock()

        guard let url = Bundle.module.url(forResource: module,
                                          withExtension: "metal",
                                          subdirectory: subdirectory) else {
            throw MetalError.missingShaderResource(module)
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw MetalError.libraryCompileFailed("\(error)")
        }
        lock.lock()
        libraries[module] = library
        lock.unlock()
        return library
    }

    func pipeline(device: MTLDevice,
                  module: String,
                  subdirectory: String,
                  name: String) throws -> MTLComputePipelineState {
        let key = PipelineKey(module: module, name: name)
        lock.lock()
        let cached = pipelines[key]
        lock.unlock()
        if let cached { return cached }

        let library = try self.library(device: device,
                                       module: module,
                                       subdirectory: subdirectory)
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        lock.lock()
        pipelines[key] = pipeline
        lock.unlock()
        return pipeline
    }
}
