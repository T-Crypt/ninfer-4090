#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    // sm_89 (Ada) has no programmatic dependent launch (sm_90+). Below sm_90 the
    // attribute is dropped and this degrades to a plain launch of the same kernel.
    // Device 0 is assumed homogeneous (single-GPU 4090 line).
    static const bool pdl_supported = [] {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess)
            return false;
        return major >= 9;
    }();

    cudaLaunchAttribute attribute{};
    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = 0;
    if (pdl_supported) {
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.numAttrs = 1;
    }

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer. Both intrinsics compile to
// `griddepcontrol` PTX, which ptxas rejects below sm_90 - they degrade to no-ops there.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void trigger_dependents() { cudaTriggerProgrammaticLaunchCompletion(); }

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() { cudaGridDependencySynchronize(); }
#else
__device__ __forceinline__ void trigger_dependents() {}
__device__ __forceinline__ void wait_for_dependencies() {}
#endif

} // namespace ninfer::pdl
