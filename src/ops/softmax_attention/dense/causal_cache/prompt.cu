// ninfer::ops - causal_softmax_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/paged_kv_storage.h"
#include "ops/common/math.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_bf16.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <typename Geometry, typename CacheView, typename Metadata>
void causal_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                  float scale, const CacheView& cache,
                                                  Metadata metadata, Tensor& out,
                                                  cudaStream_t stream) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(causal_attention_prompt_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptSmemBytes);
    CUDA_CHECK(attr_bf16);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (kv_storage_is_int8_family(cache.storage)) {
        const KvForkModeFlags mode = kv_fork_mode_flags(cache.storage);
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        const auto launch_i8 = [&]<bool PackedV, bool RotateK, bool RotateV, bool PackedK,
                                   bool E8Root>() {
            static const cudaError_t attr_i8 = cudaFuncSetAttribute(
                causal_attention_prompt_i8_kernel<Geometry, PackedV, RotateK, RotateV, PackedK,
                                                  E8Root, Metadata>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, kCausalPromptI8SmemBytes);
            CUDA_CHECK(attr_i8);
            causal_attention_prompt_i8_kernel<Geometry, PackedV, RotateK, RotateV, PackedK, E8Root,
                                              Metadata>
                <<<attention_grid, kCausalPromptI8Threads, kCausalPromptI8SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::int8_t*>(cache_k.data),
                    static_cast<const std::uint8_t*>(cache_v.data),
                    static_cast<const __half*>(cache_k_scale.data),
                    static_cast<const __half*>(cache_v_scale.data), metadata,
                    static_cast<const std::int32_t*>(positions.data), scale,
                    static_cast<__nv_bfloat16*>(out.data), tokens);
        };
        if (mode.e8_root) {
            launch_i8.template operator()<true, true, true, false, true>();
        } else if (mode.packed_k) {
            launch_i8.template operator()<true, true, true, true, false>();
        } else if (mode.packed_v) {
            launch_i8.template operator()<true, true, true, false, false>();
        } else {
            launch_i8.template operator()<false, false, false, false, false>();
        }
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kCausalPromptBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        causal_attention_prompt_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kCausalPromptThreads, kCausalPromptSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __half*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
    if (kv_fork_mode_flags(cache.storage).rotate_v) {
        kv_cache_inverse_rotate_output_kernel<Geometry::QHeads>
            <<<tokens * Geometry::QHeads * kKVCacheInt8Groups, 32, 0, stream>>>(
                static_cast<__nv_bfloat16*>(out.data), tokens, tokens, 0, nullptr);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void causal_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                              const PagedKVLayerView& cache, Tensor& out,
                                              cudaStream_t stream) {
    // v3 port: upstream storages run their own families in causal_softmax_attention.cpp; this
    // generic launcher serves BF16 and the int8 family (fork rotated/packed/E8 storages) only.
    if (cache.storage != KvCacheStorage::BFloat16 && !kv_storage_is_int8_family(cache.storage)) {
        throw std::invalid_argument("causal_attention_prompt_attention_launch: storage is not served by the generic launcher");
    }
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == CausalD256H24Kv4::QHeads) {
        causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(q, positions, scale, cache,
                                                                       metadata, out, stream);
        return;
    }
    causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                   metadata, out, stream);
}

void causal_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& positions, const Tensor& valid_columns,
                                    const Tensor& table_rows, float scale,
                                    PagedKVBatchLayerView cache, Tensor& out, cudaStream_t stream) {
    // v3 port: upstream storages run their own families in causal_softmax_attention.cpp; this
    // generic launcher serves BF16 and the int8 family (fork rotated/packed/E8 storages) only.
    if (cache.storage != KvCacheStorage::BFloat16 && !kv_storage_is_int8_family(cache.storage)) {
        throw std::invalid_argument("causal_attention_prompt_launch: storage is not served by the generic launcher");
    }
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            causal_attention_prompt_attention_launch_for<CausalD256H24Kv4>(
                q, positions, scale, cache, metadata, out, stream);
            return;
        }
        causal_attention_prompt_attention_launch_for<CausalD256H16Kv2>(q, positions, scale, cache,
                                                                       metadata, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
