#if !NINFER_ENABLE_NVFP4_FAMILY

#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/kv_cache/append/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// -- ops/linear/nvfp4/nvfp4_format.h --

Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

// -- ops/linear/nvfp4/nvfp4_dispatch.h --

std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                   std::int32_t input_rows,
                                                   LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

// -- ops/linear_add/nvfp4/nvfp4_linear_add_plan.h --

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                       std::int32_t input_rows,
                                                       LinearPolicy policy,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

// -- ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h --

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                       std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

// -- ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h --

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

// -- ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h --

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                              std::int32_t batch_size) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy,
                                  WorkspaceArena& workspace, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

void nvfp4_gdn_record_post_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, Tensor& query, Tensor& key,
                                   Tensor& value, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

void nvfp4_gdn_record_small_t_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, const Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      Tensor& conv_record, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

// -- ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h --

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
    return {};
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                   LinearPolicy policy, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

// -- ops/kv_cache/append/launch.h --

void kv_cache_append_nvfp4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   PagedKVLayerView cache, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

void kv_cache_append_nvfp4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                         const Tensor& valid_columns, const Tensor& table_rows,
                                         PagedKVBatchLayerView cache, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

void kv_cache_append_k8v4_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                  PagedKVLayerView cache, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

void kv_cache_append_k8v4_batch_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        PagedKVBatchLayerView cache, cudaStream_t stream) {
    throw std::invalid_argument(
        "NVFP4-family ops are Blackwell-only (sm_120a) and are not built into this Ada "
        "(sm_89) target");
}

} // namespace ninfer::ops::detail

#endif // !NINFER_ENABLE_NVFP4_FAMILY
