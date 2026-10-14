#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/program.h"
#include "core/device.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last, TextPhase phase = TextPhase::Verify);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream,
                          TextPhase phase = TextPhase::Verify);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               DeviceExecutionView execution);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               DeviceExecutionView execution);

} // namespace ninfer::models::qwen3_5::execution
