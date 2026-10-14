#pragma once

#include "ops/linear/bf16/bf16_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);

// Gemma 4 shapes: head (vocabulary x hidden), vision tower, assistant drafter. 5376/10752 are
// the 31B widths, 2816/5632 the 26B's.
[[nodiscard]] Bf16Launch select_bf16_n262144_k5376(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n262144_k2816(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k768(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5376_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2816_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k10752(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k5632(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n8192_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n16384_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k8192(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k16384(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n262144_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5376_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2816_k1024(std::int32_t tokens);

} // namespace ninfer::ops::detail
