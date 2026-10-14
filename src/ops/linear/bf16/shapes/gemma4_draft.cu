#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
// The Gemma 4 assistant drafter (hidden 1024) runs one token per step: single-token GEMV only.
// Wide outputs over K=1024 take eight rows per warp; the 1024-row outputs over long K split each
// row across warps so the grid still covers the device; the input projection over the concatenated
// target-plus-draft hidden state (10752 on the 31B, 5632 on the 26B) uses one row per warp.
using Wide   = Bf16A16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct,
                                   Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using Narrow = Bf16A16GemvSchedule<8, 8, 1, 8, 4, Bf16ActivationAccess::Direct,
                                   Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
using Input  = Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                                   Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;

template <int K, class Schedule>
Bf16Launch single_token(std::int32_t tokens) {
    if (tokens != 1) {
        throw std::invalid_argument("bf16 linear: the Gemma drafter shapes run one token at a time");
    }
    return launch_bf16_gemv<Bf16ScheduleInstance<Schedule, K>>;
}
} // namespace

Bf16Launch select_bf16_n1024_k10752(std::int32_t tokens) { return single_token<10752, Input>(tokens); }
Bf16Launch select_bf16_n1024_k5632(std::int32_t tokens) { return single_token<5632, Input>(tokens); }
Bf16Launch select_bf16_n8192_k1024(std::int32_t tokens) { return single_token<1024, Wide>(tokens); }
Bf16Launch select_bf16_n16384_k1024(std::int32_t tokens) { return single_token<1024, Wide>(tokens); }
Bf16Launch select_bf16_n1024_k8192(std::int32_t tokens) { return single_token<8192, Narrow>(tokens); }
Bf16Launch select_bf16_n1024_k16384(std::int32_t tokens) { return single_token<16384, Narrow>(tokens); }
Bf16Launch select_bf16_n262144_k1024(std::int32_t tokens) { return single_token<1024, Wide>(tokens); }
Bf16Launch select_bf16_n5376_k1024(std::int32_t tokens) { return single_token<1024, Wide>(tokens); }
Bf16Launch select_bf16_n2816_k1024(std::int32_t tokens) { return single_token<1024, Wide>(tokens); }

} // namespace ninfer::ops::detail
