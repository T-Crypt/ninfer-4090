#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The Gemma 4 head GEMM (vocabulary x hidden): hidden 5376 on the 31B, 2816 on the 26B. One
// token takes the GEMV route, decode widths take SIMT, prompt batches take sliced-K and the
// mma/tma tiles - the same ladder as n14336, with the head's K.
using Gemv = Bf16A16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using C2   = Bf16A16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C4   = Bf16A16SimtSchedule<4, 1, 8, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;

template <int K>
Bf16Launch head(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, K>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<C2, K, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<C4, K, 4>>;
    if (tokens <= 16)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR32T16W4, K>>;
    if (tokens <= 32) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, K>>;
    if (tokens <= 64) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, K>>;
    if (tokens <= 96) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, K>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, K>>;
}
} // namespace

Bf16Launch select_bf16_n262144_k5376(std::int32_t tokens) { return head<5376>(tokens); }
Bf16Launch select_bf16_n262144_k2816(std::int32_t tokens) { return head<2816>(tokens); }

} // namespace ninfer::ops::detail
