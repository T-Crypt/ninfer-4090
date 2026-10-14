#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// The Gemma 4 vision tower's two BF16 projections: the patch embedding (1152 x 768) over every
// patch of an image, and the soft-token projection into the text width (5376 x 1152 on the 31B,
// 2816 x 1152 on the 26B) over its pooled tokens. Images are encoded whole, so the MMA route
// carries nearly every call; a single soft token or a handful of patches takes GEMV or SIMT.
using Gemv = Bf16A16GemvSchedule<4, 1, 8, 4, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using C8   = Bf16A16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;

template <int K>
Bf16Launch vision(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, K>>;
    if (tokens <= 8) return launch_bf16_simt<Bf16ScheduleInstance<C8, K, 8>>;
    if (tokens <= 32) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, K>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, K>>;
}
} // namespace

Bf16Launch select_bf16_n1152_k768(std::int32_t tokens) { return vision<768>(tokens); }
Bf16Launch select_bf16_n5376_k1152(std::int32_t tokens) { return vision<1152>(tokens); }
Bf16Launch select_bf16_n2816_k1152(std::int32_t tokens) { return vision<1152>(tokens); }

} // namespace ninfer::ops::detail
