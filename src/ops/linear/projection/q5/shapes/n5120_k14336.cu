#include "ops/linear/projection/q5/q5_shapes.h"

namespace ninfer::ops::detail {
Q5Launch select_q5_n5120_k14336(std::int32_t tokens) {
    // Fork (prep/qwen4-27b-ops): Qwen4-27B FFN down. Routes copied from n5120_k17408 (same K blocking);
    // retune on the RTX 4090.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k14336;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w2_k14336;
    if (tokens <= 3) return launch_q5_a16_direct_r1_t3_w2_k14336;
    if (tokens <= 4) return launch_q5_a16_direct_r1_t4_w2_k14336;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 128) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 256) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
