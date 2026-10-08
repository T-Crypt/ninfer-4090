#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {

// ---------------------------------------------------------------------------
// sm_89 adaptation (KKCF9MR stage 2): ptxas rejects the TMA transport
// (cp.async.bulk.tensor + CUtensorMap descriptors) below sm_90, so this kernel
// stages the A and B tiles with per-thread cp.async copies instead. The TMA
// descriptors and expect_tx transaction counts are gone: each producer thread
// issues its own cp.async copies for its share of rows, commits one group per
// stage, waits for that group with cp.async.wait_group, and then arrives on
// the stage's full mbarrier (arrival count kProducerThreads instead of the
// single TMA transaction). Consumers keep the same full/empty barrier protocol
// and phase parities, so the empty-wait parity flip and full-wait parity below
// are unchanged. The 16-byte copy destinations are placed with
// bf16_mma_shared_col (Tma128 swizzle), bit-exactly reproducing the TMA
// CU_TENSOR_MAP_SWIZZLE_128B layout the consumers read.
//
// Known cost vs TMA: one cp.async commit group per stage per thread (completion
// is per-thread, so the stage handoff is bounded by the slowest producer
// thread's memory latency rather than a single bulk copy). Correctness first;
// the lost transport depth is accepted for now.
// ---------------------------------------------------------------------------

// Stage rows (A rows first, then B rows) are spread across the producer
// threads as contiguous runs; thread pid copies its run for each stage.
template <class Schedule>
__device__ __forceinline__ void bf16_cp_stage_rows(const __nv_bfloat16* weight,
                                                   const __nv_bfloat16* x,
                                                   __nv_bfloat16* a, __nv_bfloat16* b,
                                                   int stage, int kt, int k, int row_begin,
                                                   int token_begin, int token_end, int pid) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int P  = Schedule::kProducerThreads;
    constexpr int rows_per = (BR + BT + P - 1) / P;
    const int first  = pid * rows_per, last = min(first + rows_per, BR + BT);
    auto copy_row = [&](int row, int global_row, const __nv_bfloat16* base, __nv_bfloat16* stage_base) {
        const bool zfill = global_row >= token_end;
#pragma unroll
        for (int chunk = 0; chunk < BK / 8; ++chunk) {
            // 16-byte chunks land on the TMA 128B-swizzle positions the consumers read.
            const __nv_bfloat16* src =
                base + static_cast<std::size_t>(global_row) * k + kt * BK + chunk * 8;
            __nv_bfloat16* dst =
                stage_base + row * BK + bf16_mma_shared_col<Schedule>(row, chunk * 8);
            if (zfill) {
                cp_async_zfill<16>(dst, src, 0);
            } else {
                cp_async<16>(dst, src);
            }
        }
    };
    for (int r = first; r < min(last, BR); ++r)
        copy_row(r, row_begin + r, weight, a + stage * BR * BK);
    for (int r = max(first, BR); r < last; ++r)
        copy_row(r - BR, token_begin + r - BR, x, b + stage * BT * BK);
}

template <class Schedule, class Epilogue>
inline constexpr int bf16_tma_scratch_bytes =
    ((Schedule::kTensorBytes > bf16_epilogue_bytes<Schedule, Epilogue>
          ? Schedule::kTensorBytes
          : bf16_epilogue_bytes<Schedule, Epilogue>)+127) /
    128 * 128;

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
    const __nv_bfloat16* weight, const __nv_bfloat16* x, Output output, Epilogue epilogue,
    int rows, int input_rows, int token_offset, int count) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    const int K       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_r = rows / BR, tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, Schedule::kProducerThreads);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k = K / BK;
    if (threadIdx.x < Schedule::kProducerThreads) {
        for (int kt = 0; kt < tiles_k; ++kt) {
            const int stage = kt % S;
            cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
            bf16_cp_stage_rows<Schedule>(weight, x, a, b, stage, kt, K, row_begin, token_begin,
                                         token_offset + count, threadIdx.x);
            cp_commit();
            // Per-thread cp.async completion: this thread's group must be done before it
            // signals the stage, so every producer arrival is backed by real data.
            cp_wait<0>();
            cta_mbarrier_arrive(full + stage);
        }
        return;
    }
    const int tid  = threadIdx.x - Schedule::kProducerThreads;
    const int warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp,
                                         lane);
        // Every lane must finish its shared reads before the elected lane releases this stage.
        // cp.async visibility follows the same mbarrier ordering as TMA: each producer
        // arrives on full only after its own wait_group, and a stage is reused only
        // after every consumer warp has arrived on empty.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp,
                                               lane);
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
    // TMA descriptors are gone on sm_89; the kernel takes raw pointers, so the
    // launcher owns the validation make_bf16_tma_descriptors used to perform.
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("BF16 TMA requires complete row/K tiles");
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            bf16_prepare_shared<bytes, kernel>();
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                p.weight, p.x, output, epilogue, p.rows, p.k, offset, count);
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace ninfer::ops::detail
