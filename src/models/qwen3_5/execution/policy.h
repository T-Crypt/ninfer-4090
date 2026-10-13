#pragma once

#include "models/qwen3_5/program/program.h"
#include "ninfer/ops/linear.h"

#include <cstdlib>

namespace ninfer::models::qwen3_5::execution {

// INT8 group-64 activations for the Q4/Q5 dense body during prefill (fork carry of e369447a, soohl
// 19aa1337): +60-70% prefill on sm_89 at a perplexity cost of +0.01%. The artifact declares every
// groupwise use A16Only, so the phase decides here, as the v2 fork's groupwise_policy(phase) did.
// Decode, MTP verification, vision and the MTP layer stay on the declared policy. INT8 is lossy,
// so NINFER_A16_PREFILL=1 (read once per process) holds prefill on A16 for quality A/B.
inline bool a16_prefill_forced() {
    static const bool forced = [] {
        const char* value = std::getenv("NINFER_A16_PREFILL");
        return value != nullptr && value[0] == '1';
    }();
    return forced;
}

inline ops::LinearPolicy body_policy(const Weight& weight, ops::LinearPolicy declared,
                                     TextPhase phase) {
    const bool groupwise =
        weight.qtype == QType::Q4_G64_FP16 || weight.qtype == QType::Q5_G64_FP16;
    return phase == TextPhase::Prefill && groupwise && declared == ops::LinearPolicy::A16Only &&
                   !a16_prefill_forced()
               ? ops::LinearPolicy::AllowA8
               : declared;
}

} // namespace ninfer::models::qwen3_5::execution
