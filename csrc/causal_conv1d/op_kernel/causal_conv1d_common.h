/*!
 * \file causal_conv1d_common.h
 * \brief Common utilities and constants for CausalConv1D prefill kernel.
 */

#ifndef CAUSAL_CONV1D_COMMON_H
#define CAUSAL_CONV1D_COMMON_H

#include "kernel_operator.h"

namespace NsCausalConv1dCommon {

constexpr int32_t MAX_WIDTH = 4;
// PR-2: reduced from 4096 to 2048. UB budget at dt=2048: inBuf 20KB + outBuf 8KB
// + calcBuf 48KB + castedRingBuf 40KB = 116KB (of 192KB). dt=4096 would overflow.
constexpr int32_t MAX_BLOCK_DIM = 2048;
constexpr int32_t RING_SLOTS = 5;

__aicore__ inline int32_t SlotCurr(int32_t t)
{
    return (t + 3) % RING_SLOTS;
}

__aicore__ inline int32_t SlotHist(int32_t t, int32_t i)
{
    return (t + 3 - i) % RING_SLOTS;
}

__aicore__ inline int32_t SlotPrefetch(int32_t t)
{
    return (t + 4) % RING_SLOTS;
}

}  // namespace NsCausalConv1dCommon

#endif  // CAUSAL_CONV1D_COMMON_H
