/*!
 * \file causal_conv1d.h
 * \brief CausalConv1D (prefill/extend) AscendC kernel implementation.
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_common.h"

constexpr int32_t CCONV_DBG_SEQ = -1;
constexpr int32_t CCONV_DBG_C0 = -1;
constexpr int32_t CCONV_DBG_MAX_TOKENS = 0;
constexpr int32_t CCONV_DBG_VERBOSE_TOKENS = 0;
constexpr int32_t CCONV_DBG_DUMP_SIZE = 0;
constexpr bool CCONV_DBG_PRINT_SYNC = false;
constexpr bool CCONV_DBG_DUMP_WEIGHTS = false;
constexpr bool CCONV_DBG_DUMP_BIAS = false;
constexpr bool CCONV_DBG_DUMP_INIT_RING = false;
constexpr bool CCONV_DBG_DUMP_RUNSEQ = false;
constexpr bool CCONV_DBG_DUMP_PREFETCH = false;
constexpr bool CCONV_DBG_DUMP_STATE = false;

using namespace AscendC;

namespace NsCausalConv1d {

using namespace NsCausalConv1dCommon;
using sglang::npu_kernel::CausalConv1dTilingData;

template <typename T>
class CausalConv1d
{
public:
    __aicore__ inline CausalConv1d() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR hasInitialState, GM_ADDR y,
                                const CausalConv1dTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg);
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len, int32_t c0,
                                    int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim,
                                          bool dbg);
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();

private:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;
    // PR-2: castedRing[s] holds FP32 cast of ring[s]. Each slot cast once
    // when prefetched, reused 4× across 4 iters as it cycles slotCurr→slotH3.
    TBuf<QuePosition::VECCALC> castedRingBuf;
    // PR-4: dedicated scratch for the strided weight load (4 slices in one
    // MTE2). Separate from ring so back-to-back tasks (when grid > core_num)
    // don't race task N's WriteBackState MTE3 vs task N+1's LoadWeightAndBias MTE2.
    TBuf<QuePosition::VECIN> weightScratchBuf;

    TEventID tempVToMte2Event_;
    TEventID tempMte2ToVEvent_;
    TEventID inputMte2ToVEvent_;
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<int32_t> cacheIndicesGm;
    GlobalTensor<bool> hasInitialStateGm;
    GlobalTensor<T> yGm;

    const CausalConv1dTilingData *tilingData_{nullptr};
};

template <typename T>
__aicore__ inline void CausalConv1d<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                             GM_ADDR y, const CausalConv1dTilingData *tilingData)
{
    tilingData_ = tilingData;

    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
    if (tilingData_->hasBias != 0) {
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
    }
    convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
    queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
    cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cacheIndices));
    hasInitialStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(hasInitialState));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    // PR-2: dynamic sizing per actual dimTileSize. Slot stride = dimTileSize.
    // Clamp to MAX_BLOCK_DIM (2048) — Process() rejects dt > MAX_BLOCK_DIM.
    const int32_t dt_raw = static_cast<int32_t>(tilingData_->dimTileSize);
    const int32_t dt = (dt_raw > 0 && dt_raw <= MAX_BLOCK_DIM) ? dt_raw : MAX_BLOCK_DIM;
    pipe.InitBuffer(inBuf, RING_SLOTS * dt * sizeof(T));
    pipe.InitBuffer(outBuf, 2 * dt * sizeof(T));
    // calcBuf: weightF[MAX_WIDTH] + biasF + accF + tmpF (tmpF needed for
    // activation-out-of-place; castedRing handles input casting only).
    pipe.InitBuffer(calcBuf, (MAX_WIDTH + 3) * dt * sizeof(float));
    pipe.InitBuffer(castedRingBuf, RING_SLOTS * dt * sizeof(float));
    pipe.InitBuffer(weightScratchBuf, MAX_WIDTH * dt * sizeof(T));

    AllocEvents();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::AllocEvents()
{
    tempVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    tempMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    inputMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::ReleaseEvents()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(tempVToMte2Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(tempMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg)
{
    const int32_t dim = tilingData_->dim;
    (void)dbg;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * dimTileSize];
    LocalTensor<T> tempT = outBuf.Get<T>();

    // PR-5: between tasks (grid > core_num case), drain all pipes so prior
    // task's events/buffers are clean before this task starts.
    PipeBarrier<PIPE_ALL>();

    // PR-4: strided DataCopy loads all 4 weight slices in one MTE2 op.
    // GM stride between weight rows is `dim`, UB destination contiguous in
    // weightScratchBuf (separate from ring to avoid inter-task races).
    // One big Cast then converts everything to FP32.
    LocalTensor<T> weightScratch = weightScratchBuf.Get<T>();
    DataCopyParams weightParams{};
    weightParams.blockCount = static_cast<uint16_t>(MAX_WIDTH);
    weightParams.blockLen = static_cast<uint16_t>(dimTileSize * sizeof(T) / 32);
    weightParams.srcStride = static_cast<uint16_t>((dim - dimTileSize) * sizeof(T) / 32);
    weightParams.dstStride = 0;
    DataCopy(weightScratch, weightGm[c0], weightParams);
    SetFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
    WaitFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
    Cast(weightF, weightScratch, RoundMode::CAST_NONE, MAX_WIDTH * dimTileSize);

    if (tilingData_->hasBias != 0) {
        // Bias load reuses tempT slot (no longer needed for weight chunks).
        SetFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        DataCopy(tempT, biasGm[c0], dimTileSize);
        SetFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
        Cast(biasF, tempT, RoundMode::CAST_NONE, dimTileSize);
    } else {
        Duplicate(biasF, 0.0f, dimTileSize);
    }

}

template <typename T>
__aicore__ inline void CausalConv1d<T>::InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len, int32_t c0,
                                                 int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    (void)dbg;
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<float> castedRing = castedRingBuf.Get<float>();

    // PR-5: when grid > core_num, blocks run multiple tasks back-to-back.
    // Prior task's WriteBackState issued MTE3 reads from `ring` — drain them
    // before this task writes `ring` (MTE2 or V Duplicate). PIPE_MTE3 only,
    // doesn't stall V or MTE2 elsewhere.
    PipeBarrier<PIPE_MTE3>();

    // PR-1: fill history (slots 0..W-2) and SlotCurr(0). PR-2: also cast each
    // slot into castedRing so RunSeq's first iter can read FP32 directly.
    if (hasInit) {
        // PR-4: one strided DataCopy fetches all W-1 history slots from
        // convStates in a single MTE2 op (stride `dim` between slot rows).
        DataCopyParams histParams{};
        histParams.blockCount = static_cast<uint16_t>(MAX_WIDTH - 1);
        histParams.blockLen = static_cast<uint16_t>(dimTileSize * sizeof(T) / 32);
        histParams.srcStride = static_cast<uint16_t>((dim - dimTileSize) * sizeof(T) / 32);
        histParams.dstStride = 0;
        const int64_t stateBase = static_cast<int64_t>(cacheIdx) * stateLen * dim + c0;
        DataCopy(ring, convStatesGm[stateBase], histParams);
        SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_);
        // One big Cast across the (W-1)*dimTileSize contiguous range.
        Cast(castedRing, ring, RoundMode::CAST_NONE, (MAX_WIDTH - 1) * dimTileSize);
    } else {
        // History is zero. Fill both ring (for WriteBackState reads) and
        // castedRing (for V conv reads).
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            Duplicate(ring[i * dimTileSize], static_cast<T>(0), dimTileSize);
            Duplicate(castedRing[i * dimTileSize], 0.0f, dimTileSize);
        }
    }

    if (len > 0) {
        const int64_t xOffset = static_cast<int64_t>(start) * dim + c0;
        DataCopy(ring[SlotCurr(0) * dimTileSize], xGm[xOffset], dimTileSize);
    }
    // Signal MTE2 writes done — first RunSeq Wait then casts SlotCurr(0).
    SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_);
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim,
                                               bool dbg)
{
    (void)dbg;
    LocalTensor<float> calc = calcBuf.Get<float>();
    LocalTensor<float> weightF = calc;
    LocalTensor<float> biasF = weightF[MAX_WIDTH * dimTileSize];
    LocalTensor<float> accF = biasF[dimTileSize];
    LocalTensor<float> tmpF = accF[dimTileSize];
    LocalTensor<T> ring = inBuf.Get<T>();
    LocalTensor<T> outT = outBuf.Get<T>();
    LocalTensor<float> castedRing = castedRingBuf.Get<float>();
    const bool hasActivation = (tilingData_->activationMode != 0);

    // PR-2: each slot in ring is cast to FP32 once (right after prefetch),
    // stored in castedRing, and reused across 4 iters (slotCurr→slotH3 lifecycle).
    // V op count drops 9→6 per token: -3 Casts (4 taps now read FP32 castedRing).
    // PR-1: fine-grained sync (MTE2_V, V_MTE2, V_MTE3, MTE3_V) replaces full bars.
    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);
        const int32_t slotPref = (t + 1 < len) ? SlotPrefetch(t) : -1;
        const int32_t outSlot = t & 1;

        // V→MTE2: V of iter t-1 read slot (t-1)%5; iter t's MTE2 writes (t+4)%5
        // which is the same slot 5 iters later. FIFO MTE2 pipe orders prior MTE2,
        // but V must have READ before MTE2 overwrites.
        if (t > 0 && (t + 1 < len)) {
            WaitFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        }

        if (t + 1 < len) {
            const int64_t xOffset = static_cast<int64_t>(start + t + 1) * dim + c0;
            DataCopy(ring[slotPref * dimTileSize], xGm[xOffset], dimTileSize);
            SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_);
        }

        // Slots ready for V (Set by InitRing for t=0, by iter t-1's prefetch
        // for t>=1). len Sets, len Waits — balanced.
        WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_);

        // PR-2: Cast the slot that JUST became valid (SlotCurr of this iter)
        // exactly once. Slots H1, H2, H3 already cached from prior iters.
        Cast(castedRing[slotCurr * dimTileSize], ring[slotCurr * dimTileSize],
             RoundMode::CAST_NONE, dimTileSize);

        // PR-3: when !hasBias, fuse accF init into first Mul (saves the
        // accF=biasF DataCopy). With bias we keep the init-with-bias pattern
        // because Add-at-end + Silu-after both stress the Add pipeline.
        if (tilingData_->hasBias != 0) {
            DataCopy(accF, biasF, dimTileSize);
#pragma unroll
            for (int32_t j = 0; j < MAX_WIDTH; ++j) {
                const int32_t tap = (MAX_WIDTH - 1) - j;
                const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
                MulAddDst(accF, castedRing[slot * dimTileSize], weightF[j * dimTileSize], dimTileSize);
            }
        } else {
            const int32_t slotH3 = SlotHist(t, 3);
            Mul(accF, castedRing[slotH3 * dimTileSize], weightF[0], dimTileSize);
#pragma unroll
            for (int32_t j = 1; j < MAX_WIDTH; ++j) {
                const int32_t tap = (MAX_WIDTH - 1) - j;
                const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
                MulAddDst(accF, castedRing[slot * dimTileSize], weightF[j * dimTileSize], dimTileSize);
            }
        }

        if (hasActivation) {
            Silu(tmpF, accF, dimTileSize);  // separate dst — matches upstream
        }

        // V→MTE2: V of iter t finished reading slot t%5 (slotH3); iter t+1's
        // MTE2 will overwrite slot t%5. Only signal if iter t+1 issues MTE2.
        if (t + 2 < len) {
            SetFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        }

        // MTE3→V: iter t-2 wrote outT[outSlot]; wait until its MTE3 drained.
        if (t >= 2) {
            WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outT[outSlot * dimTileSize], tmpF, dimTileSize);
            } else {
                DataCopy(outT[outSlot * dimTileSize], accF, dimTileSize);
            }
        } else {
            if (hasActivation) {
                Cast(outT[outSlot * dimTileSize], tmpF, RoundMode::CAST_RINT, dimTileSize);
            } else {
                Cast(outT[outSlot * dimTileSize], accF, RoundMode::CAST_RINT, dimTileSize);
            }
        }
        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + c0;
        DataCopy(yGm[outOffset], outT[outSlot * dimTileSize], dimTileSize);
        SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
    }

    // Drain pending MTE3_V Sets (last two iters' outSlots have no matching Wait).
    if (len >= 1) {
        WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[(len - 1) & 1]);
    }
    if (len >= 2) {
        WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[(len - 2) & 1]);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0, int32_t dimTileSize,
                                                       int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;
    (void)dbg;
    if (len <= 0) {
        return;
    }
    const int32_t lastT = len - 1;
    LocalTensor<T> ring = inBuf.Get<T>();

    // PR-1: by entry, all MTE2/V/MTE3 ops in RunSeq have retired (drained via
    // MTE3_V WaitFlags). Ring slots are stable. PR-4: would be a single strided
    // DataCopy, but the source slots aren't contiguous in UB (they're scattered
    // ring positions selected by the mod-5 arithmetic). Keep one-MTE3-per-slot.
    for (int32_t pos = 0; pos < (MAX_WIDTH - 1); ++pos) {
        const int32_t tap = (MAX_WIDTH - 2) - pos;
        const int32_t slot = (tap == 0) ? SlotCurr(lastT) : SlotHist(lastT, tap);
        const int64_t stateOffset =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(pos) * dim + c0;
        DataCopy(convStatesGm[stateOffset], ring[slot * dimTileSize], dimTileSize);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::Process()
{
    const int32_t dim = tilingData_->dim;
    const int32_t batch = tilingData_->batch;
    const int32_t inputMode = tilingData_->inputMode;
    const int32_t seqLen = tilingData_->seqLen;
    const int32_t dimTileSize = static_cast<int32_t>(tilingData_->dimTileSize);
    const int32_t blocksPerSeq = static_cast<int32_t>(tilingData_->blocksPerSeq);

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (dimTileSize <= 0 || blocksPerSeq <= 0 || dimTileSize > MAX_BLOCK_DIM || blocksPerSeq * dimTileSize != dim) {
        ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * blocksPerSeq;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / blocksPerSeq);
        const int32_t dimBlockId = static_cast<int32_t>(task % blocksPerSeq);
        const int32_t c0 = dimBlockId * dimTileSize;
        const bool dbg = (seq == CCONV_DBG_SEQ) && (c0 == CCONV_DBG_C0);
        LoadWeightAndBias(c0, dimTileSize, dbg);

        int32_t start = 0;
        int32_t len = 0;
        if (inputMode == 0) {
            const int32_t startVal = queryStartLocGm.GetValue(seq);
            const int32_t endVal = queryStartLocGm.GetValue(seq + 1);
            start = startVal;
            len = endVal - startVal;
        } else {
            start = seq * seqLen;
            len = seqLen;
        }

        if (len <= 0) {
            continue;
        }

        const int32_t cacheIdx = cacheIndicesGm.GetValue(seq);
        if (cacheIdx == tilingData_->padSlotId) {
            continue;
        }

        const bool hasInit = hasInitialStateGm.GetValue(seq);
        InitRing(cacheIdx, hasInit, start, len, c0, dimTileSize, dim, dbg);
        RunSeq(start, len, c0, dimTileSize, dim, dbg);
        WriteBackState(cacheIdx, len, c0, dimTileSize, dim, dbg);
    }

    ReleaseEvents();
}

}  // namespace NsCausalConv1d

#endif  // CAUSAL_CONV1D_H
