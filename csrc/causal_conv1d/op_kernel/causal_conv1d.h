/*!
 * \file causal_conv1d.h
 * \brief CausalConv1D (prefill/extend) AscendC kernel implementation.
 *
 * Per-token depthwise width-4 causal 1D conv on the expanded `d_inner` dim.
 * Each AI core works on one (sequence, dim_block) task. Inside a task, a
 * ring buffer of MAX_WIDTH=4 slots holds the recent input tokens; for each
 * output position t we multiply-add the 4 most-recent slots against the
 * per-channel weight kernel and write the result to yGm.
 *
 * Hardware footprint: vector-only — UB / Vector pipe / MTE2 (GM->UB load) /
 * MTE3 (UB->GM store). Cube / L1 / L0A-L0C are unused.
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_common.h"

// ── Debug-build switches ───────────────────────────────────────────────────
// All compile-time constants; the optimizer prunes everything gated on them
// in a normal build. Flip during local debugging to dump tensor contents at
// specific (seq, c0, token) coordinates without recompiling consumers.
constexpr int32_t CCONV_DBG_SEQ = -1;            // only dump for this sequence index (-1 = no dump)
constexpr int32_t CCONV_DBG_C0  = -1;            // only dump for this channel-block offset
constexpr int32_t CCONV_DBG_MAX_TOKENS     = 0;  // dump at most this many tokens per RunSeq
constexpr int32_t CCONV_DBG_VERBOSE_TOKENS = 0;  // of those, how many get the full per-tap trace
constexpr int32_t CCONV_DBG_DUMP_SIZE      = 0;  // # of channels to dump per tensor (0 = all)
constexpr bool CCONV_DBG_PRINT_SYNC      = false; // print SetFlag/WaitFlag events (unused here)
constexpr bool CCONV_DBG_DUMP_WEIGHTS    = false; // dump weightF after Cast in LoadWeightAndBias
constexpr bool CCONV_DBG_DUMP_BIAS       = false; // dump biasF after Cast in LoadWeightAndBias
constexpr bool CCONV_DBG_DUMP_INIT_RING  = false; // dump ring slots after InitRing
constexpr bool CCONV_DBG_DUMP_RUNSEQ     = false; // dump accF/tmpF mid-conv inside RunSeq
constexpr bool CCONV_DBG_DUMP_PREFETCH   = false; // dump ring[slotPref] after each prefetch
constexpr bool CCONV_DBG_DUMP_STATE      = false; // dump convStatesGm after WriteBackState

using namespace AscendC;

namespace NsCausalConv1d {

using namespace NsCausalConv1dCommon;
using sglang::npu_kernel::CausalConv1dTilingData;

template <typename T>
class CausalConv1d
{
public:
    __aicore__ inline CausalConv1d() = default;

    // Bind GM pointers, allocate UB buffers, and reserve sync event IDs.
    // Called once per kernel launch, before Process().
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR hasInitialState, GM_ADDR y,
                                const CausalConv1dTilingData *tilingData);
    // Main driver: loops over (sequence, dim_block) tasks assigned to this AI core,
    // calling LoadWeightAndBias → InitRing → RunSeq → WriteBackState per task.
    __aicore__ inline void Process();

private:
    // Cast this task's slice of weight (W*dim) and bias (dim) from T → FP32 into
    // weightF / biasF inside calcBuf. Runs once per task. c0 = channel offset.
    __aicore__ inline void LoadWeightAndBias(int32_t c0, int32_t dimTileSize, bool dbg);
    // Prime the ring buffer: copy the (W-1) historical slots from convStatesGm
    // (or zero them if hasInit==false) and load the first input token into the
    // slot RunSeq will consume at t=0.
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len, int32_t c0,
                                    int32_t dimTileSize, int32_t dim, bool dbg);
    // The hot loop: for each token t in [0, len), prefetch the next input slot,
    // do the W-tap multiply-add against weightF, optional Silu, write to yGm.
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim, bool dbg);
    // After the last token, persist the most recent (W-1) ring slots back to
    // convStatesGm so the next call's InitRing can resume from them.
    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim,
                                          bool dbg);
    // Reserve event IDs from the TPipe scheduler. NOTE: in this upstream
    // implementation the event slots are reserved but never used — every
    // cross-pipe ordering goes through PipeBarrier<PIPE_ALL> instead. The
    // PR-1..5 optimization replaces those barriers with SetFlag/WaitFlag
    // calls using these IDs.
    __aicore__ inline void AllocEvents();
    __aicore__ inline void ReleaseEvents();

private:
    TPipe pipe;                           // owns UB allocation; passed to InitBuffer below
    TBuf<QuePosition::VECIN>  inBuf;      // ring buffer of recent input tokens (RING_SLOTS * MAX_BLOCK_DIM * T)
    TBuf<QuePosition::VECOUT> outBuf;     // staging area for outT before MTE3 writes it to yGm (2 * MAX_BLOCK_DIM * T)
    TBuf<QuePosition::VECCALC> calcBuf;   // FP32 scratch: weightF + biasF + accF + tmpF, all back-to-back

    // Pre-allocated cross-pipe event IDs. Reserved by AllocEvents() so the
    // optimized kernel can post/consume them, but UNUSED in this upstream
    // version — see AllocEvents() comment above.
    TEventID tempVToMte2Event_;           // V → MTE2 (PR-1 uses for "weight scratch free")
    TEventID tempMte2ToVEvent_;           // MTE2 → V (PR-1 uses for weight/bias load handshake)
    TEventID inputMte2ToVEvent_;          // MTE2 → V (PR-1 uses to gate ring fill handoff)
    TEventID outMte3ToVEvent_[2];         // MTE3 → V, one per outSlot 0/1 (double-buffered output)
    TEventID outVToMte3Event_[2];         // V → MTE3, one per outSlot 0/1

    // ── Global-memory tensor handles (set in Init via SetGlobalBuffer) ──────
    GlobalTensor<T>       xGm;            // input tokens — shape (batch, seq, dim) in dense mode; MTE2 reads from here
    GlobalTensor<T>       weightGm;       // depthwise weights — shape (W, dim); read once per task into weightF
    GlobalTensor<T>       biasGm;         // per-channel bias — shape (dim,); read once per task into biasF (only if hasBias)
    GlobalTensor<T>       convStatesGm;   // recurrent state cache — (num_cache_lines, state_len, dim); read by InitRing, written by WriteBackState
    GlobalTensor<int32_t> queryStartLocGm;// varlen mode: cumulative-seq-length offsets — shape (batch+1,)
    GlobalTensor<int32_t> cacheIndicesGm; // which row of convStatesGm each sequence uses — shape (batch,)
    GlobalTensor<bool>    hasInitialStateGm; // per-sequence flag: load history from convStatesGm or zero-init it — shape (batch,)
    GlobalTensor<T>       yGm;            // output tokens — shape (batch, seq, dim); MTE3 writes here

    const CausalConv1dTilingData *tilingData_{nullptr};  // host-computed tiling (dim, batch, dimTileSize, hasBias, …)
};

template <typename T>
__aicore__ inline void CausalConv1d<T>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR hasInitialState,
                                             GM_ADDR y, const CausalConv1dTilingData *tilingData)
{
    tilingData_ = tilingData;

    // Bind raw GM_ADDR (uint64_t device pointers) to typed GlobalTensor handles.
    // GlobalTensor wraps __gm__ T* and lets DataCopy figure out the access width.
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
    if (tilingData_->hasBias != 0) {
        // biasGm only meaningful when hasBias; otherwise the bias path is replaced
        // by Duplicate(biasF, 0) inside LoadWeightAndBias.
        biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
    }
    convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
    queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(queryStartLoc));
    cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cacheIndices));
    hasInitialStateGm.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(hasInitialState));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));

    // Reserve UB regions. Sizes are in BYTES.
    pipe.InitBuffer(inBuf,  RING_SLOTS  * MAX_BLOCK_DIM * sizeof(T));      // 5 slots * max-dim * T
    pipe.InitBuffer(outBuf, 2           * MAX_BLOCK_DIM * sizeof(T));      // 2-slot double buffer for outT
    pipe.InitBuffer(calcBuf,(MAX_WIDTH + 3) * MAX_BLOCK_DIM * sizeof(float)); // FP32: weightF (4 slices) + biasF + accF + tmpF

    AllocEvents();
}

// ── Allocate cross-pipe sync event IDs ──────────────────────────────────────
// Each AllocEventID<HardEvent::A_B>() reserves a SLOT in the hardware's
// cross-pipe event table. Posting a SetFlag<A_B>(id) records "pipe A reached
// this point"; the matching WaitFlag<A_B>(id) on pipe B then stalls until
// that record exists. In this upstream version the IDs are reserved but
// never posted/consumed — every cross-pipe ordering goes through the
// heavier PipeBarrier<PIPE_ALL> in the body methods below.
template <typename T>
__aicore__ inline void CausalConv1d<T>::AllocEvents()
{
    tempVToMte2Event_   = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();   // V finished using weight scratch → MTE2 may overwrite
    tempMte2ToVEvent_   = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();   // MTE2 finished loading weights/bias → V may Cast
    inputMte2ToVEvent_  = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();   // MTE2 finished prefetching ring slot → V may read it
    outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();   // MTE3 done with outT[0] → V may overwrite next iter
    outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();   // …same for outT[1] (double-buffered)
    outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();   // V finished writing outT[0] → MTE3 may store to yGm
    outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();   // …same for outT[1]
}

// Return the event slots to the hardware event table at end of Process().
// Must release exactly what was allocated (same HardEvent kind, same ID).
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
    const int32_t dim = tilingData_->dim;           // full channel count (depthwise: W*dim weights)
    const bool dbgSync = dbg && CCONV_DBG_PRINT_SYNC;
    (void)dbgSync;                                  // silence "unused" when CCONV_DBG_PRINT_SYNC=false

    // calcBuf layout (FP32, contiguous):
    //   [weightF | biasF | accF | tmpF]
    //   sizes:    MAX_WIDTH*MAX_BLOCK_DIM, MAX_BLOCK_DIM, MAX_BLOCK_DIM, MAX_BLOCK_DIM
    LocalTensor<float> calc    = calcBuf.Get<float>();               // base pointer to the FP32 scratch region
    LocalTensor<float> weightF = calc;                               // weights, FP32, indexed as weightF[j*MAX_BLOCK_DIM]
    LocalTensor<float> biasF   = weightF[MAX_WIDTH * MAX_BLOCK_DIM]; // bias slice sits right after the W weight slices
    LocalTensor<T>     tempT   = outBuf.Get<T>();                    // scratch in original-dtype: receives weight/bias from MTE2 before Cast

    // P1.A: surgical sync replaces all PIPE_ALL drains here. Two cross-pipe
    // deps on tempT that the kernel actually has:
    //   • MTE2 → V RAW  : V's Cast reads bytes MTE2 just wrote        →  SetFlag<MTE2_V>
    //   • V → MTE2 WAR  : MTE2 in iter j+1 (or bias load) overwrites
    //                     bytes V Cast of iter j is still reading      →  SetFlag<V_MTE2>
    //
    // Set/Wait counts (must balance per LoadW invocation):
    //   MTE2_V : W Sets + (hasBias ? 1 : 0)  /  same Waits
    //   V_MTE2 : (W-1) Sets between weight iters + (hasBias ? 1 : 0) for
    //            the bias entry  /  matching Waits.
    //
    // No leading fence: prev task's WriteBackState ends with a PIPE_ALL
    // (still upstream-style for now) → MTE3 drained on entry. No cross-task
    // tempT race (prev task didn't touch tempT/outBuf in WriteBackState).
    // No trailing fence: RunSeq reads weightF/biasF on V (same pipe, FIFO).
    for (int32_t j = 0; j < MAX_WIDTH; ++j) {
        const int64_t weightOffset = static_cast<int64_t>(j) * dim + c0;
        // V→MTE2 WAR: wait for prev iter's V Cast to release tempT.
        if (j > 0) {
            WaitFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        }
        DataCopy(tempT, weightGm[weightOffset], dimTileSize);        // MTE2: GM → tempT
        // MTE2→V RAW: gate V Cast on MTE2 write of tempT.
        SetFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
        Cast(weightF[j * MAX_BLOCK_DIM], tempT, RoundMode::CAST_NONE, dimTileSize); // V: tempT → weightF[j]
        // V→MTE2 signal — only when there is a future MTE2 consumer of tempT
        // (either the next weight iter, or the bias load).
        const bool more_weight_iters = (j + 1 < MAX_WIDTH);
        const bool bias_will_consume = (tilingData_->hasBias != 0);
        if (more_weight_iters || bias_will_consume) {
            SetFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        }
    }

    if (tilingData_->hasBias != 0) {
        // V→MTE2 WAR: wait for last weight Cast's tempT read to retire.
        WaitFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        DataCopy(tempT, biasGm[c0], dimTileSize);                    // MTE2: GM → tempT
        SetFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(tempMte2ToVEvent_);
        Cast(biasF, tempT, RoundMode::CAST_NONE, dimTileSize);       // V: tempT → biasF
    } else {
        Duplicate(biasF, 0.0f, dimTileSize);                         // V: biasF ← 0
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::InitRing(int32_t cacheIdx, bool hasInit, int32_t start, int32_t len, int32_t c0,
                                                 int32_t dimTileSize, int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;   // depth of the convStates cache (per cache line)
    LocalTensor<T> ring = inBuf.Get<T>();             // ring buffer in UB: RING_SLOTS * MAX_BLOCK_DIM lanes of T

    PipeBarrier<PIPE_ALL>();                          // fence prior LoadWeightAndBias work before we touch `ring`
    if (hasInit) {
        // Resuming a stream: pull the (W-1) historical tokens from the recurrent state cache
        // into slots [0, W-1). RunSeq will read these as taps t-1, t-2, t-3 of the first output.
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            // 3D offset into convStatesGm: [cacheIdx, i, c0:c0+dimTileSize]
            const int64_t stateOffset =                                       // GM byte offset of state row (cacheIdx, i)
                static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(i) * dim + c0;
            DataCopy(ring[i * MAX_BLOCK_DIM], convStatesGm[stateOffset], dimTileSize); // MTE2: GM → UB slot i
        }
    } else {
        // First call for this sequence: zero out the history so taps see implicit padding.
        for (int32_t i = 0; i < (MAX_WIDTH - 1); ++i) {
            Duplicate(ring[i * MAX_BLOCK_DIM], static_cast<T>(0), dimTileSize); // V: writes 0 into ring slot i
        }
    }
    PipeBarrier<PIPE_ALL>();                          // separate history fill from the first-token prefetch below

    if (len > 0) {
        // Load the first input token (xGm[start]) into the ring slot that RunSeq will
        // read at t=0. SlotCurr(0) maps token t=0 to its rotating-ring slot.
        const int64_t xOffset = static_cast<int64_t>(start) * dim + c0;            // GM byte offset of xGm[start]
        PipeBarrier<PIPE_ALL>();                                                   // fence prior history fill so MTE2 doesn't race with V Duplicate
        DataCopy(ring[SlotCurr(0) * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize);    // MTE2: GM → UB slot[SlotCurr(0)]
        PipeBarrier<PIPE_ALL>();                                                   // fence so RunSeq's first Cast sees the prefetched slot
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::RunSeq(int32_t start, int32_t len, int32_t c0, int32_t dimTileSize, int32_t dim,
                                               bool dbg)
{
    // ── Re-derive views into the shared UB buffers (same layout as LoadWeightAndBias) ──
    LocalTensor<float> calc    = calcBuf.Get<float>();                  // base pointer to the FP32 scratch region
    LocalTensor<float> weightF = calc;                                  // weights view — already filled by LoadWeightAndBias
    LocalTensor<float> biasF   = weightF[MAX_WIDTH * MAX_BLOCK_DIM];    // bias view — sits right after the W weight slices
    LocalTensor<float> accF    = biasF[MAX_BLOCK_DIM];                  // per-token FP32 accumulator (reset to biasF every iter)
    LocalTensor<float> tmpF    = accF[MAX_BLOCK_DIM];                   // per-tap FP32 scratch (also receives Silu(accF))
    LocalTensor<T>     ring    = inBuf.Get<T>();                        // ring of recent input tokens (already primed by InitRing)
    LocalTensor<T>     outT    = outBuf.Get<T>();                       // 2-slot output stage (double-buffered by outSlot)

    const bool dbgSync = dbg && CCONV_DBG_PRINT_SYNC;                    // gates per-iter sync prints (debug only)
    (void)dbgSync;                                                       // silence "unused" warning when DBG=false
    const bool hasActivation = (tilingData_->activationMode != 0);       // SiLU on/off, decided by the host
    const int32_t dbgMaxTokens     = CCONV_DBG_MAX_TOKENS;               // how many leading tokens get any debug dump
    const int32_t dbgVerboseTokens = CCONV_DBG_VERBOSE_TOKENS;           // of those, how many get the full per-tap trace

    // ── Per-token loop ────────────────────────────────────────────────────
    for (int32_t t = 0; t < len; ++t) {
        const bool dbgTok     = dbg && (t < dbgMaxTokens);                    // dump *any* tensors this iter?
        const bool dbgVerbose = dbg && CCONV_DBG_DUMP_RUNSEQ && (t < dbgVerboseTokens); // also dump accF/tmpF mid-conv?
        const bool dbgStep    = dbgVerbose && (t == 0);                       // extra-loud trace at t=0 only

        // Slot map (5-slot ring): SlotCurr(t)=t%5, SlotHist(t,k)=(t-k)%5.
        // For a width-4 causal conv at output position t we need taps
        // {t, t-1, t-2, t-3}, which land in slots {slotCurr, slotH1, slotH2, slotH3}.
        // The 5th slot is reserved for the *next* iter's prefetch (slotPref).
        const int32_t slotCurr = SlotCurr(t);                            // ring slot holding the current input token (tap 0)
        const int32_t slotH1   = SlotHist(t, 1);                         // dead local — kept for symmetry / readability (inner loop calls SlotHist directly)
        const int32_t slotH2   = SlotHist(t, 2);                         // dead local — same as above
        const int32_t slotH3   = SlotHist(t, 3);                         // dead local — same as above
        const int32_t slotPref = (t + 1 < len) ? SlotPrefetch(t) : -1;   // slot to prefetch into; -1 = no prefetch on the last iter
        const int32_t outSlot  = t & 1;                                  // alternates 0/1 across iters → outBuf double-buffer

        // P2.B: wait for prev iter's MTE2 prefetch of slotCurr (RAW). Iter 0 has
        // no prior set — InitRing's PIPE_ALL drains MTE2 before RunSeq starts.
        // P2.C: also wait for prev iter's V Cast (j=0, tap=3) to be done reading
        // slotPref(t), the slot this iter's MTE2 prefetch is about to overwrite.
        // slotHist(t-1, 3) == slotPref(t) always (both = (t-4)%5 == (t+1)%5 mod 5).
        if (t > 0) {
            WaitFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_);
            WaitFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
        }

        // ── Prefetch next input token while we compute this one ──────────
        if (t + 1 < len) {
            // Load xGm[start + t + 1] into the slot RunSeq will consume at iter t+1.
            const int64_t xOffset = static_cast<int64_t>(start + t + 1) * dim + c0; // GM byte offset of next token, this dim block
            // P2.C: dropped L314/L320 pre-prefetch PIPE_ALL — V→MTE2 WAR is now covered by the WaitFlag<V_MTE2> above.
            DataCopy(ring[slotPref * MAX_BLOCK_DIM], xGm[xOffset], dimTileSize); // MTE2: GM → UB slot[slotPref]
            // P2.B: signal MTE2 prefetch complete — next iter's WaitFlag<MTE2_V> pops this.
            SetFlag<HardEvent::MTE2_V>(inputMte2ToVEvent_);
        }

        // ── Init accumulator to bias ─────────────────────────────────────
        // Intra-UB DataCopy on the V pipe: accF ← biasF. With hasBias=false biasF is just zeros.
        DataCopy(accF, biasF, dimTileSize);                              // V: accF ← biasF (resets the accumulator each token)

        // ── Width-4 multiply-add: accF += weightF[j] * cast(ring[tap]) for each tap ──
        // Tap order is reversed (j=0 → oldest tap = (t-3), j=W-1 → current token = t).
        // P2.B: dropped per-tap PIPE_ALL (L328) — single WaitFlag<MTE2_V> at iter entry
        // covers MTE2→V sync for slotCurr (the slot prefetched at iter t-1). History
        // slots (taps 1..3) were prefetched many iters ago; since MTE2 is FIFO, waiting
        // on iter t-1's prefetch transitively drains every earlier MTE2 op on the ring.
        // For the very first iters before the prefetch queue is primed, InitRing's
        // pre-RunSeq PIPE_ALL drain handles visibility instead.
        for (int32_t j = 0; j < MAX_WIDTH; ++j) {
            const int32_t tap  = (MAX_WIDTH - 1) - j;                    // tap index counting back from t (3,2,1,0)
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap); // which ring slot this tap lives in
            Cast(tmpF, ring[slot * MAX_BLOCK_DIM], RoundMode::CAST_NONE, dimTileSize); // V: T → FP32 into tmpF
            // P2.C: after j=0 (tap=3 = slotPref(t+1)) Cast, signal that V is done
            // reading the slot next iter's MTE2 prefetch will overwrite. Place the
            // SetFlag as early as possible to maximize MTE2/V overlap.
            if (j == 0 && t + 1 < len) {
                SetFlag<HardEvent::V_MTE2>(tempVToMte2Event_);
            }
            MulAddDst(accF, tmpF, weightF[j * MAX_BLOCK_DIM], dimTileSize);          // V: accF += tmpF * weightF[j]  (in-place accumulate)
        }

        // Optional SiLU activation: tmpF = silu(accF). Separate dst tensor —
        // in-place Silu(accF, accF) corrupts FP32 on this NPU (we hit this in PR-3 testing).
        if (hasActivation) {
            Silu(tmpF, accF, dimTileSize);                                // V: tmpF ← silu(accF)
        }

        // P3.A: loop-carried MTE3→V WAR on outT[outSlot] (double-buffer, 2-iter distance).
        // outSlot alternates 0/1, so at iter t we may overwrite the same outT[outSlot]
        // that iter t-2's MTE3 was still reading. Wait until that prior MTE3 finished.
        // First reuse of outSlot=0 happens at iter 2; first reuse of outSlot=1 at iter 3.
        if (t >= 2) {
            WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        }

        // ── Pack result into outT (slot 0 or 1 depending on outSlot) ─────
        // FP32 path keeps DataCopy (intra-UB on V); narrower dtypes need a Cast back from FP32.
        // P2.A: dropped pre-pack PIPE_ALL — Silu/MulAddDst/DataCopy/Cast are all V-pipe ops, FIFO-ordered.
        if constexpr (IsSameType<T, float>::value) {
            // T == float: result is already FP32, just move it into the outT slot.
            if (hasActivation) {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], tmpF, dimTileSize);             // V: outT[outSlot] ← tmpF (post-Silu)
            } else {
                DataCopy(outT[outSlot * MAX_BLOCK_DIM], accF, dimTileSize);             // V: outT[outSlot] ← accF (raw conv)
            }
        } else {
            // T is fp16/bf16: Cast FP32 → T using round-to-nearest.
            if (hasActivation) {
                Cast(outT[outSlot * MAX_BLOCK_DIM], tmpF, RoundMode::CAST_RINT, dimTileSize); // V: outT[outSlot] ← (T)tmpF
            } else {
                Cast(outT[outSlot * MAX_BLOCK_DIM], accF, RoundMode::CAST_RINT, dimTileSize); // V: outT[outSlot] ← (T)accF
            }
        }
        // P3.A: V→MTE3 RAW — signal MTE3 may now read outT[outSlot]. Replaces L376 PIPE_ALL.
        SetFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);

        // ── Store this token's output to yGm ─────────────────────────────
        const int64_t outOffset = static_cast<int64_t>(start + t) * dim + c0; // GM byte offset for yGm[start+t]
        // P3.A: paired wait for the V→MTE3 set above; stalls MTE3 only (not all pipes).
        WaitFlag<HardEvent::V_MTE3>(outVToMte3Event_[outSlot]);
        DataCopy(yGm[outOffset], outT[outSlot * MAX_BLOCK_DIM], dimTileSize);  // MTE3: UB outT slot → GM yGm
        // P3.A: MTE3→V WAR — signal MTE3 done reading outT[outSlot]. Iter t+2's pack waits on this.
        SetFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[outSlot]);
        // P2.C: dropped L366/L372 post-yGm PIPE_ALL — V→MTE2 WAR on ring is now covered by
        // SetFlag<V_MTE2>/WaitFlag pair; outT WAR is now handled by the MTE3_V event above.
        // MTE2 and MTE3 are independent pipes.
    }

    // P3.A: drain leftover MTE3_V sets that no in-loop wait consumed (the last 1-2 iters
    // posted but the loop ended). One per outSlot that actually fired at least once.
    if (len >= 1) {
        WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
    }
    if (len >= 2) {
        WaitFlag<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::WriteBackState(int32_t cacheIdx, int32_t len, int32_t c0, int32_t dimTileSize,
                                                       int32_t dim, bool dbg)
{
    const int32_t stateLen = tilingData_->stateLen;   // depth of each cache row in convStatesGm
    if (len <= 0) {
        return;                                       // no tokens processed → nothing to persist
    }
    const int32_t lastT = len - 1;                    // last output position; the (W-1) taps ending here
                                                      // are what the next call must resume from.
    LocalTensor<T> ring = inBuf.Get<T>();             // re-view the ring (RunSeq left it populated, no need to reload)

    // Walk pos = 0..W-2 and persist the corresponding ring slot back to convStatesGm.
    // tap = (W-2) - pos maps "newest first" so pos=0 stores the most-recent history slot.
    for (int32_t pos = 0; pos < (MAX_WIDTH - 1); ++pos) {
        const int32_t tap  = (MAX_WIDTH - 2) - pos;                                   // tap index counting back from lastT
        const int32_t slot = (tap == 0) ? SlotCurr(lastT) : SlotHist(lastT, tap);     // which ring slot the tap currently lives in
        const int64_t stateOffset =                                                   // GM byte offset: [cacheIdx, pos, c0:c0+dimTileSize]
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(pos) * dim + c0;
        PipeBarrier<PIPE_ALL>();                                                       // gate so MTE3 sees the final ring contents from RunSeq
        DataCopy(convStatesGm[stateOffset], ring[slot * MAX_BLOCK_DIM], dimTileSize); // MTE3: UB ring slot → GM convStates row
        PipeBarrier<PIPE_ALL>();                                                       // gate so next iter's MTE3 doesn't race with this one (FIFO-redundant in practice)
    }
}

template <typename T>
__aicore__ inline void CausalConv1d<T>::Process()
{
    // Pull tiling constants out of the host-computed struct once per launch.
    const int32_t dim          = tilingData_->dim;          // total channels
    const int32_t batch        = tilingData_->batch;        // number of sequences
    const int32_t inputMode    = tilingData_->inputMode;    // 0 = varlen (queryStartLoc), 1 = dense (fixed seqLen)
    const int32_t seqLen       = tilingData_->seqLen;       // tokens-per-seq in dense mode
    const int32_t dimTileSize  = static_cast<int32_t>(tilingData_->dimTileSize);   // channels per dim block
    const int32_t blocksPerSeq = static_cast<int32_t>(tilingData_->blocksPerSeq);  // dim blocks per sequence

    // Per-core ID + total cores: this AI core processes tasks `blockIdx`, `blockIdx+blockNum`, ...
    const uint32_t blockIdx = GetBlockIdx();           // 0..blockNum-1, identifies this AI core
    const uint32_t blockNum = GetBlockNum();           // total AI cores cooperating on this launch (40 on 910B4)

    // Defensive: if tiling was rejected by the host, return immediately.
    // ReleaseEvents() balances the AllocEvents() from Init().
    if (dimTileSize <= 0 || blocksPerSeq <= 0 || dimTileSize > MAX_BLOCK_DIM || blocksPerSeq * dimTileSize != dim) {
        ReleaseEvents();                               // balance AllocEvents() — release reserved slots before bailing
        return;                                        // give up: nothing valid to compute
    }

    // Work grid: gridSize = batch * blocksPerSeq tasks total. Each task =
    // one (sequence, dim_block) tile. Stride by blockNum so AI cores split
    // the tasks round-robin.
    const int64_t gridSize = static_cast<int64_t>(batch) * blocksPerSeq;        // total number of tasks across the launch
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq        = static_cast<int32_t>(task / blocksPerSeq);   // which sequence index
        const int32_t dimBlockId = static_cast<int32_t>(task % blocksPerSeq);   // which dim block within that sequence
        const int32_t c0         = dimBlockId * dimTileSize;                    // starting channel of this dim block
        const bool dbg = (seq == CCONV_DBG_SEQ) && (c0 == CCONV_DBG_C0);        // debug gate (compile-time pruned to false in release)

        // Step 1: load this dim block's weights + bias into UB (FP32).
        LoadWeightAndBias(c0, dimTileSize, dbg);                                // sets weightF[*] and biasF in calcBuf

        // Step 2: resolve this sequence's [start, start+len) range in xGm/yGm.
        int32_t start = 0;                             // token offset into xGm/yGm
        int32_t len   = 0;                             // tokens to process for this sequence
        if (inputMode == 0) {
            // varlen mode: ranges come from queryStartLocGm[seq] .. queryStartLocGm[seq+1].
            const int32_t startVal = queryStartLocGm.GetValue(seq);             // cumulative start (scalar read from GM)
            const int32_t endVal   = queryStartLocGm.GetValue(seq + 1);         // cumulative end
            start = startVal;                          // begin at this token
            len   = endVal - startVal;                 // number of tokens in this sequence
        } else {
            // dense mode: every sequence is exactly seqLen tokens.
            start = seq * seqLen;                      // strided offset: each seq starts at seq*seqLen
            len   = seqLen;                            // and is exactly seqLen tokens
        }

        if (len <= 0) {
            continue;                                                            // empty sequence — nothing to do
        }

        // Step 3: pick the recurrent-state cache row. padSlotId = "skip this".
        const int32_t cacheIdx = cacheIndicesGm.GetValue(seq);                   // which row of convStatesGm to load/store
        if (cacheIdx == tilingData_->padSlotId) {
            continue;                                  // sentinel: caller wants this sequence skipped entirely
        }

        // Step 4: per-sequence pipeline → init ring → run conv → write back state.
        const bool hasInit = hasInitialStateGm.GetValue(seq);                    // load prior history (true) or zero-init (false)
        InitRing(cacheIdx, hasInit, start, len, c0, dimTileSize, dim, dbg);      // populate ring with W-1 history + first token
        RunSeq(start, len, c0, dimTileSize, dim, dbg);                           // per-token conv → yGm
        WriteBackState(cacheIdx, len, c0, dimTileSize, dim, dbg);                // persist last W-1 tokens back to convStatesGm
    }

    // Release every event ID we reserved in Init().
    ReleaseEvents();                                                             // returns event slots to the hardware table
}

}  // namespace NsCausalConv1d

#endif  // CAUSAL_CONV1D_H
