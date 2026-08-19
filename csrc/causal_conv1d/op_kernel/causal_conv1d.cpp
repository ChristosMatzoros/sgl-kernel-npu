// causal_conv1d_hybrid.cpp -- depthwise causal conv1d + bias + (opt) SiLU.
// K-tap contraction on the Cube (AIC); bias/SiLU/cast on the Vector unit (AIV).
//
//   y[t,c] = act( bias[c] + sum_{j=0..K-1} w[j,c] * xext[t+j,c] )
//   xext = [history(K-1), x]  ->  w[K-1] multiplies the newest token.
//
// DEPTHWISE AS A MATMUL, per 16-channel group: stacking the K taps along the contraction
// axis collapses the tap sum into ONE matmul of inner dim 16K.
//   A[m, kw*16+c] = xext[t0+m-halo+kw, c]   one TIMG2COL emits all K shifts
//   B[kw*16+c, n] = w[kw,c] * delta(c,n)    block-diagonal, built once, L2-resident
// The block diagonal wastes 16x of the MACs, measured free (deleting every TIMG2COL and
// TMATMUL moves duration 0.2%/0.0%); the AIV is ~92% of the critical path.
//
// HANDOFF VIA GM: a2a3 has no L0C->UB or L1->UB path, so Cube results reach the AIV only
// through memory. A 3-slot ring keeps that L2-resident (64 KiB/slot, 4.5 MiB, 2.3% of L2,
// 99.97% hit), so HBM stays at the 2-pass floor: x read once, y written once.
//
// The Cube instructions set this library's task type to MIX: blockDim counts AIC blocks,
// each pairing one Cube with two Vector sub-blocks.
//
// ASCII only.

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#if defined(__NPU_ARCH__)
#include <pto/common/buffer_limits.hpp>
#include "kernel_operator.h"
#endif

#define PIPE_BARRIER_VEC() AscendC::PipeBarrier<PIPE_V>()

// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format on

using namespace pto;

namespace cc1dhy {

// ---------------------------------------------------------------------------
// Cross-core (AIC <-> its 2 AIVs) FFTS flag ids. SYNCALL reserves 11-14 of the 16
// available, so the ring takes 0..5 and the barriered handshake 8/9.
// ---------------------------------------------------------------------------
constexpr uint32_t NSLOT = 3u;                // staging-ring depth (see L2 arithmetic)
constexpr uint16_t FLAG_RDY[3] = {0, 1, 2};   // AIC -> AIV : "slot s holds data"
constexpr uint16_t FLAG_FREE[3] = {3, 4, 5};  // AIV -> AIC : "slot s is consumed"
constexpr uint16_t FLAG_BAR_V2C = 8;          // barriered mode: AIV -> AIC
constexpr uint16_t FLAG_BAR_C2V = 9;          // barriered mode: AIC -> AIV
constexpr uint16_t FFTS_CV_MODE = 0x2;        // cube <-> vector PAIR sync (not all-core)

// A block-local (1 AIC + its 2 AIV) rendezvous: the same shape as SYNCALL<Mix> minus the
// all-AIC mode-0 barrier. That difference matters: different blocks run different numbers
// of tasks, so a GLOBAL barrier inside the task loop would deadlock. This one is local.
AICORE inline void pairBarrier()
{
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    wait_flag_dev(FLAG_BAR_V2C);
    ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_CV_MODE, FLAG_BAR_C2V));
#elif defined(__DAV_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_CV_MODE, FLAG_BAR_V2C));
    wait_flag_dev(FLAG_BAR_C2V);
#endif
}

// ---------------------------------------------------------------------------
// Shared task enumeration. Both engines MUST walk the same chunk sequence or the flag
// budget goes asymmetric and the device hangs -- so everything here is derived from
// kernel arguments and GM scalars only.
// ---------------------------------------------------------------------------
struct TaskGeom {
    int32_t seqStart;          // first token of this sequence in x
    int32_t tokenLo, tokenHi;  // token range [tokenLo,tokenHi) of this chunk, relative to `seqStart`
    int32_t cacheIdx;
    uint32_t chanBase;
    int32_t validChans;
    bool hasInitState;
    bool valid;
};

AICORE inline TaskGeom resolveTask(__gm__ int32_t *seqStartLoc, __gm__ int32_t *cacheIdx, __gm__ uint8_t *hasInitState,
                                   uint32_t task, uint32_t dim, uint32_t inputMode, uint32_t seqLen,
                                   uint32_t chanTileWidth, uint32_t blocksPerSeq, uint32_t numTokenChunks,
                                   int32_t padSlot, uint32_t haloTokens)
{
    TaskGeom taskGeom;
    taskGeom.valid = false;
    taskGeom.seqStart = 0;
    taskGeom.tokenLo = 0;
    taskGeom.tokenHi = 0;
    taskGeom.cacheIdx = 0;
    taskGeom.chanBase = 0;
    taskGeom.validChans = 0;
    taskGeom.hasInitState = false;

    // TASK ORDER: B x T x C -- the CHANNEL tile varies fastest. Concurrent blocks therefore
    // cover a contiguous span of channels at the same token range, and since `dim` is the
    // contiguous axis of x that is a contiguous span of memory. Cube and Vector both derive
    // their geometry here, so they cannot disagree about which tile owns which ring slot.
    const uint32_t chanTileIdx = task % blocksPerSeq;
    const uint32_t tokenChunkIdx = (task / blocksPerSeq) % numTokenChunks;
    const uint32_t seq = (task / blocksPerSeq) / numTokenChunks;

    int32_t seqStart, seqTokens;
    if (inputMode == 0u) {
        seqStart = seqStartLoc[seq];
        seqTokens = seqStartLoc[seq + 1] - seqStart;
    } else {
        seqStart = (int32_t)(seq * seqLen);
        seqTokens = (int32_t)seqLen;
    }
    if (seqTokens == 0) return taskGeom;
    const int32_t seqCacheIdx = cacheIdx[seq];
    if (seqCacheIdx == padSlot) return taskGeom;

    uint32_t tokensPerChunk = ((uint32_t)seqTokens + numTokenChunks - 1u) / numTokenChunks;
    // Clamp so only chunk 0 of a (seq, channel tile) reads convStates history. The folded
    // writeback relies on exactly one reader: it hands the writeback to that task's owning
    // block, which orders it with no barrier. Coverage holds since
    // numTokenChunks*tokensPerChunk >= seqTokens.
    if (tokensPerChunk < haloTokens) tokensPerChunk = haloTokens;
    if (tokensPerChunk < 1u) tokensPerChunk = 1u;
    const int32_t tokenLo = (int32_t)(tokenChunkIdx * tokensPerChunk);
    if (tokenLo >= seqTokens) return taskGeom;
    int32_t tokenHi = tokenLo + (int32_t)tokensPerChunk;
    if (tokenHi > seqTokens) tokenHi = seqTokens;

    const uint32_t chanBase = chanTileIdx * chanTileWidth;
    const uint32_t chansRemaining = dim - chanBase;
    taskGeom.seqStart = seqStart;
    taskGeom.tokenLo = tokenLo;
    taskGeom.tokenHi = tokenHi;
    taskGeom.cacheIdx = seqCacheIdx;
    taskGeom.chanBase = chanBase;
    taskGeom.validChans = (int32_t)(chansRemaining > chanTileWidth ? chanTileWidth : chansRemaining);
    taskGeom.hasInitState = hasInitState[seq] != 0;
    taskGeom.valid = true;
    return taskGeom;
}

// AIV prologue: weight (K,dim) -> block-diagonal Cube B operand, stored transposed as a
// per-group row-major [16, 16K] (the Layout::DN view, so TEXTRACT->L0B does not transpose).
// TAP ORDER: k-block kw carries tap kw, NOT K-1-kw. Reversing it gives plausible magnitudes
// with ~zero correlation and was the original numerical bug.
template <typename IoT, uint32_t RS>
AICORE void expandWeights(__gm__ IoT *weight, __gm__ IoT *wexp, uint32_t dim, uint32_t K)
{
    constexpr uint32_t KMAX = 16u * RS;
    using Nd2 = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Nd2S = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GIo = pto::GlobalTensor<IoT, Nd2, Nd2S, pto::Layout::ND>;

    using EyeT = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using RowIoT = Tile<TileType::Vec, IoT, 1, 16, BLayout::RowMajor, 1, 16>;
    using RowF32 = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16>;
    // Each tap's diagonal is built as a COMPACT [16,16] block (Cols == ValidCol) so
    // TCOLEXPANDMUL takes its norm-mode path, and is written into columns
    // [kw*16, kw*16+16) of the group's [16, 16K] GM matrix with one strided TSTORE.
    using DiagF32 = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using DiagIoT = Tile<TileType::Vec, IoT, 16, 16, BLayout::RowMajor, 16, 16>;

    // UB map for the expansion region (base 0; the main AIV region starts after it).
    constexpr uint32_t OFF_EYE = 0u;                            // 16*16 fp32   1 KiB
    constexpr uint32_t OFF_WIO = OFF_EYE + 16u * 16u * 4u;      // RS*16  IoT
    constexpr uint32_t OFF_WF = OFF_WIO + RS * 16u * 2u;        // RS*16  fp32
    constexpr uint32_t OFF_DF = OFF_WF + RS * 16u * 4u;         // RS * [16,16] fp32
    constexpr uint32_t OFF_DIO = OFF_DF + RS * 16u * 16u * 4u;  // RS * [16,16] IoT
    constexpr uint32_t EXP_END = OFF_DIO + RS * 16u * 16u * 2u;
    static_assert(EXP_END <= 192u * 1024u, "conv1d hybrid: weight-expansion UB overflow");

    EyeT eye;
    TASSIGN(eye, OFF_EYE);
    for (uint32_t r = 0; r < 16u; ++r) {
        for (uint32_t c = 0; c < 16u; ++c) {
            eye.SetValue(r * 16u + c, (r == c) ? 1.0f : 0.0f);
        }
    }
    pipe_barrier(PIPE_ALL);  // scalar (PIPE_S) stores -> vector reads

    // One logical AIV per (block, subblock); stride the 16-channel groups across them.
    const uint32_t nAiv = get_block_num() * get_subblockdim();
    const uint32_t aivId = get_block_idx() + get_subblockid() * get_block_num();
    const uint32_t numChanGroups = dim / 16u;
    const uint32_t kExtent = 16u * K;

    for (uint32_t grp = aivId; grp < numChanGroups; grp += nAiv) {
        // (1) MTE2: stage the K tap rows for this group IN ORDER (tap kw -> k-block kw);
        //     see the TAP ORDER note above -- this must NOT be reversed.
        for (uint32_t kw = 0; kw < K; ++kw) {
            const uint32_t tap = kw;
            GIo wG(weight + (uint64_t)tap * dim + grp * 16u, Nd2(1, 16), Nd2S(16));
            RowIoT tapRowIo;
            TASSIGN(tapRowIo, OFF_WIO + kw * 16u * 2u);
            TLOAD(tapRowIo, wG);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // (2) V: cast to fp32, expand each tap into a diagonal block, cast back.
        for (uint32_t kw = 0; kw < K; ++kw) {
            RowIoT tapRowIo;
            RowF32 tapRowF32;
            TASSIGN(tapRowIo, OFF_WIO + kw * 16u * 2u);
            TASSIGN(tapRowF32, OFF_WF + kw * 16u * 4u);
            TCVT(tapRowF32, tapRowIo, pto::RoundMode::CAST_NONE);
        }
        PIPE_BARRIER_VEC();
        for (uint32_t kw = 0; kw < K; ++kw) {
            RowF32 tapRowF32;
            DiagF32 diagF32;
            TASSIGN(tapRowF32, OFF_WF + kw * 16u * 4u);
            TASSIGN(diagF32, OFF_DF + kw * 16u * 16u * 4u);
            TCOLEXPANDMUL(diagF32, eye, tapRowF32);
        }
        PIPE_BARRIER_VEC();
        for (uint32_t kw = 0; kw < K; ++kw) {
            DiagF32 diagF32;
            DiagIoT diagIo;
            TASSIGN(diagF32, OFF_DF + kw * 16u * 16u * 4u);
            TASSIGN(diagIo, OFF_DIO + kw * 16u * 16u * 2u);
            TCVT(diagIo, diagF32, pto::RoundMode::CAST_NONE);
        }
        PIPE_BARRIER_VEC();
        // (3) MTE3: one strided store per tap into the group's [16, 16K] operand.
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        for (uint32_t kw = 0; kw < K; ++kw) {
            GIo oG(wexp + (uint64_t)grp * 16u * kExtent + kw * 16u, Nd2(16, 16), Nd2S(kExtent));
            DiagIoT diagOut;
            TASSIGN(diagOut, OFF_DIO + kw * 16u * 16u * 2u);
            TSTORE(oG, diagOut);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);  // UB free for the next group
    }
    // Drain the eye region's last handshake is unnecessary (all pairs above are balanced).
}

// ===========================================================================
// AIC: conv on the Cube. Per chunk: x -> L1 (NZ) -> TIMG2COL -> L0A;  wexp -> L1 (ZN) ->
// TEXTRACT -> L0B;  TMATMUL -> L0C;  fixpipe L0C -> GM staging slot (fp32 -> IoT cast).
// ===========================================================================
template <typename IoT, uint32_t RS, uint32_t T_TILE, uint32_t D_TILE>
AICORE void runCube(__gm__ IoT *x, __gm__ IoT *convStates, __gm__ IoT *wexp, __gm__ IoT *ring,
                    __gm__ int32_t *seqStartLoc, __gm__ int32_t *cacheIdx, __gm__ uint8_t *hasInitState, uint32_t dim,
                    uint32_t stateLen, uint32_t inputMode, uint32_t seqLen, uint32_t K, uint32_t chanTileWidth,
                    uint32_t blocksPerSeq, uint32_t numTokenChunks, uint32_t gridSize, int32_t padSlot,
                    uint32_t pipelined)
{
    constexpr uint32_t KMAX = 16u * RS;
    constexpr uint32_t HALO_PAD = (RS <= 16u) ? 16u : RS;
    constexpr uint32_t L1ROWS = T_TILE + HALO_PAD;
    constexpr uint32_t NGRP = D_TILE / 16u;

    // L1 map
    constexpr uint32_t L1_X = 0u;
    constexpr uint32_t L1_X_BYTES = L1ROWS * D_TILE * sizeof(IoT);
    constexpr uint32_t L1_B = L1_X + L1_X_BYTES;
    constexpr uint32_t L1_B_STRIDE = KMAX * 16u * sizeof(IoT);
    static_assert(L1_B + NGRP * L1_B_STRIDE <= 512u * 1024u, "conv1d hybrid: L1 overflow");
    static_assert(T_TILE * KMAX * sizeof(IoT) <= 64u * 1024u, "conv1d hybrid: L0A overflow");
    static_assert(KMAX * 16u * sizeof(IoT) <= 64u * 1024u, "conv1d hybrid: L0B overflow");
    static_assert(T_TILE * D_TILE * sizeof(float) <= 128u * 1024u, "conv1d hybrid: L0C overflow");

    using Nd2 = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Nd2S = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GIo = pto::GlobalTensor<IoT, Nd2, Nd2S, pto::Layout::ND>;
    // DN view of the logical [16K,16] B operand: GM holds it as a [16, 16K] row-major
    // matrix, so dim3 = 16K rows of the logical operand, dim4 = 16 cols, stride4 = 16K.
    using Dn2 = pto::Shape<1, 1, 1, DYNAMIC, 16>;
    using Dn2S = pto::Stride<1, 1, 1, 1, DYNAMIC>;
    using GDn = pto::GlobalTensor<IoT, Dn2, Dn2S, pto::Layout::DN>;

    // L1 x tile: ND(GM) -> NZ(L1). The fractal row stride is the STATIC Rows, so a second
    // tile assigned at base + rowOffset*16*sizeof(IoT) writes rows [rowOffset, ...) of
    // every 16-channel block; that is how conv_states history rows are prepended.
    using XMat = Tile<TileType::Mat, IoT, L1ROWS, D_TILE, BLayout::ColMajor, DYNAMIC, DYNAMIC, SLayout::RowMajor>;
    // The same L1 bytes viewed as a 1-channel-block (C1=1) feature map for load3d.
    using XConv = ConvTile<TileType::Mat, IoT, (int)(L1ROWS * 16u * sizeof(IoT)), pto::Layout::NC1HWC0,
                           pto::ConvTileShape<1, 1, 1, DYNAMIC, 16>>;
    using BMat = Tile<TileType::Mat, IoT, KMAX, 16, BLayout::RowMajor, DYNAMIC, 16, SLayout::ColMajor>;
    using ATile = TileLeft<IoT, T_TILE, KMAX, DYNAMIC, DYNAMIC>;
    using BTile = TileRight<IoT, KMAX, 16, DYNAMIC, 16>;
    using CGrp = TileAcc<float, T_TILE, 16, DYNAMIC, 16>;
    using CAll = TileAcc<float, T_TILE, D_TILE, DYNAMIC, DYNAMIC>;

    const int32_t haloTokens = (int32_t)K - 1;
    const uint32_t kExtent = 16u * K;
    const uint32_t numBlocks = get_block_num();
    const uint32_t blockId = get_block_idx();
    uint32_t chunkCount = 0u;

    for (uint32_t task = blockId; task < gridSize; task += numBlocks) {
        TaskGeom taskGeom = resolveTask(seqStartLoc, cacheIdx, hasInitState, task, dim, inputMode, seqLen,
                                        chanTileWidth, blocksPerSeq, numTokenChunks, padSlot, (uint32_t)haloTokens);
        if (!taskGeom.valid) continue;
        const uint32_t numChanGroups = ((uint32_t)taskGeom.validChans + 15u) / 16u;

        // WAR on L1: the previous task's MTE1 still reads L1_X/L1_B while (A) below refills
        // L1_B on MTE2. Drains only the AIC's own pipes; the cross-core ring overlap is
        // untouched.
        if (pipelined) pipe_barrier(PIPE_ALL);

        // (A) B operand for this whole channel tile: loaded once, reused by every chunk.
        for (uint32_t grp = 0; grp < numChanGroups; ++grp) {
            BMat bL1(kExtent);
            TASSIGN(bL1, L1_B + grp * L1_B_STRIDE);
            GDn bG(wexp + (uint64_t)(taskGeom.chanBase / 16u + grp) * 16u * kExtent, Dn2(kExtent), Dn2S(kExtent));
            TLOAD(bL1, bG);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);

        for (int32_t tokenBase = taskGeom.tokenLo; tokenBase < taskGeom.tokenHi; tokenBase += (int32_t)T_TILE) {
            int32_t tileTokens = (int32_t)T_TILE;
            if (tokenBase + tileTokens > taskGeom.tokenHi) tileTokens = taskGeom.tokenHi - tokenBase;
            const uint32_t slot = chunkCount % NSLOT;
            ++chunkCount;

            // ---- wait until the staging slot is free ----
            if (pipelined) {
                wait_flag_dev(FLAG_FREE[slot]);
                // WAR on L1_X: the previous chunk's MTE1 reads the bytes (B) refills on
                // MTE2, and wait_flag_dev gates only the cross-core credit, not local
                // pipes. L1_X is single-buffered here, so this must be a drain.
                pipe_barrier(PIPE_ALL);
            } else {
                pairBarrier();
            }

            // ---- (B) MTE2: xext rows [tokenBase-haloTokens, tokenBase+tileTokens) into L1 ----
            const int32_t numHistRows = (haloTokens > tokenBase) ? (haloTokens - tokenBase) : 0;  // token index < 0
            const int32_t numRealRows = haloTokens + tileTokens - numHistRows;
            uint16_t fmapW;
            uint8_t padLeft;
            if (numHistRows > 0 && !taskGeom.hasInitState) {
                // zero history: load only the real rows and let load3d left-pad with 0.
                XMat xL1((uint32_t)numRealRows, (uint32_t)taskGeom.validChans);
                TASSIGN(xL1, L1_X);
                GIo xG(x + (uint64_t)taskGeom.seqStart * dim + taskGeom.chanBase, Nd2(numRealRows, taskGeom.validChans),
                       Nd2S(dim));
                TLOAD(xL1, xG);
                fmapW = (uint16_t)numRealRows;
                padLeft = (uint8_t)numHistRows;
            } else {
                if (numHistRows > 0) {
                    // history rows convStates[cacheIdx, tokenBase .. haloTokens) -> L1 rows [0, numHistRows)
                    XMat histL1((uint32_t)numHistRows, (uint32_t)taskGeom.validChans);
                    TASSIGN(histL1, L1_X);
                    GIo hG(convStates + ((uint64_t)taskGeom.cacheIdx * stateLen + (uint32_t)tokenBase) * dim +
                               taskGeom.chanBase,
                           Nd2(numHistRows, taskGeom.validChans), Nd2S(dim));
                    TLOAD(histL1, hG);
                }
                XMat xL1((uint32_t)numRealRows, (uint32_t)taskGeom.validChans);
                TASSIGN(xL1, L1_X + (uint32_t)numHistRows * 16u * sizeof(IoT));
                const int32_t firstTok = (tokenBase - haloTokens) + numHistRows;  // == max(0, tokenBase-haloTokens)
                GIo xG(x + (uint64_t)(taskGeom.seqStart + firstTok) * dim + taskGeom.chanBase,
                       Nd2(numRealRows, taskGeom.validChans), Nd2S(dim));
                TLOAD(xL1, xG);
                fmapW = (uint16_t)(haloTokens + tileTokens);
                padLeft = 0u;
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            // ---- (C) per 16-channel group: one img2col + one matmul into its L0C block
            {
                XConv fmap((int64_t)fmapW);
                fmap.SetFmapH(1);
                fmap.SetFmapW(fmapW);
                fmap.SetChannelSize(16);
                fmap.SetFilterH(1);
                fmap.SetFilterW((uint16_t)K);
                fmap.SetStrideH(1);
                fmap.SetStrideW(1);
                fmap.SetDilationH(1);
                fmap.SetDilationW(1);
                fmap.SetPadList(0, padLeft);
                fmap.SetPadList(1, 0);
                fmap.SetPadList(2, 0);
                fmap.SetPadList(3, 0);
                TASSIGN(fmap, L1_X);
                SETFMATRIX(fmap);
                set_padding(0);  // pad value 0 for every 2-byte IO dtype

                for (uint32_t grp = 0; grp < numChanGroups; ++grp) {
                    TASSIGN(fmap, L1_X + grp * L1ROWS * 16u * sizeof(IoT));
                    ATile a((uint32_t)tileTokens, kExtent);
                    BTile b(kExtent);
                    CGrp c((uint32_t)tileTokens);
                    BMat bL1(kExtent);
                    TASSIGN(a, 0u);
                    TASSIGN(b, 0u);
                    TASSIGN(c, grp * T_TILE * 16u * sizeof(float));
                    TASSIGN(bL1, L1_B + grp * L1_B_STRIDE);

                    TIMG2COL(a, fmap, 0, 0);
                    TEXTRACT(b, bL1, 0, 0);
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                    TMATMUL(c, a, b);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                }
            }

            // ---- (D) FIX: one store of the whole [tileTokens, validChans] tile into the slot ----
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
            {
                CAll cAll((uint32_t)tileTokens, (uint32_t)taskGeom.validChans);
                TASSIGN(cAll, 0u);
                GIo rG(ring + (uint64_t)(blockId * NSLOT + slot) * T_TILE * D_TILE,
                       Nd2(tileTokens, taskGeom.validChans), Nd2S(D_TILE));
                TSTORE(rG, cAll);
            }
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);

            // ---- signal "slot ready" ----
            if (pipelined) {
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_CV_MODE, FLAG_RDY[slot]));
            } else {
                pairBarrier();  // release the AIV to consume this slot
                pairBarrier();  // and wait until it has
            }
        }
    }

    // Constant-length, per-slot-balanced drain: each AIV signalled FLAG_FREE[s] once
    // before its loop, so exactly NSLOT credits are outstanding whatever the trip count.
    // (Barriered mode issues no unpaired flags at all.)
    if (pipelined) {
        for (uint32_t s = 0; s < NSLOT; ++s) wait_flag_dev(FLAG_FREE[s]);
    }
}

// AIC v2 -- the shipping Cube half, ring 2/4/8/16 (width <= 16). Wider filters go to
// runCube above, whose L0B fits where v2's NGRP tiles would not. Pipeline is kept full by:
// all NGRP B tiles resident in L0B (one TEXTRACT per task), ABUF L0A tiles, L1_X double
// buffer + prefetch, L1_B ping-pong. fmapW is constant, so SETFMATRIX fires only on padLeft.
constexpr uint32_t HY_CFG_V2 = 1u;

// Launch-floor bits (cfg bits 4..7), ~34 us of per-launch fixed cost:
//   NOEXP  skip the AIV prologue (`wexp` is a cached model constant)
//   NOSYNC no prologue -> nothing for its barrier to order
//   NOBODY prologue only: the one-off `wexp` fill launch
//   WB     fold the conv_states writeback in, deleting a second launch
constexpr uint32_t HY_CFG_NOEXP = 16u;
constexpr uint32_t HY_CFG_NOSYNC = 32u;
constexpr uint32_t HY_CFG_NOBODY = 64u;
constexpr uint32_t HY_CFG_WB = 128u;
//   WB2     the folded writeback WITHOUT the SYNCALL<Mix> whole-chip barrier: each
//           writeback unit is executed by the block that owns the one conv task which
//           reads the rows it overwrites, so the ordering is already implied by the
//           ring handshake (see the OWNER MODE note on runWbVec).
constexpr uint32_t HY_CFG_WB2 = 256u;

#if defined(__DAV_CUBE__)
// A runtime-selected event id has to expand to a compile-time EVENT_ID constant;
// every index below is 0 or 1, so a two-way if/else covers it.
#define HYV2_EV2(FN, SP, DP, I, E0, E1) \
    do {                                \
        if ((I) == 0u) {                \
            FN(SP, DP, E0);             \
        } else {                        \
            FN(SP, DP, E1);             \
        }                               \
    } while (0)

#define A_RDY_SET(s) HYV2_EV2(set_flag, PIPE_MTE1, PIPE_M, s, EVENT_ID0, EVENT_ID1)
#define A_RDY_WAIT(s) HYV2_EV2(wait_flag, PIPE_MTE1, PIPE_M, s, EVENT_ID0, EVENT_ID1)
#define A_FREE_SET(s) HYV2_EV2(set_flag, PIPE_M, PIPE_MTE1, s, EVENT_ID0, EVENT_ID1)
#define A_FREE_WAIT(s) HYV2_EV2(wait_flag, PIPE_M, PIPE_MTE1, s, EVENT_ID0, EVENT_ID1)
#define X_RDY_SET(p) HYV2_EV2(set_flag, PIPE_MTE2, PIPE_MTE1, p, EVENT_ID0, EVENT_ID1)
#define X_RDY_WAIT(p) HYV2_EV2(wait_flag, PIPE_MTE2, PIPE_MTE1, p, EVENT_ID0, EVENT_ID1)
#define X_FREE_SET(p) HYV2_EV2(set_flag, PIPE_MTE1, PIPE_MTE2, p, EVENT_ID0, EVENT_ID1)
#define X_FREE_WAIT(p) HYV2_EV2(wait_flag, PIPE_MTE1, PIPE_MTE2, p, EVENT_ID0, EVENT_ID1)
#define B_RDY_SET(p) HYV2_EV2(set_flag, PIPE_MTE2, PIPE_MTE1, p, EVENT_ID2, EVENT_ID3)
#define B_RDY_WAIT(p) HYV2_EV2(wait_flag, PIPE_MTE2, PIPE_MTE1, p, EVENT_ID2, EVENT_ID3)
#define B_FREE_SET(p) HYV2_EV2(set_flag, PIPE_MTE1, PIPE_MTE2, p, EVENT_ID2, EVENT_ID3)
#define B_FREE_WAIT(p) HYV2_EV2(wait_flag, PIPE_MTE1, PIPE_MTE2, p, EVENT_ID2, EVENT_ID3)
#define C_RDY_SET(p) HYV2_EV2(set_flag, PIPE_M, PIPE_FIX, p, EVENT_ID0, EVENT_ID1)
#define C_RDY_WAIT(p) HYV2_EV2(wait_flag, PIPE_M, PIPE_FIX, p, EVENT_ID0, EVENT_ID1)
#define C_FREE_SET(p) HYV2_EV2(set_flag, PIPE_FIX, PIPE_M, p, EVENT_ID0, EVENT_ID1)
#define C_FREE_WAIT(p) HYV2_EV2(wait_flag, PIPE_FIX, PIPE_M, p, EVENT_ID0, EVENT_ID1)

template <typename IoT, uint32_t RS, uint32_t T_TILE, uint32_t D_TILE>
AICORE void runCubeV2(__gm__ IoT *x, __gm__ IoT *convStates, __gm__ IoT *wexp, __gm__ IoT *ring,
                      __gm__ int32_t *seqStartLoc, __gm__ int32_t *cacheIdx, __gm__ uint8_t *hasInitState, uint32_t dim,
                      uint32_t stateLen, uint32_t inputMode, uint32_t seqLen, uint32_t K, uint32_t chanTileWidth,
                      uint32_t blocksPerSeq, uint32_t numTokenChunks, uint32_t gridSize, int32_t padSlot, uint32_t cfg)
{
    constexpr uint32_t KMAX = 16u * RS;
    constexpr uint32_t HALO_PAD = 16u;
    constexpr uint32_t L1ROWS = T_TILE + HALO_PAD;
    constexpr uint32_t NGRP = D_TILE / 16u;

    constexpr uint32_t A_BYTES = T_TILE * KMAX * sizeof(IoT);
    constexpr uint32_t B_BYTES = KMAX * 16u * sizeof(IoT);
    constexpr uint32_t C_BYTES = T_TILE * D_TILE * sizeof(float);
    constexpr uint32_t ABUF = (2u * A_BYTES <= 64u * 1024u) ? 2u : 1u;
    constexpr uint32_t CBUF = (2u * C_BYTES <= 128u * 1024u) ? 2u : 1u;
    static_assert(ABUF * A_BYTES <= 64u * 1024u, "conv1d hybrid v2: L0A overflow");
    static_assert(NGRP * B_BYTES <= 64u * 1024u, "conv1d hybrid v2: L0B overflow");
    static_assert(CBUF * C_BYTES <= 128u * 1024u, "conv1d hybrid v2: L0C overflow");

    constexpr uint32_t L1_X_BYTES = L1ROWS * D_TILE * sizeof(IoT);
    constexpr uint32_t L1_B_BYTES = NGRP * B_BYTES;
    constexpr uint32_t L1_B0 = 2u * L1_X_BYTES;
    static_assert(L1_B0 + 2u * L1_B_BYTES <= 512u * 1024u, "conv1d hybrid v2: L1 overflow");

    using Nd2 = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Nd2S = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GIo = pto::GlobalTensor<IoT, Nd2, Nd2S, pto::Layout::ND>;
    using Dn2 = pto::Shape<1, 1, 1, DYNAMIC, 16>;
    using Dn2S = pto::Stride<1, 1, 1, 1, DYNAMIC>;
    using GDn = pto::GlobalTensor<IoT, Dn2, Dn2S, pto::Layout::DN>;

    using XMat = Tile<TileType::Mat, IoT, L1ROWS, D_TILE, BLayout::ColMajor, DYNAMIC, DYNAMIC, SLayout::RowMajor>;
    using XConv = ConvTile<TileType::Mat, IoT, (int)(L1ROWS * 16u * sizeof(IoT)), pto::Layout::NC1HWC0,
                           pto::ConvTileShape<1, 1, 1, DYNAMIC, 16>>;
    using BMat = Tile<TileType::Mat, IoT, KMAX, 16, BLayout::RowMajor, DYNAMIC, 16, SLayout::ColMajor>;
    using ATile = TileLeft<IoT, T_TILE, KMAX, DYNAMIC, DYNAMIC>;
    using BTile = TileRight<IoT, KMAX, 16, DYNAMIC, 16>;
    using CGrp = TileAcc<float, T_TILE, 16, DYNAMIC, 16>;
    using CAll = TileAcc<float, T_TILE, D_TILE, DYNAMIC, DYNAMIC>;

    const int32_t haloTokens = (int32_t)K - 1;
    const uint32_t kExtent = 16u * K;
    const uint32_t numBlocks = get_block_num();
    const uint32_t blockId = get_block_idx();
    // Measured dead, kept as compile-time constants so the branches below fold away without
    // restructuring the chunk loop: XDBUF and CDBUF are EXACT no-ops at the shipping tiling
    // (CBUF is 1 at T128xD256, and with one chunk per task there is no next chunk to
    // prefetch), and BAR only restored the old pipe_barrier(PIPE_ALL) drains for ablation.
    constexpr bool enableXDoubleBuf = false;
    constexpr bool enableCDoubleBuf = false;
    constexpr bool useBarriers = false;

    uint32_t chunkCount = 0u;
    uint32_t taskCount = 0u;
    bool aPend[2] = {false, false};
    bool xPend[2] = {false, false};
    bool bPend[2] = {false, false};
    bool cPend[2] = {false, false};
    uint32_t curPadLeft = 0xffffffffu;

    XConv fmap((int64_t)L1ROWS);
    fmap.SetFmapH(1);
    fmap.SetFmapW((uint16_t)L1ROWS);
    fmap.SetChannelSize(16);
    fmap.SetFilterH(1);
    fmap.SetFilterW((uint16_t)K);
    fmap.SetStrideH(1);
    fmap.SetStrideW(1);
    fmap.SetDilationH(1);
    fmap.SetDilationW(1);
    fmap.SetPadList(0, 0);
    fmap.SetPadList(1, 0);
    fmap.SetPadList(2, 0);
    fmap.SetPadList(3, 0);
    set_padding(0);

// The xext rows [tokenBase-haloTokens, tokenBase+tileTokens) of the current task's channel tile into
// L1 at `dstBase`. Identical geometry to runCube's (B) block; a macro so the prefetch,
// the single-buffer path and the prologue all share one copy.
#define HYV2_LOAD_X(dstBase, l0v, tv)                                                                        \
    do {                                                                                                     \
        const int32_t tokenBase_ = (l0v);                                                                    \
        const int32_t numHistRows_ = (haloTokens > tokenBase_) ? (haloTokens - tokenBase_) : 0;              \
        const int32_t numRealRows_ = haloTokens + (tv)-numHistRows_;                                         \
        if (numHistRows_ > 0 && !taskGeom.hasInitState) {                                                    \
            XMat xL1_((uint32_t)numRealRows_, (uint32_t)taskGeom.validChans);                                \
            TASSIGN(xL1_, (dstBase));                                                                        \
            GIo xG_(x + (uint64_t)taskGeom.seqStart * dim + taskGeom.chanBase,                               \
                    Nd2(numRealRows_, taskGeom.validChans), Nd2S(dim));                                      \
            TLOAD(xL1_, xG_);                                                                                \
        } else {                                                                                             \
            if (numHistRows_ > 0) {                                                                          \
                XMat histL1_((uint32_t)numHistRows_, (uint32_t)taskGeom.validChans);                         \
                TASSIGN(histL1_, (dstBase));                                                                 \
                GIo hG_(convStates + ((uint64_t)taskGeom.cacheIdx * stateLen + (uint32_t)tokenBase_) * dim + \
                            taskGeom.chanBase,                                                               \
                        Nd2(numHistRows_, taskGeom.validChans), Nd2S(dim));                                  \
                TLOAD(histL1_, hG_);                                                                         \
            }                                                                                                \
            XMat xL1_((uint32_t)numRealRows_, (uint32_t)taskGeom.validChans);                                \
            TASSIGN(xL1_, (dstBase) + (uint32_t)numHistRows_ * 16u * sizeof(IoT));                           \
            const int32_t firstTok_ = (tokenBase_ - haloTokens) + numHistRows_;                              \
            GIo xG_(x + (uint64_t)(taskGeom.seqStart + firstTok_) * dim + taskGeom.chanBase,                 \
                    Nd2(numRealRows_, taskGeom.validChans), Nd2S(dim));                                      \
            TLOAD(xL1_, xG_);                                                                                \
        }                                                                                                    \
    } while (0)

    for (uint32_t task = blockId; task < gridSize; task += numBlocks) {
        TaskGeom taskGeom = resolveTask(seqStartLoc, cacheIdx, hasInitState, task, dim, inputMode, seqLen,
                                        chanTileWidth, blocksPerSeq, numTokenChunks, padSlot, (uint32_t)haloTokens);
        if (!taskGeom.valid) continue;
        const uint32_t numChanGroups = ((uint32_t)taskGeom.validChans + 15u) / 16u;
        const uint32_t l1BufParity = enableXDoubleBuf ? (taskCount & 1u) : 0u;
        const uint32_t l1BBase = L1_B0 + l1BufParity * L1_B_BYTES;
        ++taskCount;

        if (useBarriers) pipe_barrier(PIPE_ALL);

        // ---- (A) B operand: GM -> L1 (MTE2) -> L0B (MTE1), ONCE for the whole
        // channel tile, one distinct L0B tile per 16-channel group.
        if (bPend[l1BufParity]) {
            B_FREE_WAIT(l1BufParity);
            bPend[l1BufParity] = false;
        }
        for (uint32_t grp = 0; grp < numChanGroups; ++grp) {
            BMat bL1(kExtent);
            TASSIGN(bL1, l1BBase + grp * B_BYTES);
            GDn bG(wexp + (uint64_t)(taskGeom.chanBase / 16u + grp) * 16u * kExtent, Dn2(kExtent), Dn2S(kExtent));
            TLOAD(bL1, bG);
        }
        B_RDY_SET(l1BufParity);
        B_RDY_WAIT(l1BufParity);
        // WAR on L0B: the previous task's TMATMULs must have retired before MTE1
        // overwrites the L0B tiles.
        for (uint32_t aSlot = 0; aSlot < ABUF; ++aSlot) {
            if (aPend[aSlot]) {
                A_FREE_WAIT(aSlot);
                aPend[aSlot] = false;
            }
        }
        for (uint32_t grp = 0; grp < numChanGroups; ++grp) {
            BMat bL1(kExtent);
            BTile b(kExtent);
            TASSIGN(bL1, l1BBase + grp * B_BYTES);
            TASSIGN(b, grp * B_BYTES);
            TEXTRACT(b, bL1, 0, 0);
        }
        B_FREE_SET(l1BufParity);
        bPend[l1BufParity] = true;

        int32_t tokenBase = taskGeom.tokenLo;
        if (enableXDoubleBuf && tokenBase < taskGeom.tokenHi) {
            int32_t firstTileTokens = (int32_t)T_TILE;
            if (tokenBase + firstTileTokens > taskGeom.tokenHi) firstTileTokens = taskGeom.tokenHi - tokenBase;
            const uint32_t xParity = chunkCount & 1u;
            if (xPend[xParity]) {
                X_FREE_WAIT(xParity);
                xPend[xParity] = false;
            }
            HYV2_LOAD_X(xParity * L1_X_BYTES, tokenBase, firstTileTokens);
            X_RDY_SET(xParity);
        }

        for (; tokenBase < taskGeom.tokenHi; tokenBase += (int32_t)T_TILE) {
            int32_t tileTokens = (int32_t)T_TILE;
            if (tokenBase + tileTokens > taskGeom.tokenHi) tileTokens = taskGeom.tokenHi - tokenBase;
            const uint32_t slot = chunkCount % NSLOT;
            const uint32_t xParity = enableXDoubleBuf ? (chunkCount & 1u) : 0u;
            const uint32_t cParity = enableCDoubleBuf ? (chunkCount & 1u) : 0u;
            ++chunkCount;

            if (!enableXDoubleBuf) {
                if (useBarriers) pipe_barrier(PIPE_ALL);
                if (xPend[0]) {
                    X_FREE_WAIT(0);
                    xPend[0] = false;
                }
                HYV2_LOAD_X(0u, tokenBase, tileTokens);
                X_RDY_SET(0);
            }
            X_RDY_WAIT(xParity);

            // Issue the NEXT chunk's x load now, so MTE2 runs under this chunk's
            // MTE1/M work. The ring credit is NOT needed for this -- it gates the
            // fixpipe destination, not L1.
            if (enableXDoubleBuf && tokenBase + (int32_t)T_TILE < taskGeom.tokenHi) {
                const int32_t nextTokenBase = tokenBase + (int32_t)T_TILE;
                int32_t nextTileTokens = (int32_t)T_TILE;
                if (nextTokenBase + nextTileTokens > taskGeom.tokenHi)
                    nextTileTokens = taskGeom.tokenHi - nextTokenBase;
                const uint32_t xParityNext = xParity ^ 1u;
                if (xPend[xParityNext]) {
                    X_FREE_WAIT(xParityNext);
                    xPend[xParityNext] = false;
                }
                HYV2_LOAD_X(xParityNext * L1_X_BYTES, nextTokenBase, nextTileTokens);
                X_RDY_SET(xParityNext);
            }

            const int32_t numHistRows = (haloTokens > tokenBase) ? (haloTokens - tokenBase) : 0;
            const uint32_t padLeft = (numHistRows > 0 && !taskGeom.hasInitState) ? (uint32_t)numHistRows : 0u;
            if (padLeft != curPadLeft) {
                pipe_barrier(PIPE_ALL);  // fmatrix is PIPE_S; order it against in-flight MTE1
                fmap.SetPadList(0, (uint8_t)padLeft);
                SETFMATRIX(fmap);
                curPadLeft = padLeft;
            }

            if (cPend[cParity]) {
                C_FREE_WAIT(cParity);
                cPend[cParity] = false;
            }

            {
                const uint32_t l1XBase = xParity * L1_X_BYTES;
                uint32_t issued = 0u;
                while (issued < numChanGroups && issued < ABUF) {
                    const uint32_t aSlot = issued & (ABUF - 1u);
                    if (aPend[aSlot]) {
                        A_FREE_WAIT(aSlot);
                        aPend[aSlot] = false;
                    }
                    ATile a((uint32_t)tileTokens, kExtent);
                    TASSIGN(a, aSlot * A_BYTES);
                    TASSIGN(fmap, l1XBase + issued * L1ROWS * 16u * sizeof(IoT));
                    TIMG2COL(a, fmap, 0, 0);
                    A_RDY_SET(aSlot);
                    ++issued;
                }
                for (uint32_t grp = 0; grp < numChanGroups; ++grp) {
                    const uint32_t aSlot = grp & (ABUF - 1u);
                    ATile a((uint32_t)tileTokens, kExtent);
                    BTile b(kExtent);
                    CGrp c((uint32_t)tileTokens);
                    TASSIGN(a, aSlot * A_BYTES);
                    TASSIGN(b, grp * B_BYTES);
                    TASSIGN(c, cParity * C_BYTES + grp * T_TILE * 16u * sizeof(float));
                    A_RDY_WAIT(aSlot);
                    TMATMUL(c, a, b);
                    A_FREE_SET(aSlot);
                    aPend[aSlot] = true;
                    if (issued < numChanGroups) {
                        const uint32_t aSlot2 = issued & (ABUF - 1u);
                        if (aPend[aSlot2]) {
                            A_FREE_WAIT(aSlot2);
                            aPend[aSlot2] = false;
                        }
                        ATile a2((uint32_t)tileTokens, kExtent);
                        TASSIGN(a2, aSlot2 * A_BYTES);
                        TASSIGN(fmap, l1XBase + issued * L1ROWS * 16u * sizeof(IoT));
                        TIMG2COL(a2, fmap, 0, 0);
                        A_RDY_SET(aSlot2);
                        ++issued;
                    }
                }
            }
            X_FREE_SET(xParity);
            xPend[xParity] = true;

            C_RDY_SET(cParity);
            C_RDY_WAIT(cParity);
            wait_flag_dev(FLAG_FREE[slot]);
            {
                CAll cAll((uint32_t)tileTokens, (uint32_t)taskGeom.validChans);
                TASSIGN(cAll, cParity * C_BYTES);
                GIo rG(ring + (uint64_t)(blockId * NSLOT + slot) * T_TILE * D_TILE,
                       Nd2(tileTokens, taskGeom.validChans), Nd2S(D_TILE));
                TSTORE(rG, cAll);
            }
            C_FREE_SET(cParity);
            cPend[cParity] = true;
            ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_CV_MODE, FLAG_RDY[slot]));
        }
    }
#undef HYV2_LOAD_X

    // Drain every outstanding event so the flag file is clean for the next launch.
    for (uint32_t aSlot = 0; aSlot < ABUF; ++aSlot) {
        if (aPend[aSlot]) A_FREE_WAIT(aSlot);
    }
    for (uint32_t parity = 0; parity < 2u; ++parity) {
        if (xPend[parity]) X_FREE_WAIT(parity);
    }
    for (uint32_t parity = 0; parity < 2u; ++parity) {
        if (bPend[parity]) B_FREE_WAIT(parity);
    }
    for (uint32_t parity = 0; parity < 2u; ++parity) {
        if (cPend[parity]) C_FREE_WAIT(parity);
    }
    for (uint32_t slot = 0; slot < NSLOT; ++slot) wait_flag_dev(FLAG_FREE[slot]);
}
#endif  // __DAV_CUBE__

// ===========================================================================
// AIV: consume the staging slot -> +bias -> SiLU -> cast -> y.
// The two sub-blocks of a block split the chunk's token rows.
// ===========================================================================
template <typename IoT, uint32_t RS, uint32_t T_TILE, uint32_t D_TILE, uint32_t ROWS_V, uint32_t UB_BASE>
AICORE void runVec(__gm__ IoT *y, __gm__ IoT *bias, __gm__ IoT *ring, __gm__ int32_t *seqStartLoc,
                   __gm__ int32_t *cacheIdx, __gm__ uint8_t *hasInitState, uint32_t dim, uint32_t inputMode,
                   uint32_t seqLen, uint32_t chanTileWidth, uint32_t blocksPerSeq, uint32_t numTokenChunks,
                   uint32_t gridSize, uint32_t activation, uint32_t hasBias, int32_t padSlot, uint32_t pipelined,
                   uint32_t haloTokens)
{
    using Nd2 = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Nd2S = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GIo = pto::GlobalTensor<IoT, Nd2, Nd2S, pto::Layout::ND>;

    // ValidCol is STATIC on every tile that COMPUTES, which is what keeps TCOLEXPANDADD in
    // norm mode -- the mode is a compile-time Cols==ValidCol test, not a runtime extent
    // (pto/npu/a2a3/TColExpandBinOp.hpp:76). A ragged last channel tile is therefore PADDED
    // to D_TILE from the ring and discarded at the store; nothing hot is masked.
    using SlabIo = Tile<TileType::Vec, IoT, ROWS_V, D_TILE, BLayout::RowMajor, DYNAMIC, D_TILE>;
    using SlabF = Tile<TileType::Vec, float, ROWS_V, D_TILE, BLayout::RowMajor, DYNAMIC, D_TILE>;
    using BiasIo = Tile<TileType::Vec, IoT, 1, D_TILE, BLayout::RowMajor, 1, D_TILE>;
    using BiasF = Tile<TileType::Vec, float, 1, D_TILE, BLayout::RowMajor, 1, D_TILE>;
    // RAGGED VIEWS of the same UB bytes, for the only two transfers touching CALLER-OWNED
    // memory: the bias read and the y store. Those cannot be padded -- a full D_TILE
    // transfer would read past `bias` and write into the next y token. dim % 16 == 0 makes
    // the strided TSTORE exact, since its source gap is counted in 32-byte blocks.
    using SlabIoV = Tile<TileType::Vec, IoT, ROWS_V, D_TILE, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
    using BiasIoV = Tile<TileType::Vec, IoT, 1, D_TILE, BLayout::RowMajor, 1, DYNAMIC>;

    // OFF_IN is DOUBLE BUFFERED for CORRECTNESS, not tuning. The MTE2->V and V->MTE3 flags
    // order slab n's load before its read, but nothing orders that read before slab n+1's
    // WRITE -- a WAR race. D_TILE 128/256 win it on timing; 512 lost it in 4.5-24.7% of runs
    // with a run-varying error mask. Two buffers + an explicit V->MTE2 credit: <=0.1 us.
    constexpr uint32_t IN_BUF = 2u;
    constexpr uint32_t IN_BYTES = ROWS_V * D_TILE * sizeof(IoT);
    constexpr uint32_t OFF_IN = UB_BASE;                                  // IN_BUF * IN_BYTES
    constexpr uint32_t OFF_ACC = OFF_IN + IN_BUF * IN_BYTES;              // fp32
    constexpr uint32_t OFF_TMP = OFF_ACC + ROWS_V * D_TILE * 4u;          // fp32
    constexpr uint32_t OFF_OUT = OFF_TMP + ROWS_V * D_TILE * 4u;          // IoT
    constexpr uint32_t OFF_BIO = OFF_OUT + ROWS_V * D_TILE * sizeof(IoT);
    constexpr uint32_t OFF_BF = OFF_BIO + D_TILE * sizeof(IoT);
    constexpr uint32_t UB_END = OFF_BF + D_TILE * 4u;
    static_assert(UB_END <= 192u * 1024u, "conv1d hybrid: AIV UB overflow");

    set_mask_norm();
    set_vector_mask(-1, -1);

    const uint32_t numBlocks = get_block_num();
    const uint32_t blockId = get_block_idx();
    const uint32_t subBlockIdx = get_subblockid();
    const uint32_t numSubBlocks = get_subblockdim();
    uint32_t chunkCount = 0u;

    // Prime the ring: NSLOT free credits, matched by the AIC's constant-length drain.
    if (pipelined) {
        pipe_barrier(PIPE_ALL);
        for (uint32_t s = 0; s < NSLOT; ++s) {
            ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(FFTS_CV_MODE, FLAG_FREE[s]));
        }
    }

    for (uint32_t task = blockId; task < gridSize; task += numBlocks) {
        TaskGeom taskGeom = resolveTask(seqStartLoc, cacheIdx, hasInitState, task, dim, inputMode, seqLen,
                                        chanTileWidth, blocksPerSeq, numTokenChunks, padSlot, (uint32_t)haloTokens);
        if (!taskGeom.valid) continue;

        const uint32_t validChans = (uint32_t)taskGeom.validChans;
        if (hasBias) {
            BiasIoV biasIoV(validChans);  // read only the real channels: bias is (dim,)
            BiasIo biasIo;                // ...then cast the padded row (pad channels are junk)
            BiasF biasF32;
            TASSIGN(biasIoV, OFF_BIO);
            TASSIGN(biasIo, OFF_BIO);
            TASSIGN(biasF32, OFF_BF);
            GIo bG(bias + taskGeom.chanBase, Nd2(1, validChans), Nd2S(validChans));
            TLOAD(biasIoV, bG);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
            TCVT(biasF32, biasIo, pto::RoundMode::CAST_NONE);
            PIPE_BARRIER_VEC();
        }

        for (int32_t tokenBase = taskGeom.tokenLo; tokenBase < taskGeom.tokenHi; tokenBase += (int32_t)T_TILE) {
            int32_t tileTokens = (int32_t)T_TILE;
            if (tokenBase + tileTokens > taskGeom.tokenHi) tileTokens = taskGeom.tokenHi - tokenBase;
            const uint32_t slot = chunkCount % NSLOT;
            ++chunkCount;

            if (pipelined) {
                wait_flag_dev(FLAG_RDY[slot]);
            } else {
                pairBarrier();  // slot free (mirrors the AIC's pre-produce barrier)
                pairBarrier();  // slot ready
            }

            // split this chunk's rows between the two sub-blocks
            const int32_t rowsPerSubBlock = (tileTokens + (int32_t)numSubBlocks - 1) / (int32_t)numSubBlocks;
            const int32_t rowBase = (int32_t)subBlockIdx * rowsPerSubBlock;
            int32_t rowCount = rowsPerSubBlock;
            if (rowBase >= tileTokens) {
                rowCount = 0;
            } else if (rowBase + rowCount > tileTokens) {
                rowCount = tileTokens - rowBase;
            }

            uint32_t ubInBufIdx = 0u;  // which OFF_IN buffer this slab loads into
            uint32_t inFly = 0u;       // loads issued whose reader-credit has not been consumed
            for (int32_t slabRow = 0; slabRow < rowCount; slabRow += (int32_t)ROWS_V) {
                int32_t slabRows = (int32_t)ROWS_V;
                if (slabRow + slabRows > rowCount) slabRows = rowCount - slabRow;

                SlabIo in((uint32_t)slabRows);
                SlabF acc((uint32_t)slabRows);
                SlabF tmp((uint32_t)slabRows);
                SlabIo out((uint32_t)slabRows);
                // WAR: do not overwrite a buffer whose TCVT has not run. The first IN_BUF
                // loads go straight through; after that each one waits for a reader credit.
                if (inFly >= IN_BUF) {
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
                } else {
                    ++inFly;
                }
                TASSIGN(in, OFF_IN + ubInBufIdx * IN_BYTES);
                TASSIGN(acc, OFF_ACC);
                TASSIGN(tmp, OFF_TMP);
                TASSIGN(out, OFF_OUT);

                GIo sG(ring + (uint64_t)(blockId * NSLOT + slot) * T_TILE * D_TILE +
                           (uint64_t)(rowBase + slabRow) * D_TILE,
                       Nd2(slabRows, D_TILE), Nd2S(D_TILE));
                TLOAD(in, sG);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                TCVT(acc, in, pto::RoundMode::CAST_NONE);
                PIPE_BARRIER_VEC();
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);  // this OFF_IN buffer is free again
                ubInBufIdx = (ubInBufIdx + 1u) % IN_BUF;
                if (hasBias) {
                    BiasF biasF32;
                    TASSIGN(biasF32, OFF_BF);
                    TCOLEXPANDADD(acc, acc, biasF32);
                    PIPE_BARRIER_VEC();
                }
                if (activation) {
                    TMULS(tmp, acc, -1.0f);
                    PIPE_BARRIER_VEC();
                    TEXP(tmp, tmp);
                    PIPE_BARRIER_VEC();
                    TADDS(tmp, tmp, 1.0f);
                    PIPE_BARRIER_VEC();
                    TDIV(acc, acc, tmp);
                    PIPE_BARRIER_VEC();
                }
                TCVT(out, acc, pto::RoundMode::CAST_NONE);
                PIPE_BARRIER_VEC();
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                SlabIoV outV((uint32_t)slabRows, validChans);
                TASSIGN(outV, OFF_OUT);
                GIo yG(y + (uint64_t)(taskGeom.seqStart + tokenBase + rowBase + slabRow) * dim + taskGeom.chanBase,
                       Nd2(slabRows, validChans), Nd2S(dim));
                TSTORE(yG, outV);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
            // Balance the WAR credits before the chunk-closing barrier: every slab set one,
            // and only (slabs - inFly) were consumed above. An unconsumed set_flag would be
            // taken by a later wait_flag and let a load run ahead of its reader again.
            while (inFly > 0u) {
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
                --inFly;
            }

            if (pipelined) {
                pipe_barrier(PIPE_ALL);
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(FFTS_CV_MODE, FLAG_FREE[slot]));
            } else {
                pairBarrier();  // slot consumed -- MUST match the AIC's third barrier
            }
        }
    }
}

// ===========================================================================
template <typename IoT, uint32_t RS, uint32_t D_TILE, uint32_t UB_BASE>
AICORE void runWbVec(__gm__ IoT *x, __gm__ IoT *convStates, __gm__ int32_t *seqStartLoc, __gm__ int32_t *cacheIdx,
                     __gm__ uint8_t *hasInitState, uint32_t dim, uint32_t batch, uint32_t inputMode, uint32_t seqLen,
                     uint32_t stateLen, uint32_t K, uint32_t chanTileWidth, uint32_t blocksPerSeq, int32_t padSlot,
                     uint32_t numTokenChunks, uint32_t ownerMode)
{
    using Nd2 = pto::Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using Nd2S = pto::Stride<1, 1, 1, DYNAMIC, 1>;
    using GIo = pto::GlobalTensor<IoT, Nd2, Nd2S, pto::Layout::ND>;
    // ValidCol is DYNAMIC here, unlike the conv slab: every transfer below is a SINGLE
    // row (nBurst == 1), so the 32-byte-block UB gap that constrains a strided store is
    // never applied and any lane count is exact. The writeback must be exact rather than
    // padded because conv_states is caller-owned AND is compared bit-exactly.
    using RowIo = Tile<TileType::Vec, IoT, 1, D_TILE, BLayout::RowMajor, 1, DYNAMIC>;
    using RowF = Tile<TileType::Vec, float, 1, D_TILE, BLayout::RowMajor, 1, DYNAMIC>;

    constexpr uint32_t ROW_B = D_TILE * sizeof(IoT);
    constexpr uint32_t OFF_ROWS = UB_BASE;                      // up to RS-1 staged rows
    constexpr uint32_t OFF_SCR = OFF_ROWS + (RS - 1u) * ROW_B;   // fp32 zeroing scratch
    static_assert(OFF_SCR + D_TILE * 4u <= 192u * 1024u, "conv1d hybrid: wb UB overflow");

    set_mask_norm();
    set_vector_mask(-1, -1);

    const uint32_t numBlocks = get_block_num();
    const uint32_t subBlockIdx = get_subblockid();
    const uint32_t nAiv = numBlocks * get_subblockdim();
    const uint32_t aivId = get_block_idx() + subBlockIdx * numBlocks;
    const uint32_t blockId = get_block_idx();
    const uint32_t gridSize = batch * blocksPerSeq;
    const int32_t haloTokens = (int32_t)K - 1;
    // ownerMode: walk every unit and keep the ones whose tokenChunkIdx==0 conv task landed on
    // this block, then alternate them between the two sub-blocks. The scan is O(batch *
    // dim/D_TILE) scalar iterations (<= 512 in the widest supported shape).
    uint32_t owned = 0u;

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);  // UB starts free
    for (uint32_t task = ownerMode ? 0u : aivId; task < gridSize; task += (ownerMode ? 1u : nAiv)) {
        // The conv task that READS this unit's history is its lc==0 task. Under the
        // B x T x C packing in resolveTask (task = chanTile + blocksPerSeq*(chunk + numTokenChunks*seq))
        // that index is chanTile + blocksPerSeq*numTokenChunks*seq -- NOT `task * numTokenChunks`,
        // which is the B x C x T form and coincides only when numTokenChunks == 1. Getting
        // this wrong hands the writeback to a block that never read the history, which breaks
        // the barrier-free ordering this whole mode depends on.
        const uint32_t convTask0 =
            (task % blocksPerSeq) + blocksPerSeq * numTokenChunks * (task / blocksPerSeq);
        if (ownerMode == 1u) {
            // Cube path: conv tasks go to BLOCKS; either sub-block of the owning block is
            // ordered after its AIC by the ring handshake.
            if (convTask0 % numBlocks != blockId) continue;
            if ((owned++ & 1u) != subBlockIdx) continue;
        } else if (ownerMode == 2u) {
            // AIV-direct path: conv tasks go to AIVs, so the SAME core that read
            // convStates[seqCacheIdx,0:K-1] for this unit writes it back -- plain program
            // order on one core's MTE2/MTE3 queues, no barrier of any kind.
            if (convTask0 % nAiv != aivId) continue;
        }
        const uint32_t chanTileIdx = task % blocksPerSeq;
        const uint32_t seq = task / blocksPerSeq;
        int32_t seqStart, seqTokens;
        if (inputMode == 0u) {
            seqStart = seqStartLoc[seq];
            seqTokens = seqStartLoc[seq + 1] - seqStart;
        } else {
            seqStart = (int32_t)(seq * seqLen);
            seqTokens = (int32_t)seqLen;
        }
        if (seqTokens == 0) continue;
        const int32_t seqCacheIdx = cacheIdx[seq];
        if (seqCacheIdx == padSlot) continue;
        const bool seqHasInitState = hasInitState[seq] != 0;
        const uint32_t chanBase = chanTileIdx * chanTileWidth;
        // Channel tiles stay DISJOINT (tile chanTileIdx owns exactly [chanBase,
        // min(chanBase+chanTileWidth, dim))), so the barrier-free owner rule above is untouched:
        // no two writeback units, and no conv task other than this tile's own tokenChunkIdx==0
        // chunk, touch these columns.
        const uint32_t validChans = (dim - chanBase > chanTileWidth) ? chanTileWidth : (dim - chanBase);

        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        // (A) MTE2: stage xext[seqTokens .. seqTokens+K-2]. extIdx >= K-1 -> x[extIdx-(K-1)];
        //     extIdx < K-1 -> old history (convStates when seqHasInitState, else zero). The zero
        //     case loads x[seqStart] as a finite placeholder and multiplies it by 0 in (B).
        for (int32_t i = 0; i < haloTokens; ++i) {
            const int32_t extIdx = seqTokens + i;
            const int32_t xrow = extIdx - haloTokens;
            RowIo row(validChans);
            TASSIGN(row, OFF_ROWS + (uint32_t)i * ROW_B);
            if (xrow >= 0) {
                GIo sG(x + (uint64_t)(seqStart + xrow) * dim + chanBase, Nd2(1, validChans), Nd2S(validChans));
                TLOAD(row, sG);
            } else if (seqHasInitState) {
                GIo hG(convStates + ((uint64_t)seqCacheIdx * stateLen + (uint32_t)extIdx) * dim + chanBase,
                       Nd2(1, validChans), Nd2S(validChans));
                TLOAD(row, hG);
            } else {
                GIo sG(x + (uint64_t)seqStart * dim + chanBase, Nd2(1, validChans), Nd2S(validChans));
                TLOAD(row, sG);
            }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // (B) V: zero the pure-history rows when there is no initial state. TMULS has no
        //     bf16 overload, hence the fp32 round trip.
        for (int32_t i = 0; i < haloTokens; ++i) {
            const int32_t extIdx = seqTokens + i;
            if (extIdx - haloTokens < 0 && !seqHasInitState) {
                RowIo row(validChans);
                RowF f32(validChans);
                TASSIGN(row, OFF_ROWS + (uint32_t)i * ROW_B);
                TASSIGN(f32, OFF_SCR);
                TCVT(f32, row, pto::RoundMode::CAST_NONE);
                PIPE_BARRIER_VEC();
                TMULS(f32, f32, 0.0f);
                PIPE_BARRIER_VEC();
                TCVT(row, f32, pto::RoundMode::CAST_NONE);
            }
        }
        PIPE_BARRIER_VEC();
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        // (C) MTE3: convStates[seqCacheIdx, 0:K-1, chanBase:chanBase+D_TILE]
        for (int32_t i = 0; i < haloTokens; ++i) {
            RowIo row(validChans);
            TASSIGN(row, OFF_ROWS + (uint32_t)i * ROW_B);
            GIo dG(convStates + ((uint64_t)seqCacheIdx * stateLen + (uint32_t)i) * dim + chanBase, Nd2(1, validChans),
                   Nd2S(validChans));
            TSTORE(dG, row);
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);  // drain the final store
}

}  // namespace cc1dhy

// ---------------------------------------------------------------------------
// Entries. Per-RS (T_TILE, D_TILE) table -- must match the host's hybridTileForRing().
// T_TILE is capped so L0A (T_TILE * 16*RS * 2 B) stays <= 32 KiB.
// ---------------------------------------------------------------------------
#define HY_CONV_PARAMS                                                                                            \
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR seqStartLoc, GM_ADDR cacheIdx,           \
        GM_ADDR hasInitState, GM_ADDR y, GM_ADDR wexp, GM_ADDR ring, uint32_t dim, uint32_t batch,                \
        uint32_t inputMode, uint32_t seqLen, uint32_t stateLen, uint32_t width, uint32_t chanTileWidth,           \
        uint32_t blocksPerSeq, uint32_t numTokenChunks, uint32_t activation, uint32_t hasBias,                    \
        int32_t padSlot, uint32_t pipelined, uint32_t cfg

// AIV slab rows. The slab costs ROWS_V*D_TILE*(2+4+4+2) B of UB (in/acc/tmp/out), so
// a wide channel tile needs fewer rows to stay inside the 192 KiB budget.
#define HY_ROWS_V(TT) ((TT) < 32u ? (TT) : 32u)
#define HY_ROWS_V_DT(TT, DT) ((DT) >= 512u ? ((TT) < 16u ? (TT) : 16u) : HY_ROWS_V(TT))
// UB base of the main AIV region = end of the weight-expansion region.
// eye 1 KiB + wIo 32*RS + wF 64*RS + dF 1024*RS + dIo 512*RS  (all 32 B aligned)
#define HY_EXP_BYTES(RS) (1024u + 1632u * (RS))

// Prologue (AIVs publish the B operand) then ONE global barrier. Safe here because it runs
// exactly once per core -- inside the task loop, differing per-block trip counts would
// deadlock a SYNCALL. The HY_CFG bits let the host strip the prologue and fold the writeback.
#define HY_ENTRY_BODY(BODY, T, RS, TT, DT)                                                                     \
    if (!(cfg & cc1dhy::HY_CFG_NOEXP)) {                                                                       \
        HY_PROLOGUE(T, RS);                                                                                    \
    }                                                                                                          \
    if (!(cfg & cc1dhy::HY_CFG_NOSYNC)) {                                                                      \
        SYNCALL<SyncCoreType::Mix>();                                                                          \
    }                                                                                                          \
    if (!(cfg & cc1dhy::HY_CFG_NOBODY)) {                                                                      \
            BODY(T, RS, TT, DT);                                                                               \
    }                                                                                                          \
    if (cfg & cc1dhy::HY_CFG_WB) {                                                                             \
        if (!(cfg & cc1dhy::HY_CFG_WB2)) {                                                                     \
            SYNCALL<SyncCoreType::Mix>();                                                                      \
        }                                                                                                      \
            HY_WB(T, RS, DT);                                                                                  \
    }

// Declared MIX ratio. Left unset this is inferred, and the inferred choice brings up 1 Cube +
// 2 Vector per block; an EMPTY MIX kernel then costs 1.36 us + 0.408 us per AIC block (~10.2 us
// at blockDim 24) against ~0.04 us/block for an AIV-only launch. HY_MIX_RATIO lets that be
// measured rather than assumed.
#ifndef HY_TASK_TYPE
#define HY_DECLARE_TASK_TYPE()
#else
#define HY_DECLARE_TASK_TYPE() KERNEL_TASK_TYPE_DEFAULT(HY_TASK_TYPE)
#endif

#define DEF_HY(SUF, T, RS, TT, DT)                                                                             \
    extern "C" __global__ AICORE void causal_conv1d_hy_##SUF(HY_CONV_PARAMS)                                   \
    {                                                                                                          \
        HY_DECLARE_TASK_TYPE();                                                                                \
        const uint32_t gridSize = batch * blocksPerSeq * numTokenChunks;                                              \
        (void)x;                                                                                               \
        (void)weight;                                                                                             \
        (void)bias;                                                                                             \
        (void)convStates;                                                                                      \
        (void)y;                                                                                               \
        (void)stateLen;                                                                                        \
        (void)activation;                                                                                      \
        (void)hasBias;                                                                                         \
        (void)gridSize;                                                                                        \
        (void)cfg;                                                                                             \
        HY_ENTRY_BODY(HY_BODY, T, RS, TT, DT);                                                                 \
    }

// Channel-tile capacity of the AIV-direct body, per accumulator-ring size. Identical to
// the ring-size table above, because this path is the same decomposition as
// budget with a static_assert either way.

#if defined(__DAV_CUBE__)
#define HY_PROLOGUE(T, RS) ((void)0)
// The AIV-direct dataflow is pure Vector work; on the cube pass it compiles to nothing,
// so this entry still holds exactly ONE cube body.
#define HY_CUBE_V1(T, RS, TT, DT)                                                                               \
    cc1dhy::runCube<T, RS, TT, DT>((__gm__ T *)x, (__gm__ T *)convStates, (__gm__ T *)wexp, (__gm__ T *)ring,   \
                                   (__gm__ int32_t *)seqStartLoc, (__gm__ int32_t *)cacheIdx,                   \
                                   (__gm__ uint8_t *)hasInitState, dim, stateLen, inputMode, seqLen, width,     \
                                   chanTileWidth, blocksPerSeq, numTokenChunks, gridSize, padSlot, pipelined)
#define HY_CUBE_V2(T, RS, TT, DT)                                                                                 \
    cc1dhy::runCubeV2<T, RS, TT, DT>((__gm__ T *)x, (__gm__ T *)convStates, (__gm__ T *)wexp, (__gm__ T *)ring,   \
                                     (__gm__ int32_t *)seqStartLoc, (__gm__ int32_t *)cacheIdx,                   \
                                     (__gm__ uint8_t *)hasInitState, dim, stateLen, inputMode, seqLen, width,     \
                                     chanTileWidth, blocksPerSeq, numTokenChunks, gridSize, padSlot, cfg)
#define HY_BODY(T, RS, TT, DT) HY_CUBE_V1(T, RS, TT, DT)
// The folded writeback is pure AIV work; the Cube side only has to reach the barrier.
#define HY_WB(T, RS, DT) ((void)0)
// ONE cube body per entry: inlining both behind `if (cfg)` costs AIC scalar registers and
// measurably slows V1, so the host picks the entry. V2 needs NGRP L0B tiles, hence RS <= 16.
#define HY_BODY2(T, RS, TT, DT) HY_CUBE_V2(T, RS, TT, DT)
#elif defined(__DAV_VEC__)
#define HY_PROLOGUE(T, RS) cc1dhy::expandWeights<T, RS>((__gm__ T *)weight, (__gm__ T *)wexp, dim, width)
#define HY_BODY(T, RS, TT, DT)                                                                                   \
    cc1dhy::runVec<T, RS, TT, DT, HY_ROWS_V_DT(TT, DT), HY_EXP_BYTES(RS)>(                                       \
        (__gm__ T *)y, (__gm__ T *)bias, (__gm__ T *)ring, (__gm__ int32_t *)seqStartLoc,                        \
        (__gm__ int32_t *)cacheIdx, (__gm__ uint8_t *)hasInitState, dim, inputMode, seqLen, chanTileWidth,       \
        blocksPerSeq, numTokenChunks, gridSize, activation, hasBias, padSlot, pipelined, width - 1u)
#define HY_BODY2(T, RS, TT, DT) HY_BODY(T, RS, TT, DT)
#define HY_WB(T, RS, DT)                                                                                         \
    cc1dhy::runWbVec<T, RS, DT, HY_EXP_BYTES(RS)>(                                                               \
        (__gm__ T *)x, (__gm__ T *)convStates, (__gm__ int32_t *)seqStartLoc, (__gm__ int32_t *)cacheIdx,        \
        (__gm__ uint8_t *)hasInitState, dim, batch, inputMode, seqLen, stateLen, width, chanTileWidth,           \
        blocksPerSeq, padSlot, numTokenChunks, (cfg & cc1dhy::HY_CFG_WB2) ? 1u : 0u)
#else
#define HY_PROLOGUE(T, RS) ((void)0)
#define HY_BODY(T, RS, TT, DT) ((void)0)
#define HY_BODY2(T, RS, TT, DT) ((void)0)
#define HY_WB(T, RS, DT) ((void)0)
#endif

#define DEF_HY2(SUF, T, RS, TT, DT)                                                                            \
    extern "C" __global__ AICORE void causal_conv1d_hy_##SUF(HY_CONV_PARAMS)                                   \
    {                                                                                                          \
        HY_DECLARE_TASK_TYPE();                                                                                \
        const uint32_t gridSize = batch * blocksPerSeq * numTokenChunks;                                              \
        (void)x;                                                                                               \
        (void)weight;                                                                                             \
        (void)bias;                                                                                             \
        (void)convStates;                                                                                      \
        (void)y;                                                                                               \
        (void)stateLen;                                                                                        \
        (void)activation;                                                                                      \
        (void)hasBias;                                                                                         \
        (void)gridSize;                                                                                        \
        (void)cfg;                                                                                             \
        HY_ENTRY_BODY(HY_BODY2, T, RS, TT, DT);                                                                \
    }

// (ringSize, tokenTile, channelTile). The plain entries are unchanged from the
// original hybrid (runCube only); _v2 and _v2t256 are the new Cube body.
#define FOR_EACH_HY_RING(DO) \
    DO(2, 128, 128) DO(4, 128, 128) DO(8, 128, 128) DO(16, 64, 128) DO(32, 32, 128) DO(64, 16, 128)
#define FOR_EACH_HY_RING_V2(DO) DO(2, 128, 128) DO(4, 128, 128) DO(8, 128, 128) DO(16, 64, 128)
// WIDER CHANNEL TILE. D_TILE sets the GM burst of the x load and y store (D_TILE*2 bytes
// per row; 256 B at 128, against the Vector baseline's 5120 B). L0C caps T_TILE*D_TILE*4 at
// 128 KiB, so a wider tile buys burst back by shortening the token tile at equal TMATMULs.
#define FOR_EACH_HY_RING_D256(DO) DO(2, 128, 256) DO(4, 128, 256) DO(8, 128, 256)

#define DEFINE_HY_ENTRIES(RS, TT, DT)       \
    DEF_HY(rs##RS##_half, half, RS, TT, DT) \
    DEF_HY(rs##RS##_bf16, bfloat16_t, RS, TT, DT)
FOR_EACH_HY_RING(DEFINE_HY_ENTRIES)
#undef DEFINE_HY_ENTRIES

#define DEFINE_HY_ENTRIES_V2(RS, TT, DT)             \
    DEF_HY2(rs##RS##_half_v2, half, RS, TT, DT)      \
    DEF_HY2(rs##RS##_bf16_v2, bfloat16_t, RS, TT, DT)
FOR_EACH_HY_RING_V2(DEFINE_HY_ENTRIES_V2)
#undef DEFINE_HY_ENTRIES_V2


#define DEFINE_HY_ENTRIES_D256(RS, TT, DT)                \
    DEF_HY2(rs##RS##_half_v2d256, half, RS, TT, DT)       \
    DEF_HY2(rs##RS##_bf16_v2d256, bfloat16_t, RS, TT, DT)
FOR_EACH_HY_RING_D256(DEFINE_HY_ENTRIES_D256)
#undef DEFINE_HY_ENTRIES_D256

