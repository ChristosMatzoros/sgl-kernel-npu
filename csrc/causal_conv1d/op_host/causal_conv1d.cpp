#include "causal_conv1d.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_helper.h"

// Ring sizes the kernel is compiled for -- must match FOR_EACH_RING_SIZE in
// op_kernel/causal_conv1d.cpp. Each row is (ringSize, maxTileWidth); this one list
// drives the launch-stub declarations, the per-ring tile-width lookup, and the dispatch.
#define FOR_EACH_RING_SIZE(DO) DO(2, 4096) DO(4, 3072) DO(8, 1536) DO(16, 896) DO(32, 384) DO(64, 128)
#define FOR_EACH_V2_RING(DO) DO(2) DO(4) DO(8) DO(16)
#define FOR_EACH_WIDE_CHAN_TILE_RING(DO) DO(2) DO(4) DO(8)

// AscendC emits one host launch stub (aclrtlaunch_<entry>) per kernel entry. A macro
// can't generate #include directives, so rather than pull in 24 generated headers we
// forward-declare the stubs from the ring-size list (their definitions come from the
// linked causal_conv1d_kernel lib; same approach as causal_conv1d_update).

// EXEC_KERNEL_CMD launches via ACLRT_LAUNCH_KERNEL(name) -> the aclrtlaunch_<name> symbol.
#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif
// clang-format off
// Launch-stub arg types = (blockDim, stream, then the kernel params with GM_ADDR ->
// void*); must mirror CONV_PARAMS / WB_PARAMS in op_kernel/causal_conv1d.cpp.
#define CONV_STUB_PARAMS                                                                                       \
    uint32_t, aclrtStream, void *, void *, void *, void *, void *, void *, void *, void *, uint32_t, uint32_t, \
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t
#define WRITEBACK_STUB_PARAMS                                                                                  \
    uint32_t, aclrtStream, void *, void *, void *, void *, void *, uint32_t, uint32_t, uint32_t, uint32_t,     \
        uint32_t, uint32_t, uint32_t, uint32_t, int32_t
// Worker macro: one ring size -> the conv + writeback stub, each for half and bf16.
#define HY_CONV_STUB_PARAMS                                                                                  \
    uint32_t, aclrtStream, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,   \
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,  \
        uint32_t, int32_t, uint32_t, uint32_t
#define DECLARE_LAUNCH_STUBS(ringSize, maxTileWidth)                                             \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_rs##ringSize##_half(CONV_STUB_PARAMS);         \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_rs##ringSize##_bf16(CONV_STUB_PARAMS);         \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_hy_rs##ringSize##_half(HY_CONV_STUB_PARAMS);   \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_hy_rs##ringSize##_bf16(HY_CONV_STUB_PARAMS);   \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_wbf_rs##ringSize##_half(CONV_STUB_PARAMS);     \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_wbf_rs##ringSize##_bf16(CONV_STUB_PARAMS);
FOR_EACH_RING_SIZE(DECLARE_LAUNCH_STUBS)  // expand the list -> all launch-stub declarations
#undef DECLARE_LAUNCH_STUBS
#define DECLARE_V2_STUBS(ringSize)                                                                  \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_hy_rs##ringSize##_half_v2(HY_CONV_STUB_PARAMS);   \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_hy_rs##ringSize##_bf16_v2(HY_CONV_STUB_PARAMS);
FOR_EACH_V2_RING(DECLARE_V2_STUBS)
#undef DECLARE_V2_STUBS
#define DECLARE_D256_STUBS(ringSize)                                                                    \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_hy_rs##ringSize##_half_v2d256(HY_CONV_STUB_PARAMS);   \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_hy_rs##ringSize##_bf16_v2d256(HY_CONV_STUB_PARAMS);
FOR_EACH_WIDE_CHAN_TILE_RING(DECLARE_D256_STUBS)
#undef DECLARE_D256_STUBS
#define DECLARE_FMA_STUBS(ringSize, maxTileWidth)                                        \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_rs##ringSize##_half_fma(CONV_STUB_PARAMS); \
    extern "C" uint32_t aclrtlaunch_causal_conv1d_rs##ringSize##_bf16_fma(CONV_STUB_PARAMS);
FOR_EACH_RING_SIZE(DECLARE_FMA_STUBS)
#undef DECLARE_FMA_STUBS
#undef HY_CONV_STUB_PARAMS
#undef WRITEBACK_STUB_PARAMS
#undef CONV_STUB_PARAMS
// clang-format on

namespace sglang {
namespace npu_kernel {
namespace {  // file-local helpers (internal linkage; avoids clashes with other ops)

// Accumulator-ring size = smallest power of two >= width. The kernel templates on the
// ring size (compile-time) and takes the width at runtime; the host computes the ring
// size here and launches the matching variant, so any width <= ring size reuses it.
constexpr uint32_t roundUpToPow2(uint32_t width)
{
    uint32_t n = (width != 0u) ? width - 1u : 0u;
    n |= n >> 1u;
    n |= n >> 2u;
    n |= n >> 4u;
    n |= n >> 8u;
    n |= n >> 16u;
    return n + 1u;
}

#define RING_EQ_OR(ringSize) || (ringSize##u == r)
constexpr bool isV2Ring(uint32_t r)
{
    return false FOR_EACH_V2_RING(RING_EQ_OR);
}
constexpr bool isWideChanTileRing(uint32_t r)
{
    return false FOR_EACH_WIDE_CHAN_TILE_RING(RING_EQ_OR);
}
#undef RING_EQ_OR

// Per-ring channel-tile width -- MUST match the (ringSize, maxTileWidth) the kernel is
// compiled with. Architectures with larger UB, can use wider tiles.
// The host has no compile-time device-arch macro, so the caller picks the table at runtime
#define FOR_EACH_RING_SIZE_A5(DO) DO(2, 5120) DO(4, 4096) DO(8, 2048) DO(16, 1152) DO(32, 512) DO(64, 128)
#define MAX_WIDTH_CASE(ringSize, maxTileWidth) \
    case ringSize:                             \
        return maxTileWidth##u;
uint32_t maxTileWidthForRing(uint32_t ringSize, bool wideUb)
{
    if (wideUb) {
        switch (ringSize) {
            FOR_EACH_RING_SIZE_A5(MAX_WIDTH_CASE)
            default:
                return 0u;
        }
    }
    switch (ringSize) {
        FOR_EACH_RING_SIZE(MAX_WIDTH_CASE)
        default:
            return 0u;
    }
}
#undef MAX_WIDTH_CASE
#undef FOR_EACH_RING_SIZE_A5

// Supported filter widths: any width in [2, 64], routed to the roundUpToPow2(width)
// ring variant. width > 64 would need ring 128, which does not fit UB.
constexpr bool isSupportedWidth(uint32_t width)
{
    return width >= 2u && width <= 64u;
}

template <class T>
constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

std::pair<uint32_t, uint32_t> tiling_causal_conv1d(uint64_t numCores, uint64_t batch, const uint64_t dim,
                                                   const uint64_t seqLength, const uint64_t width,
                                                   const uint64_t maxChanTileWidth)
{
    // 128 = Number of Vector Lanes in FP16/BF16
    constexpr uint64_t minChanTileWidth = 128;  // Must divide maxChanTileWidth

    uint64_t gcdCoreBatch = std::gcd(numCores, batch);
    numCores /= gcdCoreBatch;
    batch /= gcdCoreBatch;

    uint64_t numChanTiles = ceil_div(dim, maxChanTileWidth);
    uint64_t chanTileWidth = ceil_div(ceil_div(dim, numChanTiles), minChanTileWidth) * minChanTileWidth;

    uint64_t numTokenChunks = 1u;
    double bestScore = std::numeric_limits<double>::infinity();

    numChanTiles = ceil_div(dim, chanTileWidth);
    const uint64_t depthNumerator = batch * numChanTiles;

    const uint64_t maxNumChunks = numCores / std::gcd(numCores, numChanTiles);
    for (uint64_t numChunks = 1u; numChunks <= maxNumChunks; ++numChunks) {
        uint64_t depth = ceil_div(depthNumerator * numChunks, numCores);
        uint64_t tokensPerChunk = ceil_div(seqLength, numChunks);
        uint64_t work = tokensPerChunk + width;
        double score = static_cast<double>(depth) * static_cast<double>(work);
        if (score < bestScore) {
            bestScore = score;
            numTokenChunks = numChunks;
        }
    }

    return {static_cast<uint32_t>(chanTileWidth), static_cast<uint32_t>(numTokenChunks)};
}

constexpr uint32_t HY_CHAN_TILE_DEFAULT = 128u;
constexpr uint32_t HY_CHAN_TILE_WIDE = 256u;
constexpr uint32_t HY_TOKEN_TILE_WIDE_CHAN = 128u;
constexpr uint32_t HY_RING_SLOTS = 3u;
constexpr int64_t HY_WEXP_GROUP = 16;
constexpr uint32_t DIM_ALIGN_ELEMS = 16u;
#define FOR_EACH_HY_RING(DO) DO(2, 128) DO(4, 128) DO(8, 128) DO(16, 64) DO(32, 32) DO(64, 16)
#define HY_TOKEN_TILE_CASE(ringSize, tokenTile) \
    case ringSize:                              \
        return tokenTile##u;
uint32_t hybridTokenTileForRing(uint32_t ringSize)
{
    switch (ringSize) {
        FOR_EACH_HY_RING(HY_TOKEN_TILE_CASE)
        default:
            return 0u;
    }
}
#undef HY_TOKEN_TILE_CASE

constexpr uint32_t HY_CFG_NOEXP = 16u;
constexpr uint32_t HY_CFG_NOSYNC = 32u;
constexpr uint32_t HY_CFG_NOBODY = 64u;
constexpr uint32_t HY_CFG_WB = 128u;
constexpr uint32_t HY_CFG_WB2 = 256u;

constexpr size_t WORKSPACE_CACHE_MAX = 4;  // per device; entries are ~dim*width*16 elements each

struct WorkspaceCacheEntry {
    at::Tensor wexp;
    at::Tensor weightRef;  // strong ref: blocks allocator address reuse (guard 1)
    const void *weightPtr = nullptr;
    int64_t weightVersion = -1;
    at::ScalarType weightDtype = at::kFloat;
    uint32_t weightWidth = 0, weightDim = 0;
    int64_t wexpNumel = 0;
    void *filledOnStream = nullptr;  // stream that issued the fill; nullptr = not yet filled
    uint64_t lastUseTick = 0;
};

struct DeviceWorkspaceCache {
    std::vector<WorkspaceCacheEntry> entries;
    at::Tensor ring;  // pure scratch, one per device, grown to the largest request
    int64_t ringNumel = 0;
    at::ScalarType ringDtype = at::kFloat;
};

std::mutex &workspaceCacheMutex()
{
    static std::mutex *m = new std::mutex();  // intentionally leaked
    return *m;
}
std::map<int, DeviceWorkspaceCache> &workspaceCache()
{
    static std::map<int, DeviceWorkspaceCache> *c = new std::map<int, DeviceWorkspaceCache>();  // leaked
    return *c;
}

bool resolveHybridWorkspace(const at::Tensor &weight, const at::Tensor &x, uint32_t width, uint32_t dim,
                            int64_t wexpNumel, int64_t ringNumel, at::ScalarType dtype, void *stream,
                            at::Tensor &wexpOut, at::Tensor &ringOut)
{
    static uint64_t tick = 0;
    std::lock_guard<std::mutex> guard(workspaceCacheMutex());
    DeviceWorkspaceCache &cache = workspaceCache()[x.device().index()];

    WorkspaceCacheEntry *entry = nullptr;
    for (WorkspaceCacheEntry &candidate : cache.entries) {
        if (candidate.weightPtr == weight.const_data_ptr() && candidate.weightVersion == weight._version() &&
            candidate.weightDtype == weight.scalar_type() && candidate.weightWidth == width &&
            candidate.weightDim == dim && candidate.wexpNumel == wexpNumel) {
            entry = &candidate;
            break;
        }
    }
    if (entry == nullptr) {
        if (cache.entries.size() >= WORKSPACE_CACHE_MAX) {  // evict LRU
            size_t victim = 0;
            for (size_t i = 1; i < cache.entries.size(); ++i) {
                if (cache.entries[i].lastUseTick < cache.entries[victim].lastUseTick) victim = i;
            }
            cache.entries.erase(cache.entries.begin() + static_cast<ptrdiff_t>(victim));
        }
        WorkspaceCacheEntry fresh;
        fresh.wexp = at::empty({wexpNumel}, x.options());
        fresh.weightRef = weight;  // strong ref pins the storage -> address is unique to it
        fresh.weightPtr = weight.const_data_ptr();
        fresh.weightVersion = weight._version();
        fresh.weightDtype = weight.scalar_type();
        fresh.weightWidth = width;
        fresh.weightDim = dim;
        fresh.wexpNumel = wexpNumel;
        cache.entries.push_back(std::move(fresh));
        entry = &cache.entries.back();
    }
    entry->lastUseTick = ++tick;
    const bool needFill = entry->filledOnStream != stream;
    entry->filledOnStream = stream;  // claim it, so a same-stream follower does not refill

    if (!cache.ring.defined() || cache.ringNumel < ringNumel || cache.ringDtype != dtype) {
        cache.ring = at::zeros({ringNumel}, x.options());
        cache.ringNumel = ringNumel;
        cache.ringDtype = dtype;
    }
    wexpOut = entry->wexp;
    ringOut = cache.ring;
    return needFill;
}

struct LaunchPlanKey {
    uint32_t batch = 0, dim = 0, seqLen = 0, inputMode = 0xffu, width = 0, xrows = 0;
    int32_t dtype = -1;
    bool operator==(const LaunchPlanKey &o) const
    {
        return batch == o.batch && dim == o.dim && seqLen == o.seqLen && inputMode == o.inputMode &&
               width == o.width && xrows == o.xrows && dtype == o.dtype;
    }
};

struct LaunchPlan {
    LaunchPlanKey key;
    bool used = false;
    uint32_t chanTileWidth = 0, numChanTiles = 0, numTokenChunks = 0, blockDimConv = 0, blockDimWb = 0;
    bool hyEligible = false;
    uint32_t hyChanTileWidth = 0, hyTokenTileLen = 0, hyNumChanTiles = 0, hyNumTokenChunks = 0, blockDimHy = 0;
    uint32_t launchChanTileWidth = 0, launchNumChanTiles = 0, launchNumTokenChunks = 0;
    uint32_t pipelined = 0, cfg = 0, cfgFill = 0;
    bool useV2Cube = false, useWideChanTile = false, hoistExp = false, foldWb = false;
    int64_t wexpNumel = 0, ringNumel = 0;
};

struct DeviceGeometry {
    uint32_t coreNumAiv = 0, coreNumAic = 0;
    bool wideUb = false;
};
const DeviceGeometry &deviceGeometry()
{
    static const DeviceGeometry geometry = [] {
        DeviceGeometry g;
        auto platformInfo = platform_ascendc::PlatformAscendCManager::GetInstance();
        TORCH_CHECK(platformInfo != nullptr, "no AscendC platform");
        g.coreNumAiv = static_cast<uint32_t>(platformInfo->GetCoreNumAiv());
        g.coreNumAic = static_cast<uint32_t>(platformInfo->GetCoreNumAic());
        uint64_t ubBytes = 0;
        platformInfo->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
        constexpr uint64_t WIDE_UB_MIN_BYTES = 248u * 1024u;  // UB the wide table is sized for
        g.wideUb = (ubBytes >= WIDE_UB_MIN_BYTES);
        return g;
    }();
    return geometry;
}

constexpr size_t LAUNCH_PLAN_SLOTS = 8;

#ifndef HY_CUBE_MIN_ELEMS
#define HY_CUBE_MIN_ELEMS 0ull
#endif

LaunchPlan derivePlan(const LaunchPlanKey &key, uint32_t ringSize, const DeviceGeometry &geom)
{
    const uint32_t batch = key.batch;
    const uint32_t dim = key.dim;
    const uint32_t seqLen = key.seqLen;
    const uint32_t inputMode = key.inputMode;
    const uint32_t width = key.width;
    const uint32_t numAivCores = geom.coreNumAiv;

    LaunchPlan plan;
    plan.key = key;
    plan.used = true;
    const uint32_t avgSeqLen = (inputMode == 1) ? seqLen : std::max<uint32_t>(1u, key.xrows / batch);
    const uint32_t maxChannelsPerTile = maxTileWidthForRing(ringSize, geom.wideUb);  // UB-bound tile
    const auto [tileWidth, tokenChunks] =
        tiling_causal_conv1d(numAivCores, batch, dim, avgSeqLen, width, maxChannelsPerTile);
    plan.chanTileWidth = tileWidth;
    plan.numTokenChunks = tokenChunks;
    plan.numChanTiles = ceil_div(dim, plan.chanTileWidth);
    // conv has one task per (batch, channel tile, seq chunk); the writeback only
    plan.blockDimConv = std::min<uint32_t>(batch * plan.numChanTiles * plan.numTokenChunks, numAivCores);
    plan.blockDimWb = std::min<uint32_t>(batch * plan.numChanTiles, numAivCores);

    {
        const uint32_t numAicCores = geom.coreNumAic;
        const uint32_t cubeBits = 1u;
        plan.cfg = cubeBits | HY_CFG_NOEXP | HY_CFG_NOSYNC | HY_CFG_WB | HY_CFG_WB2;
        plan.hoistExp = true;
        plan.foldWb = true;
        plan.cfgFill = (plan.cfg & ~(HY_CFG_NOEXP | HY_CFG_NOSYNC | HY_CFG_WB | HY_CFG_WB2)) | HY_CFG_NOBODY;
        plan.useV2Cube = (plan.cfg & 1u) != 0u && isV2Ring(ringSize);
        const uint32_t wantChanTile = HY_CHAN_TILE_WIDE;
        plan.useWideChanTile =
            plan.useV2Cube && wantChanTile == HY_CHAN_TILE_WIDE && isWideChanTileRing(ringSize);
        plan.hyChanTileWidth = plan.useWideChanTile ? HY_CHAN_TILE_WIDE : HY_CHAN_TILE_DEFAULT;
        plan.hyTokenTileLen =
            plan.useWideChanTile ? HY_TOKEN_TILE_WIDE_CHAN : hybridTokenTileForRing(ringSize);
        plan.hyEligible = numAicCores > 0 && plan.hyTokenTileLen > 0 && (dim % DIM_ALIGN_ELEMS) == 0u;
        if (plan.hyEligible) {
            plan.hyNumChanTiles = ceil_div(dim, plan.hyChanTileWidth);
            auto [unusedChanTileWidth, tokenChunksRaw] =
                tiling_causal_conv1d(numAicCores, batch, dim, avgSeqLen, width, plan.hyChanTileWidth);
            (void)unusedChanTileWidth;  // always hyChanTileWidth by construction (min==max channels)
            const uint32_t maxTokenChunks = std::max<uint32_t>(1u, avgSeqLen / plan.hyTokenTileLen);
            plan.hyNumTokenChunks = std::min<uint32_t>(tokenChunksRaw, maxTokenChunks);
            constexpr uint64_t MIN_TILES_PER_AIC = 2ull;  // what the ring needs to overlap
            const uint64_t totalTokenTiles = static_cast<uint64_t>(batch) * plan.hyNumChanTiles *
                                             std::max<uint32_t>(1u, ceil_div(avgSeqLen, plan.hyTokenTileLen));
            if (totalTokenTiles <= MIN_TILES_PER_AIC * static_cast<uint64_t>(numAicCores)) {
                plan.hyNumTokenChunks = 1u;
            }
            if (0u != 0u) {
                plan.hyNumTokenChunks = std::max<uint32_t>(1u, 0u);
            }
            plan.blockDimHy =
                std::min<uint32_t>(batch * plan.hyNumChanTiles * plan.hyNumTokenChunks, numAicCores);
            plan.pipelined = 1u;

            plan.launchChanTileWidth = plan.hyChanTileWidth;
            plan.launchNumChanTiles = plan.hyNumChanTiles;
            plan.launchNumTokenChunks = plan.hyNumTokenChunks;

            if (0u != 0u) {
                plan.blockDimHy = std::min<uint32_t>(numAicCores, std::max<uint32_t>(1u, 0u));
            }
            plan.wexpNumel = static_cast<int64_t>(dim) * static_cast<int64_t>(width) * HY_WEXP_GROUP;
            plan.ringNumel = static_cast<int64_t>(plan.blockDimHy) * HY_RING_SLOTS * plan.hyTokenTileLen *
                             plan.hyChanTileWidth;
        }
    }
    return plan;
}

}  // namespace

HOST_API at::Tensor causal_conv1d_impl(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &conv_states,
                                       const at::Tensor &query_start_loc, const at::Tensor &cache_indices,
                                       const at::Tensor &has_initial_state, const at::Tensor &bias,
                                       bool activation_mode, int64_t pad_slot_id)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must be 2D [cu_seqlen, dim] or 3D [batch, seq_len, dim]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [width, dim], got shape ", weight.sizes());
    const uint32_t width = static_cast<uint32_t>(weight.size(0));
    TORCH_CHECK(isSupportedWidth(width), "Only filter widths 2..64 are supported, got ", weight.size(0));
    TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D [num_cache_lines, state_len, dim]");
    const at::ScalarType dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "Only BF16 and FP16 are supported, got ", dtype);
    TORCH_CHECK(weight.scalar_type() == dtype, "weight dtype must match x dtype");
    TORCH_CHECK(conv_states.scalar_type() == dtype, "conv_states dtype must match x dtype");
    TORCH_CHECK(query_start_loc.scalar_type() == at::kInt, "query_start_loc dtype must be int32");
    TORCH_CHECK(cache_indices.scalar_type() == at::kInt, "cache_indices dtype must be int32");
    TORCH_CHECK(has_initial_state.scalar_type() == at::kBool, "has_initial_state dtype must be bool");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && conv_states.is_contiguous(),
                "inputs must be contiguous");

    const bool has_bias = bias.numel() > 0;
    uint32_t inputMode, batch, seqLen, dim;
    if (x.dim() == 2) {
        inputMode = 0;
        dim = static_cast<uint32_t>(x.size(1));
        seqLen = 0;
        // Guard the size(0)-1 below: an empty/too-short qsl would underflow batch to ~4e9.
        TORCH_CHECK(query_start_loc.dim() == 1 && query_start_loc.size(0) >= 2,
                    "query_start_loc must be 1D and have at least 2 elements");
        batch = static_cast<uint32_t>(query_start_loc.size(0) - 1);
    } else {
        inputMode = 1;
        batch = static_cast<uint32_t>(x.size(0));
        seqLen = static_cast<uint32_t>(x.size(1));
        dim = static_cast<uint32_t>(x.size(2));
    }
    TORCH_CHECK(batch > 0 && dim > 0, "bad batch/dim");
    // cache_indices[seq] and has_initial_state[seq] are read per sequence in both layouts.
    TORCH_CHECK(cache_indices.dim() == 1 && cache_indices.size(0) >= static_cast<int64_t>(batch),
                "cache_indices must be 1D and have size >= batch");
    TORCH_CHECK(has_initial_state.dim() == 1 && has_initial_state.size(0) >= static_cast<int64_t>(batch),
                "has_initial_state must be 1D and have size >= batch");
    TORCH_CHECK(dim % DIM_ALIGN_ELEMS == 0, "dim must be multiple of 16 for fp16/bf16 alignment, but got ", dim);
    TORCH_CHECK(weight.size(1) == static_cast<int64_t>(dim), "weight.shape[1] must equal dim");
    TORCH_CHECK(conv_states.size(2) == static_cast<int64_t>(dim), "conv_states.shape[2] must equal dim");
    if (has_bias) {  // bias is read in the I/O dtype and cast to fp32 in the kernel
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == static_cast<int64_t>(dim), "bias must be 1D [dim]");
        TORCH_CHECK(bias.scalar_type() == dtype, "bias dtype must match x dtype");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
    const uint32_t stateLen = static_cast<uint32_t>(conv_states.size(1));
    TORCH_CHECK(stateLen >= width - 1, "state_len must be >= width-1");

    const DeviceGeometry &geom = deviceGeometry();
    const uint32_t numAivCores = geom.coreNumAiv;
    TORCH_CHECK(numAivCores > 0, "bad numAivCores");

    const uint32_t ringSize = roundUpToPow2(width);  // compile-time ring variant to launch

    LaunchPlanKey planKey;
    planKey.batch = batch;
    planKey.dim = dim;
    planKey.seqLen = seqLen;
    planKey.inputMode = inputMode;
    planKey.width = width;
    planKey.xrows = (inputMode == 1) ? 0u : static_cast<uint32_t>(x.size(0));
    planKey.dtype = static_cast<int32_t>(dtype);

    static thread_local LaunchPlan planSlots[LAUNCH_PLAN_SLOTS];
    static thread_local size_t planNext = 0;
    LaunchPlan *cached = nullptr;
    for (LaunchPlan &slot : planSlots) {
        if (slot.used && slot.key == planKey) {
            cached = &slot;
            break;
        }
    }
    if (cached == nullptr) {
        planSlots[planNext] = derivePlan(planKey, ringSize, geom);
        cached = &planSlots[planNext];
        planNext = (planNext + 1) % LAUNCH_PLAN_SLOTS;
    }
    const LaunchPlan &plan = *cached;
    const uint32_t chanTileWidth = plan.chanTileWidth;
    const uint32_t numChanTiles = plan.numChanTiles;
    const uint32_t numTokenChunks = plan.numTokenChunks;
    const uint32_t blockDimConv = plan.blockDimConv;
    const uint32_t blockDimWb = plan.blockDimWb;

    // weight/bias enter in the I/O dtype (fp16/bf16) and are cast to fp32 inside the
    // kernel; pass a native empty placeholder when there is no bias.
    const at::Tensor biasArg = has_bias ? bias : at::empty({0}, x.options());
    at::Tensor y = at::empty_like(x);

    const uint32_t actFlag = activation_mode ? 1u : 0u;
    const uint32_t biasFlag = has_bias ? 1u : 0u;
    const int32_t padSlot = static_cast<int32_t>(pad_slot_id);
    const bool isHalf = (dtype == at::kHalf);

#define LAUNCH_CONV_ENTRY(entry)                                                                          \
    EXEC_KERNEL_CMD(entry, blockDimConv, x, weight, biasArg, conv_states, query_start_loc, cache_indices, \
                    has_initial_state, y, dim, batch, inputMode, seqLen, stateLen, width, chanTileWidth,  \
                    numChanTiles, numTokenChunks, actFlag, biasFlag, padSlot)
#define LAUNCH_WB_ENTRY(entry)                                                                                 \
    EXEC_KERNEL_CMD(entry, blockDimWb, x, conv_states, query_start_loc, cache_indices, has_initial_state, dim, \
                    batch, inputMode, seqLen, stateLen, width, chanTileWidth, numChanTiles, padSlot)

    {
        const uint32_t cfg = plan.cfg;
        const uint32_t cfgFill = plan.cfgFill;
        const bool hoistExp = plan.hoistExp;
        const bool foldWb = plan.foldWb;
        const bool useV2Cube = plan.useV2Cube, useWideChanTile = plan.useWideChanTile;
        if (plan.hyEligible) {
            const uint32_t launchChanTileWidth = plan.launchChanTileWidth;
            const uint32_t launchNumChanTiles = plan.launchNumChanTiles;
            const uint32_t launchNumTokenChunks = plan.launchNumTokenChunks;
            const uint32_t blockDimHy = plan.blockDimHy;
            const uint32_t pipelined = plan.pipelined;
            const int64_t wexpNumel = plan.wexpNumel;
            const int64_t ringNumel = plan.ringNumel;
            at::Tensor wexp, ringBuf;
            bool wexpStale = true;
            if (hoistExp) {
                wexpStale = resolveHybridWorkspace(weight, x, width, dim, wexpNumel, ringNumel, dtype,
                                                   c10_npu::getCurrentNPUStream().stream(false), wexp, ringBuf);
            } else {
                wexp = at::empty({wexpNumel}, x.options());
                ringBuf = at::empty({ringNumel}, x.options());
            }
#define LAUNCH_HY_MIX(entry, cfgWord)                                                                          \
    EXEC_KERNEL_CMD(entry, blockDimHy, x, weight, biasArg, conv_states, query_start_loc, cache_indices,        \
                    has_initial_state, y, wexp, ringBuf, dim, batch, inputMode, seqLen, stateLen, width,       \
                    launchChanTileWidth, launchNumChanTiles, launchNumTokenChunks, actFlag, biasFlag, padSlot, \
                    pipelined, cfgWord)
#define LAUNCH_HY(convEntry)                   \
    do {                                       \
        if (hoistExp && wexpStale) {           \
            LAUNCH_HY_MIX(convEntry, cfgFill); \
        }                                      \
        LAUNCH_HY_MIX(convEntry, cfg);         \
    } while (0)
#define DISPATCH_HY_CASE(ringSize, maxTileWidth)                                                        \
    case ringSize:                                                                                      \
        if (isHalf) {                                                                                   \
            LAUNCH_HY(causal_conv1d_hy_rs##ringSize##_half); \
        } else {                                                                                        \
            LAUNCH_HY(causal_conv1d_hy_rs##ringSize##_bf16); \
        }                                                                                               \
        break;
#define DISPATCH_HY_V2_CASE(ringSize)                                                                      \
    case ringSize:                                                                                         \
        if (isHalf) {                                                                                      \
            LAUNCH_HY(causal_conv1d_hy_rs##ringSize##_half_v2); \
        } else {                                                                                           \
            LAUNCH_HY(causal_conv1d_hy_rs##ringSize##_bf16_v2); \
        }                                                                                                  \
        break;
#define DISPATCH_HY_WIDE_CASE(ringSize)                                 \
    case ringSize:                                                      \
        if (isHalf) {                                                   \
            LAUNCH_HY(causal_conv1d_hy_rs##ringSize##_half_v2d256);       \
        } else {                                                        \
            LAUNCH_HY(causal_conv1d_hy_rs##ringSize##_bf16_v2d256);       \
        }                                                               \
        break;
            if (useWideChanTile) {
                switch (ringSize) {
                    FOR_EACH_WIDE_CHAN_TILE_RING(DISPATCH_HY_WIDE_CASE)
                    default:
                        TORCH_CHECK(false, "causal_conv1d: no wide-channel-tile Cube entry for ring size ",
                                    ringSize);
                }
            } else if (useV2Cube) {
                switch (ringSize) {
                    FOR_EACH_V2_RING(DISPATCH_HY_V2_CASE)
                    default:
                        TORCH_CHECK(false, "causal_conv1d: no _v2 Cube entry for ring size ", ringSize);
                }
            } else {
                switch (ringSize) {
                    FOR_EACH_RING_SIZE(DISPATCH_HY_CASE)
                    default:
                        TORCH_CHECK(false, "causal_conv1d: no hybrid entry for ring size ", ringSize);
                }
            }
#undef DISPATCH_HY_WIDE_CASE
#undef DISPATCH_HY_V2_CASE
#undef DISPATCH_HY_CASE
#undef LAUNCH_HY
#undef LAUNCH_HY_MIX
            return y;
        }
    }

    TORCH_CHECK(false, "causal_conv1d: no Cube entry for this shape");
    return y;
}

}  // namespace npu_kernel
}  // namespace sglang
