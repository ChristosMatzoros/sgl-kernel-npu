// Standalone benchmark for sgl-kernel-npu's causal_conv1d kernel.
//
// No PyTorch / torch_npu dependency. Uses ACLRT_LAUNCH_KERNEL directly
// (same pattern as our bench/v2_polyvolver_1d/main.cpp).
//
// Purpose: measure real NPU kernel time before and after each PR
// modification (PR-1 sync refactor, PR-2 batching, ..., PR-6 Winograd).
// Establish a baseline on this 910B4, then quantify each PR's gain.
//
// Build/run:
//   cd bench_polyvolver
//   source /usr/local/Ascend/cann/set_env.sh
//   cmake -B build -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH
//   cmake --build build -j
//   LD_LIBRARY_PATH=build/lib ./build/bench_causal_conv1d

#include "acl/acl.h"
#include "aclrtlaunch_causal_conv1d_half.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_tiling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#define CHECK_ACL(expr)                                                       \
    do { aclError _e = (expr);                                                \
         if (_e != ACL_SUCCESS) {                                             \
             std::fprintf(stderr, "ACL %d at %s:%d: %s\n", _e, __FILE__,      \
                          __LINE__, #expr); std::exit(1); } } while (0)

namespace {

#ifndef CONV_DIM
#define CONV_DIM 4096
#endif
#ifndef CONV_BATCH
#define CONV_BATCH 2
#endif
#ifndef CONV_SEQ
#define CONV_SEQ 128
#endif
#ifndef CONV_WIDTH
#define CONV_WIDTH 4
#endif
#ifndef CONV_STATELEN
#define CONV_STATELEN 3  // width - 1 minimum
#endif

constexpr int DIM = CONV_DIM;
constexpr int BATCH = CONV_BATCH;
constexpr int SEQ = CONV_SEQ;
constexpr int WIDTH = CONV_WIDTH;
constexpr int STATELEN = CONV_STATELEN;
constexpr int N_ITER = 50;
constexpr int N_CACHE_LINES = BATCH;  // 1-1 mapping for the benchmark
constexpr int PAD_SLOT_ID = -1;

uint16_t F32toF16(float v) {
    uint32_t b; std::memcpy(&b, &v, 4);
    uint32_t s = (b >> 31) & 1;
    int32_t  e = ((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = b & 0x7FFFFF;
    if (e <= 0) return uint16_t(s << 15);
    if (e >= 31) return uint16_t((s << 15) | (0x1F << 10));
    return uint16_t((s << 15) | (e << 10) | (m >> 13));
}
float F16toF32(uint16_t v) {
    uint32_t s = (v >> 15) & 1, e = (v >> 10) & 0x1F, m = v & 0x3FF;
    uint32_t b;
    if (e == 0)        b = s << 31;
    else if (e == 31)  b = (s << 31) | (0xFF << 23) | (m << 13);
    else               b = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &b, 4);
    return f;
}

// CPU FP32 reference: per-batch, per-dim, per-token causal width-4 conv.
// x shape (B, S, D), w shape (W, D), conv_state (cache_lines, statelen, D),
// out shape (B, S, D). All FP16 storage, FP32 compute.
void reference_causal_conv1d_fp32(
    const uint16_t *x_fp16,        // (B, S, D)
    const uint16_t *w_fp16,        // (W, D)
    const uint16_t *bias_fp16,     // (D,) or nullptr
    const uint16_t *cs_fp16_in,    // (cache_lines, statelen, D) prior state
    uint16_t *out_fp16,            // (B, S, D)
    uint16_t *cs_fp16_out,         // (cache_lines, statelen, D) updated state
    const int32_t *cache_indices,  // (B,)
    const bool *has_initial_state, // (B,)
    bool activation)
{
    for (int b = 0; b < BATCH; ++b) {
        int cache_idx = cache_indices[b];
        if (cache_idx == PAD_SLOT_ID) continue;

        std::vector<float> hist(STATELEN * DIM, 0.0f);
        if (has_initial_state[b]) {
            for (int i = 0; i < STATELEN; ++i)
                for (int c = 0; c < DIM; ++c)
                    hist[i * DIM + c] = F16toF32(
                        cs_fp16_in[cache_idx * STATELEN * DIM + i * DIM + c]);
        }

        std::vector<float> x_ext((WIDTH - 1 + SEQ) * DIM, 0.0f);
        for (int i = 0; i < WIDTH - 1; ++i)
            for (int c = 0; c < DIM; ++c)
                x_ext[i * DIM + c] = hist[i * DIM + c];
        for (int t = 0; t < SEQ; ++t)
            for (int c = 0; c < DIM; ++c)
                x_ext[(t + WIDTH - 1) * DIM + c] =
                    F16toF32(x_fp16[b * SEQ * DIM + t * DIM + c]);

        for (int t = 0; t < SEQ; ++t) {
            for (int c = 0; c < DIM; ++c) {
                float acc = bias_fp16 ? F16toF32(bias_fp16[c]) : 0.0f;
                for (int j = 0; j < WIDTH; ++j) {
                    float w_v = F16toF32(w_fp16[j * DIM + c]);
                    // y[t,c] = Σ_j w[j,c] · x_ext[t + j, c]  (matches kernel)
                    float x_v = x_ext[(t + j) * DIM + c];
                    acc += w_v * x_v;
                }
                if (activation) {
                    acc = acc / (1.0f + std::exp(-acc));  // SiLU
                }
                out_fp16[b * SEQ * DIM + t * DIM + c] = F32toF16(acc);
            }
        }

        // Write back state: last (WIDTH-1) input positions
        for (int i = 0; i < WIDTH - 1; ++i)
            for (int c = 0; c < DIM; ++c)
                cs_fp16_out[cache_idx * STATELEN * DIM + i * DIM + c] =
                    F32toF16(x_ext[(SEQ + i) * DIM + c]);
    }
}

double compare_fp16(const uint16_t *a, const uint16_t *b, size_t n) {
    double mae = 0.0, maxabs = 0.0;
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        float av = F16toF32(a[i]), bv = F16toF32(b[i]);
        float e = std::fabs(av - bv);
        if (std::isfinite(e)) {
            mae = std::fmax(mae, e);
            maxabs = std::fmax(maxabs, std::fmax(std::fabs(av), std::fabs(bv)));
            ++valid;
        }
    }
    return mae / std::fmax(maxabs, 1e-6);
}

}  // namespace

int main() {
    std::printf("=== bench_causal_conv1d: dim=%d batch=%d seq=%d width=%d ===\n",
                DIM, BATCH, SEQ, WIDTH);
    static_assert(DIM % 16 == 0, "DIM must be multiple of 16");
    static_assert(WIDTH == 4, "current causal_conv1d hardcodes width=4");

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtContext ctx; CHECK_ACL(aclrtCreateContext(&ctx, 0));
    aclrtStream s; CHECK_ACL(aclrtCreateStream(&s));

    const size_t x_n = (size_t)BATCH * SEQ * DIM;
    const size_t w_n = (size_t)WIDTH * DIM;
    const size_t cs_n = (size_t)N_CACHE_LINES * STATELEN * DIM;

    // ---- Host data ----
    std::vector<uint16_t> xH(x_n), wH(w_n), csH(cs_n), outH(x_n);
    std::vector<uint16_t> outRef(x_n), csRef(cs_n);
    std::vector<int32_t> qsl(BATCH + 1), cidx(BATCH);
    std::vector<uint8_t> hasInit(BATCH);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : xH) v = F32toF16(dist(rng));
    for (auto &v : wH) v = F32toF16(dist(rng));
    for (auto &v : csH) v = F32toF16(dist(rng));
    qsl[0] = 0;
    for (int b = 0; b < BATCH; ++b) {
        qsl[b + 1] = qsl[b] + SEQ;
        cidx[b] = b;
        hasInit[b] = (b & 1);  // alternate has-init
    }

    // ---- Device alloc ----
    void *xD, *wD, *biasD, *csD, *qslD, *cidxD, *hasInitD, *outD, *wsD, *tilingD;
    CHECK_ACL(aclrtMalloc(&xD, x_n * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&wD, w_n * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&biasD, DIM * 2, ACL_MEM_MALLOC_HUGE_FIRST));  // 0-filled
    CHECK_ACL(aclrtMalloc(&csD, cs_n * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&qslD, (BATCH + 1) * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&cidxD, BATCH * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&hasInitD, BATCH, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&outD, x_n * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&wsD, 16 * 1024 * 1024, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&tilingD, sizeof(sglang::npu_kernel::CausalConv1dTilingData),
                          ACL_MEM_MALLOC_HUGE_FIRST));

    // Zero bias on device (we run no-bias mode for the smoke test)
    std::vector<uint16_t> biasZero(DIM, 0);
    CHECK_ACL(aclrtMemcpy(biasD, DIM * 2, biasZero.data(), DIM * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(wD, w_n * 2, wH.data(), w_n * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(qslD, (BATCH + 1) * 4, qsl.data(), (BATCH + 1) * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(cidxD, BATCH * 4, cidx.data(), BATCH * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(hasInitD, BATCH, hasInit.data(), BATCH,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- Tiling ----
    sglang::npu_kernel::CausalConv1dTilingData tilingData{};
    SGLang::CausalConv1d::ComputeTilingData(
        BATCH, BATCH * SEQ, SEQ, /*input_mode=*/1, DIM, WIDTH,
        STATELEN, N_CACHE_LINES, /*has_bias=*/false, /*activation=*/false,
        PAD_SLOT_ID, /*core_num=*/40, tilingData);
    CHECK_ACL(aclrtMemcpy(tilingD, sizeof(tilingData), &tilingData,
                          sizeof(tilingData), ACL_MEMCPY_HOST_TO_DEVICE));
    std::printf("tiling: dimTileSize=%ld blocksPerSeq=%ld\n",
                tilingData.dimTileSize, tilingData.blocksPerSeq);

    const int32_t grid = BATCH * (int32_t)tilingData.blocksPerSeq;
    const int32_t block_dim = std::min(grid, 40);

    // ---- One-shot correctness ----
    CHECK_ACL(aclrtMemcpy(xD, x_n * 2, xH.data(), x_n * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(csD, cs_n * 2, csH.data(), cs_n * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(ACLRT_LAUNCH_KERNEL(causal_conv1d_half)(
        block_dim, s,
        xD, wD, biasD, csD, qslD, cidxD, hasInitD, outD, wsD, tilingD));
    CHECK_ACL(aclrtMemcpy(outH.data(), x_n * 2, outD, x_n * 2,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    // Read back updated state for comparison
    std::vector<uint16_t> csUpdated(cs_n);
    CHECK_ACL(aclrtMemcpy(csUpdated.data(), cs_n * 2, csD, cs_n * 2,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    // std::vector<bool> doesn't expose .data() — use plain bool array.
    std::unique_ptr<bool[]> hasInitBool(new bool[BATCH]);
    for (int b = 0; b < BATCH; ++b) hasInitBool[b] = static_cast<bool>(hasInit[b]);
    reference_causal_conv1d_fp32(
        xH.data(), wH.data(), nullptr,
        csH.data(), outRef.data(), csRef.data(),
        cidx.data(), hasInitBool.get(), false);

    double rel_out = compare_fp16(outH.data(), outRef.data(), x_n);
    double rel_cs = compare_fp16(csUpdated.data(), csRef.data(), cs_n);
    std::printf("correctness: out rel_max=%.3e   state rel_max=%.3e   [%s]\n",
                rel_out, rel_cs,
                (rel_out < 5e-2 && rel_cs < 5e-2) ? "PASS" : "FAIL");

    // ---- Timing ----
    // Restore inputs (kernel mutates state)
    CHECK_ACL(aclrtMemcpy(csD, cs_n * 2, csH.data(), cs_n * 2, ACL_MEMCPY_HOST_TO_DEVICE));

    auto run = [&]() {
        CHECK_ACL(aclrtMemcpy(csD, cs_n * 2, csH.data(), cs_n * 2,
                              ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(ACLRT_LAUNCH_KERNEL(causal_conv1d_half)(
            block_dim, s,
            xD, wD, biasD, csD, qslD, cidxD, hasInitD, outD, wsD, tilingD));
        CHECK_ACL(aclrtSynchronizeStream(s));
    };
    run();  // warm
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITER; ++i) run();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_ITER;

    std::printf("kernel time: %.2f us/iter (median over %d)\n", us, N_ITER);

    // ---- Cleanup ----
    aclrtFree(xD); aclrtFree(wD); aclrtFree(biasD); aclrtFree(csD);
    aclrtFree(qslD); aclrtFree(cidxD); aclrtFree(hasInitD); aclrtFree(outD);
    aclrtFree(wsD); aclrtFree(tilingD);
    aclrtDestroyStream(s);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
