// Dummy AIC-only kernel — needed because the ascendc_library cmake helper
// expects both AIC and AIV outputs to merge. Our actual benchmark kernel
// (causal_conv1d) is AIV-only, so without this stub the merge step fails
// with "device_aic.o not found".
//
// This kernel is never launched; it just gives the AIC build something
// to emit so the merge succeeds.

#define ASCENDC_CUBE_ONLY
#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void polyvolver_aic_stub_kernel(GM_ADDR unused)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    (void)unused;
#if defined(__DAV_C220_CUBE__)
    // do nothing
#endif
}
