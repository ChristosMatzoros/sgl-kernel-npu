# Standalone benchmark for causal_conv1d kernel — no PyTorch dependency.
#
# Builds the op_kernel side as an ascendc_library and links a C++ driver
# that uses ACLRT_LAUNCH_KERNEL directly (mirroring the pattern from
# bench/v2_polyvolver_1d/ in the parent convtest repo).
#
# Usage:
#   cmake -B build -DASCEND_CANN_PACKAGE_PATH=$ASCEND_HOME_PATH \
#                  -DCONV_DIM=4096 -DCONV_BATCH=2 -DCONV_SEQ=128 \
#                  -DCONV_WIDTH=4 -DCONV_DTYPE=half
#   cmake --build build -j
#   LD_LIBRARY_PATH=build/lib ./build/bench_causal_conv1d

cmake_minimum_required(VERSION 3.16)
project(bench_causal_conv1d LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

if(NOT DEFINED ENV{ASCEND_HOME_PATH} AND NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "ASCEND_HOME_PATH or ASCEND_CANN_PACKAGE_PATH must be set; source set_env.sh first")
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(SOC_VERSION "Ascend910B4")
set(RUN_MODE "npu")
set(ASCEND_PYTHON_EXECUTABLE python3)
list(APPEND CMAKE_MODULE_PATH "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
include("${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake")

# Build-time configuration (mirrors v2_polyvolver_1d convention).
if(NOT DEFINED CONV_DIM)
    set(CONV_DIM 4096)
endif()
if(NOT DEFINED CONV_BATCH)
    set(CONV_BATCH 2)
endif()
if(NOT DEFINED CONV_SEQ)
    set(CONV_SEQ 128)
endif()
if(NOT DEFINED CONV_WIDTH)
    set(CONV_WIDTH 4)
endif()
if(NOT DEFINED CONV_STATELEN)
    set(CONV_STATELEN 3)
endif()

message(STATUS "Building bench_causal_conv1d: dim=${CONV_DIM} batch=${CONV_BATCH} seq=${CONV_SEQ} width=${CONV_WIDTH}")

# Build the kernel library directly from sgl-kernel-npu's op_kernel sources.
# The causal_conv1d kernel is AIV-only — we add a dummy AIC stub so the
# ascendc_library merge step (which expects both AIC and AIV outputs) succeeds.
set(SGL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
ascendc_library(kernel_lib SHARED
    ${SGL_ROOT}/csrc/causal_conv1d/op_kernel/causal_conv1d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dummy_aic_stub.cpp
)
ascendc_compile_definitions(kernel_lib PRIVATE
    -DCCONV_BENCH=1
)
target_include_directories(kernel_lib PUBLIC
    ${SGL_ROOT}/csrc/causal_conv1d/op_kernel
)

# The C++ driver — pure CANN, no ATen.
add_executable(bench_causal_conv1d bench_causal_conv1d.cpp)
target_compile_definitions(bench_causal_conv1d PRIVATE
    CONV_DIM=${CONV_DIM}
    CONV_BATCH=${CONV_BATCH}
    CONV_SEQ=${CONV_SEQ}
    CONV_WIDTH=${CONV_WIDTH}
    CONV_STATELEN=${CONV_STATELEN}
)
target_include_directories(bench_causal_conv1d PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
    ${SGL_ROOT}/csrc/causal_conv1d/op_kernel
    ${SGL_ROOT}/csrc/causal_conv1d/op_host/tiling
)
target_link_directories(bench_causal_conv1d PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64
)
target_link_libraries(bench_causal_conv1d PRIVATE
    kernel_lib
    ascendcl
    stdc++
)
target_link_options(bench_causal_conv1d PRIVATE -Wl,--allow-shlib-undefined)
