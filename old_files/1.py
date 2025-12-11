import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# ---- C++ stub: declare the function so load_inline can bind it ----
gemv_cpp = r"""
#include <torch/extension.h>

// Forward declaration so PyTorch can bind it (definition is in the CUDA source).
torch::Tensor cuda_nvfp4_gemv(torch::Tensor A,
                            torch::Tensor B,
                            torch::Tensor C,
                            torch::Tensor SFA,
                            torch::Tensor SFB);
"""

# ---- CUDA source: struct, kernel, launcher, and Python-facing wrapper ----
gemv_cuda = r"""
#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp4.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ---- gemv.h ----
struct Gemv_params {
    using index_t = uint64_t;

    int b, m, k, real_k;

    void *__restrict__ a_ptr;
    void *__restrict__ b_ptr;
    void *__restrict__ sfa_ptr;
    void *__restrict__ sfb_ptr;
    void *__restrict__ o_ptr;

    index_t a_batch_stride;
    index_t b_batch_stride;
    index_t sfa_batch_stride;
    index_t sfb_batch_stride;
    index_t o_batch_stride;

    index_t a_row_stride;
    index_t b_row_stride;
    index_t sfa_row_stride;
    index_t sfb_row_stride;
    index_t o_row_stride;
};

static constexpr int BLOCK_SIZE = 128; // 128

// ---- load helpers ----
__device__ __forceinline__ void load_block_16x2fp4_generic(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]),
          "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    asm volatile(
        "ld.global.u16 %0, [%2];\n\t"
        "ld.global.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// k = 3584
__device__ __forceinline__ void load_block_16x2fp4_k3584(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.cs.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.L2::128B.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]),
          "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    asm volatile(
        "ld.global.cs.u16 %0, [%2];\n\t"
        "ld.global.L2::128B.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// k = 8192
__device__ __forceinline__ void load_block_16x2fp4_k8192(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.cs.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]),
          "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    asm volatile(
        "ld.global.lu.u16 %0, [%2];\n\t"
        "ld.global.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// k = 1024
__device__ __forceinline__ void load_block_16x2fp4_k1024(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);
    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base);

    asm volatile(
        "ld.global.cs.u64.v2 {%0, %1}, [%4];\n\t"
        "ld.global.u64.v2 {%2, %3}, [%5];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]),
          "=l"(b_regs[0]), "=l"(b_regs[1])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    asm volatile(
        "ld.global.cs.u16 %0, [%2];\n\t"
        "ld.global.u16 %1, [%3];\n\t"
        : "=h"(sfa_regs), "=h"(sfb_regs)
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

// Compile-time dispatcher
template<int K>
__device__ __forceinline__ void load_block_16x2fp4(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[2],
    uint64_t (&b_regs)[2],
    uint16_t &sfa_regs,
    uint16_t &sfb_regs)
{
    if constexpr (K == 3584) {
        load_block_16x2fp4_k3584(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    } else if constexpr (K == 8192) {
        load_block_16x2fp4_k8192(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    } else if constexpr (K == 1024) {
        load_block_16x2fp4_k1024(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    } else {
        // generic / fallback
        load_block_16x2fp4_generic(
            rowA, vecB, rowS_u16, vecS_u16,
            elem_base, block_base, a_regs, b_regs, sfa_regs, sfb_regs);
    }
}

__device__ __forceinline__ void load_block_32x2fp4(
    const __nv_fp4x2_e2m1* rowA,
    const __nv_fp4x2_e2m1* vecB,
    const uint16_t*        rowS_u16,
    const uint16_t*        vecS_u16,
    int                    elem_base,
    int                    block_base,
    uint64_t (&a_regs)[4],
    uint64_t (&b_regs)[4],
    uint16_t (&sfa_regs)[2],
    uint16_t (&sfb_regs)[2])
{
    uint64_t rowA_addr = reinterpret_cast<uint64_t>(rowA + elem_base);
    uint64_t vecB_addr = reinterpret_cast<uint64_t>(vecB + elem_base);

    asm volatile(
        "ld.global.L1::no_allocate.L2::evict_first.L2::256B.v4.u64 {%0, %1, %2, %3}, [%8];\n\t"
        "ld.global.L1::evict_last.L2::evict_last.v4.u64 {%4, %5, %6, %7}, [%9];\n\t"
        : "=l"(a_regs[0]), "=l"(a_regs[1]), "=l"(a_regs[2]), "=l"(a_regs[3]),
          "=l"(b_regs[0]), "=l"(b_regs[1]), "=l"(b_regs[2]), "=l"(b_regs[3])
        : "l"(rowA_addr), "l"(vecB_addr)
    );

    uint64_t rowS_addr = reinterpret_cast<uint64_t>(rowS_u16 + block_base * 2);
    uint64_t vecS_addr = reinterpret_cast<uint64_t>(vecS_u16 + block_base * 2);

    asm volatile(
        "ld.global.L1::no_allocate.v2.u16 {%0, %1}, [%4];\n\t"
        "ld.global.L1::evict_last.v2.u16 {%2, %3}, [%5];\n\t"
        : "=h"(sfa_regs[0]), "=h"(sfa_regs[1]),
          "=h"(sfb_regs[0]), "=h"(sfb_regs[1])
        : "l"(rowS_addr), "l"(vecS_addr)
    );
}

__device__ __forceinline__ __half block_scaled_fma_32x2fp4(
    const uint64_t (&a_regs)[4],
    const uint64_t (&b_regs)[4],
    const uint16_t (&sfa_regs)[2],
    const uint16_t (&sfb_regs)[2])
{
    const uint32_t* a_regs_packed = reinterpret_cast<const uint32_t*>(&a_regs);
    const uint32_t* b_regs_packed = reinterpret_cast<const uint32_t*>(&b_regs);

    uint16_t out_half_bits;

    asm volatile(
        "{\n"
        // --- Register declarations ---
        ".reg .b8 a0, a1, a2, a3, a4, a5, a6, a7;\n"
        ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
        ".reg .f16x2 sfa_f16x2_0, sfa_f16x2_1, sfb_f16x2_0, sfb_f16x2_1;\n"
        ".reg .f16x2 sf_f16x2_0, sf_f16x2_1;\n"
        ".reg .f16x2 scale0, scale1, scale2, scale3;\n"
        ".reg .f16x2 accum_total, accum_group;\n"
        ".reg .f16x2 cvt_a0, cvt_a1, cvt_a2, cvt_a3, cvt_a4, cvt_a5, cvt_a6, cvt_a7;\n"
        ".reg .f16x2 cvt_b0, cvt_b1, cvt_b2, cvt_b3, cvt_b4, cvt_b5, cvt_b6, cvt_b7;\n"
        ".reg .f16 lane0, lane1, result_f16;\n"

        // --- Prepare 4 scales from the u16[2] inputs ---
        "cvt.rn.f16x2.e4m3x2 sfa_f16x2_0, %17;\n" // sfa_regs[0]
        "cvt.rn.f16x2.e4m3x2 sfa_f16x2_1, %18;\n" // sfa_regs[1]
        "cvt.rn.f16x2.e4m3x2 sfb_f16x2_0, %19;\n" // sfb_regs[0]
        "cvt.rn.f16x2.e4m3x2 sfb_f16x2_1, %20;\n" // sfb_regs[1]

        "mul.rn.f16x2 sf_f16x2_0, sfa_f16x2_0, sfb_f16x2_0;\n"
        "mul.rn.f16x2 sf_f16x2_1, sfa_f16x2_1, sfb_f16x2_1;\n"

        "mov.b32 {lane0, lane1}, sf_f16x2_0;\n"
        "mov.b32 scale0, {lane0, lane0};\n"
        "mov.b32 scale1, {lane1, lane1};\n"
        "mov.b32 {lane0, lane1}, sf_f16x2_1;\n"
        "mov.b32 scale2, {lane0, lane0};\n"
        "mov.b32 scale3, {lane1, lane1};\n"
        "mov.b32 accum_total, 0;\n"

        // --- FMA Block 0 (uses scale0) ---
        "mov.b32 {a0,a1,a2,a3}, %1; mov.b32 {a4,a5,a6,a7}, %2;\n"
        "mov.b32 {b0,b1,b2,b3}, %9; mov.b32 {b4,b5,b6,b7}, %10;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_a0,a0; cvt.rn.f16x2.e2m1x2 cvt_a1,a1; cvt.rn.f16x2.e2m1x2 cvt_a2,a2; cvt.rn.f16x2.e2m1x2 cvt_a3,a3; cvt.rn.f16x2.e2m1x2 cvt_a4,a4; cvt.rn.f16x2.e2m1x2 cvt_a5,a5; cvt.rn.f16x2.e2m1x2 cvt_a6,a6; cvt.rn.f16x2.e2m1x2 cvt_a7,a7;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_b0,b0; cvt.rn.f16x2.e2m1x2 cvt_b1,b1; cvt.rn.f16x2.e2m1x2 cvt_b2,b2; cvt.rn.f16x2.e2m1x2 cvt_b3,b3; cvt.rn.f16x2.e2m1x2 cvt_b4,b4; cvt.rn.f16x2.e2m1x2 cvt_b5,b5; cvt.rn.f16x2.e2m1x2 cvt_b6,b6; cvt.rn.f16x2.e2m1x2 cvt_b7,b7;\n"
        "mov.b32 accum_group,0; fma.rn.f16x2 accum_group,cvt_a0,cvt_b0,accum_group; fma.rn.f16x2 accum_group,cvt_a1,cvt_b1,accum_group; fma.rn.f16x2 accum_group,cvt_a2,cvt_b2,accum_group; fma.rn.f16x2 accum_group,cvt_a3,cvt_b3,accum_group; fma.rn.f16x2 accum_group,cvt_a4,cvt_b4,accum_group; fma.rn.f16x2 accum_group,cvt_a5,cvt_b5,accum_group; fma.rn.f16x2 accum_group,cvt_a6,cvt_b6,accum_group; fma.rn.f16x2 accum_group,cvt_a7,cvt_b7,accum_group;\n"
        "mul.rn.f16x2 accum_group,scale0,accum_group; add.rn.f16x2 accum_total,accum_total,accum_group;\n"

        // --- FMA Block 1 (uses scale1) ---
        "mov.b32 {a0,a1,a2,a3}, %3; mov.b32 {a4,a5,a6,a7}, %4;\n"
        "mov.b32 {b0,b1,b2,b3}, %11; mov.b32 {b4,b5,b6,b7}, %12;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_a0,a0; cvt.rn.f16x2.e2m1x2 cvt_a1,a1; cvt.rn.f16x2.e2m1x2 cvt_a2,a2; cvt.rn.f16x2.e2m1x2 cvt_a3,a3; cvt.rn.f16x2.e2m1x2 cvt_a4,a4; cvt.rn.f16x2.e2m1x2 cvt_a5,a5; cvt.rn.f16x2.e2m1x2 cvt_a6,a6; cvt.rn.f16x2.e2m1x2 cvt_a7,a7;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_b0,b0; cvt.rn.f16x2.e2m1x2 cvt_b1,b1; cvt.rn.f16x2.e2m1x2 cvt_b2,b2; cvt.rn.f16x2.e2m1x2 cvt_b3,b3; cvt.rn.f16x2.e2m1x2 cvt_b4,b4; cvt.rn.f16x2.e2m1x2 cvt_b5,b5; cvt.rn.f16x2.e2m1x2 cvt_b6,b6; cvt.rn.f16x2.e2m1x2 cvt_b7,b7;\n"
        "mov.b32 accum_group,0; fma.rn.f16x2 accum_group,cvt_a0,cvt_b0,accum_group; fma.rn.f16x2 accum_group,cvt_a1,cvt_b1,accum_group; fma.rn.f16x2 accum_group,cvt_a2,cvt_b2,accum_group; fma.rn.f16x2 accum_group,cvt_a3,cvt_b3,accum_group; fma.rn.f16x2 accum_group,cvt_a4,cvt_b4,accum_group; fma.rn.f16x2 accum_group,cvt_a5,cvt_b5,accum_group; fma.rn.f16x2 accum_group,cvt_a6,cvt_b6,accum_group; fma.rn.f16x2 accum_group,cvt_a7,cvt_b7,accum_group;\n"
        "mul.rn.f16x2 accum_group,scale1,accum_group; add.rn.f16x2 accum_total,accum_total,accum_group;\n"


        // --- FMA Block 2 (uses scale2) ---
        "mov.b32 {a0,a1,a2,a3}, %5; mov.b32 {a4,a5,a6,a7}, %6;\n"
        "mov.b32 {b0,b1,b2,b3}, %13; mov.b32 {b4,b5,b6,b7}, %14;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_a0,a0; cvt.rn.f16x2.e2m1x2 cvt_a1,a1; cvt.rn.f16x2.e2m1x2 cvt_a2,a2; cvt.rn.f16x2.e2m1x2 cvt_a3,a3; cvt.rn.f16x2.e2m1x2 cvt_a4,a4; cvt.rn.f16x2.e2m1x2 cvt_a5,a5; cvt.rn.f16x2.e2m1x2 cvt_a6,a6; cvt.rn.f16x2.e2m1x2 cvt_a7,a7;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_b0,b0; cvt.rn.f16x2.e2m1x2 cvt_b1,b1; cvt.rn.f16x2.e2m1x2 cvt_b2,b2; cvt.rn.f16x2.e2m1x2 cvt_b3,b3; cvt.rn.f16x2.e2m1x2 cvt_b4,b4; cvt.rn.f16x2.e2m1x2 cvt_b5,b5; cvt.rn.f16x2.e2m1x2 cvt_b6,b6; cvt.rn.f16x2.e2m1x2 cvt_b7,b7;\n"
        "mov.b32 accum_group,0; fma.rn.f16x2 accum_group,cvt_a0,cvt_b0,accum_group; fma.rn.f16x2 accum_group,cvt_a1,cvt_b1,accum_group; fma.rn.f16x2 accum_group,cvt_a2,cvt_b2,accum_group; fma.rn.f16x2 accum_group,cvt_a3,cvt_b3,accum_group; fma.rn.f16x2 accum_group,cvt_a4,cvt_b4,accum_group; fma.rn.f16x2 accum_group,cvt_a5,cvt_b5,accum_group; fma.rn.f16x2 accum_group,cvt_a6,cvt_b6,accum_group; fma.rn.f16x2 accum_group,cvt_a7,cvt_b7,accum_group;\n"
        "mul.rn.f16x2 accum_group,scale2,accum_group; add.rn.f16x2 accum_total,accum_total,accum_group;\n"

        // --- FMA Block 3 (uses scale3) ---
        "mov.b32 {a0,a1,a2,a3}, %7; mov.b32 {a4,a5,a6,a7}, %8;\n"
        "mov.b32 {b0,b1,b2,b3}, %15; mov.b32 {b4,b5,b6,b7}, %16;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_a0,a0; cvt.rn.f16x2.e2m1x2 cvt_a1,a1; cvt.rn.f16x2.e2m1x2 cvt_a2,a2; cvt.rn.f16x2.e2m1x2 cvt_a3,a3; cvt.rn.f16x2.e2m1x2 cvt_a4,a4; cvt.rn.f16x2.e2m1x2 cvt_a5,a5; cvt.rn.f16x2.e2m1x2 cvt_a6,a6; cvt.rn.f16x2.e2m1x2 cvt_a7,a7;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_b0,b0; cvt.rn.f16x2.e2m1x2 cvt_b1,b1; cvt.rn.f16x2.e2m1x2 cvt_b2,b2; cvt.rn.f16x2.e2m1x2 cvt_b3,b3; cvt.rn.f16x2.e2m1x2 cvt_b4,b4; cvt.rn.f16x2.e2m1x2 cvt_b5,b5; cvt.rn.f16x2.e2m1x2 cvt_b6,b6; cvt.rn.f16x2.e2m1x2 cvt_b7,b7;\n"
        "mov.b32 accum_group,0; fma.rn.f16x2 accum_group,cvt_a0,cvt_b0,accum_group; fma.rn.f16x2 accum_group,cvt_a1,cvt_b1,accum_group; fma.rn.f16x2 accum_group,cvt_a2,cvt_b2,accum_group; fma.rn.f16x2 accum_group,cvt_a3,cvt_b3,accum_group; fma.rn.f16x2 accum_group,cvt_a4,cvt_b4,accum_group; fma.rn.f16x2 accum_group,cvt_a5,cvt_b5,accum_group; fma.rn.f16x2 accum_group,cvt_a6,cvt_b6,accum_group; fma.rn.f16x2 accum_group,cvt_a7,cvt_b7,accum_group;\n"
        "mul.rn.f16x2 accum_group,scale3,accum_group; add.rn.f16x2 accum_total,accum_total,accum_group;\n"

        // --- Final reduction ---
        "mov.b32 {lane0, lane1}, accum_total;\n"
        "add.rn.f16 result_f16, lane0, lane1;\n"
        "mov.b16 %0, result_f16;\n"
        "}\n"
        : "=h"(out_half_bits)
        : "r"(a_regs_packed[0]), "r"(a_regs_packed[1]), "r"(a_regs_packed[2]), "r"(a_regs_packed[3]),
          "r"(a_regs_packed[4]), "r"(a_regs_packed[5]), "r"(a_regs_packed[6]), "r"(a_regs_packed[7]),
          "r"(b_regs_packed[0]), "r"(b_regs_packed[1]), "r"(b_regs_packed[2]), "r"(b_regs_packed[3]),
          "r"(b_regs_packed[4]), "r"(b_regs_packed[5]), "r"(b_regs_packed[6]), "r"(b_regs_packed[7]),
          "h"(sfa_regs[0]), "h"(sfa_regs[1]), "h"(sfb_regs[0]), "h"(sfb_regs[1])
        : "memory"
    );

    union { uint16_t u; __half h; } conv;
    conv.u = out_half_bits;
    return conv.h;
}


__device__ __forceinline__ float block_scaled_fma_16x2fp4(
    const uint64_t (&a_regs)[2],
    const uint64_t (&b_regs)[2],
    uint16_t       sfa_regs,
    uint16_t       sfb_regs)
{
    uint32_t const* a_regs_packed = reinterpret_cast<uint32_t const*>(&a_regs);
    uint32_t const* b_regs_packed = reinterpret_cast<uint32_t const*>(&b_regs);

    float out_f32;

    asm volatile(
        "{\n"
        // 8 bytes of A and B at a time (reused for upper half)
        ".reg .b8 a0_0, a0_1, a0_2, a0_3;\n"
        ".reg .b8 a0_4, a0_5, a0_6, a0_7;\n"
        ".reg .b8 b0_0, b0_1, b0_2, b0_3;\n"
        ".reg .b8 b0_4, b0_5, b0_6, b0_7;\n"

        // scales and accumulators
        ".reg .f16x2 sfa_f16x2, sfb_f16x2, sf_f16x2;\n"
        ".reg .f16x2 scale0_f16x2, scale1_f16x2;\n"
        ".reg .f16x2 accum_total, accum_group;\n"

        // converted fp4 -> f16x2 (only 8 per vector kept live)
        ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\n"
        ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\n"
        ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\n"
        ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\n"

        ".reg .f16 lane0, lane1, result_f16;\n"
        ".reg .f32 result_f32;\n"

        // scales
        "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %5;\n"
        "cvt.rn.f16x2.e4m3x2 sfb_f16x2, %6;\n"
        "mul.rn.f16x2 sf_f16x2, sfa_f16x2, sfb_f16x2;\n"
        "mov.b32 {lane0, lane1}, sf_f16x2;\n"
        "mov.b32 scale0_f16x2, {lane0, lane0};\n"
        "mov.b32 scale1_f16x2, {lane1, lane1};\n"

        "mov.b32 accum_total, 0;\n"

        //----------------------------------------------------------------------
        // First 8×(2×FP4) -> uses scale0
        //----------------------------------------------------------------------
        "mov.b32 {a0_0, a0_1, a0_2, a0_3}, %1;\n"
        "mov.b32 {a0_4, a0_5, a0_6, a0_7}, %2;\n"
        "mov.b32 {b0_0, b0_1, b0_2, b0_3}, %3;\n"
        "mov.b32 {b0_4, b0_5, b0_6, b0_7}, %4;\n"

        // all conversions first
        "cvt.rn.f16x2.e2m1x2 cvt_0_0, a0_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_0, b0_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_1, a0_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_1, b0_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_2, a0_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_2, b0_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_3, a0_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_3, b0_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_4, a0_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_4, b0_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_5, a0_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_5, b0_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_6, a0_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_6, b0_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_7, a0_7;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_7, b0_7;\n"

        // then all FMAs into one group accumulator
        "mov.b32 accum_group, 0;\n"
        "fma.rn.f16x2 accum_group, cvt_0_0, cvt_1_0, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_1, cvt_1_1, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_2, cvt_1_2, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_3, cvt_1_3, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_4, cvt_1_4, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_5, cvt_1_5, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_6, cvt_1_6, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_7, cvt_1_7, accum_group;\n"

        "mul.rn.f16x2 accum_group, scale0_f16x2, accum_group;\n"
        "add.rn.f16x2 accum_total, accum_total, accum_group;\n"

        //----------------------------------------------------------------------
        // Second 8×(2×FP4) -> uses scale1, reusing all the same regs
        //----------------------------------------------------------------------
        "mov.b32 {a0_0, a0_1, a0_2, a0_3}, %7;\n"
        "mov.b32 {a0_4, a0_5, a0_6, a0_7}, %8;\n"
        "mov.b32 {b0_0, b0_1, b0_2, b0_3}, %9;\n"
        "mov.b32 {b0_4, b0_5, b0_6, b0_7}, %10;\n"

        // conversions for upper half
        "cvt.rn.f16x2.e2m1x2 cvt_0_0, a0_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_0, b0_0;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_1, a0_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_1, b0_1;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_2, a0_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_2, b0_2;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_3, a0_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_3, b0_3;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_4, a0_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_4, b0_4;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_5, a0_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_5, b0_5;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_6, a0_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_6, b0_6;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_0_7, a0_7;\n"
        "cvt.rn.f16x2.e2m1x2 cvt_1_7, b0_7;\n"

        "mov.b32 accum_group, 0;\n"
        "fma.rn.f16x2 accum_group, cvt_0_0, cvt_1_0, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_1, cvt_1_1, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_2, cvt_1_2, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_3, cvt_1_3, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_4, cvt_1_4, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_5, cvt_1_5, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_6, cvt_1_6, accum_group;\n"
        "fma.rn.f16x2 accum_group, cvt_0_7, cvt_1_7, accum_group;\n"

        "mul.rn.f16x2 accum_group, scale1_f16x2, accum_group;\n"
        "add.rn.f16x2 accum_total, accum_total, accum_group;\n"

        // final reduction to scalar f16 -> then upconvert to f32
        "mov.b32 {lane0, lane1}, accum_total;\n"
        "add.rn.f16 result_f16, lane0, lane1;\n"
        "cvt.f32.f16 result_f32, result_f16;\n"
        "mov.b32 %0, result_f32;\n"

        "}\n"
        : "=f"(out_f32)
        : "r"(a_regs_packed[0]), "r"(a_regs_packed[1]),
          "r"(b_regs_packed[0]), "r"(b_regs_packed[1]),
          "h"(sfa_regs), "h"(sfb_regs),
          "r"(a_regs_packed[2]), "r"(a_regs_packed[3]),
          "r"(b_regs_packed[2]), "r"(b_regs_packed[3])
        : "memory"
    );

    return out_f32;
}

template <int ROWS_PER_BLOCK, int THREADS_PER_ROW, int ITERS, int K_SPECIAL, bool USE_32X2>
__global__ void __launch_bounds__(ROWS_PER_BLOCK * THREADS_PER_ROW, 8)
gemv_kernel(const __grid_constant__ Gemv_params params)
{
    const int tid   = threadIdx.x;
    const int rib   = tid / THREADS_PER_ROW;
    const int lane  = tid % THREADS_PER_ROW;
    const int batch = blockIdx.z;
    const int row   = blockIdx.x * ROWS_PER_BLOCK + rib;

    const size_t A_batch_base   = static_cast<size_t>(batch) * params.a_batch_stride;
    const size_t SFA_batch_base = static_cast<size_t>(batch) * params.sfa_batch_stride;
    const size_t B_batch_base   = static_cast<size_t>(batch) * params.b_batch_stride;
    const size_t SFB_batch_base = static_cast<size_t>(batch) * params.sfb_batch_stride;
    const size_t C_batch_base   = static_cast<size_t>(batch) * params.o_batch_stride;

    const __nv_fp4x2_e2m1* rowA = static_cast<const __nv_fp4x2_e2m1*>(params.a_ptr) + A_batch_base   + row * params.a_row_stride;
    const __nv_fp8_e4m3*   rowS = static_cast<const __nv_fp8_e4m3*>(params.sfa_ptr) + SFA_batch_base + row * params.sfa_row_stride;
    const __nv_fp4x2_e2m1* vecB = static_cast<const __nv_fp4x2_e2m1*>(params.b_ptr) + B_batch_base;
    const __nv_fp8_e4m3*   vecS = static_cast<const __nv_fp8_e4m3*>(params.sfb_ptr) + SFB_batch_base;

    const uint16_t* rowS_u16 = reinterpret_cast<const uint16_t*>(rowS);
    const uint16_t* vecS_u16 = reinterpret_cast<const uint16_t*>(vecS);

    float sum = 0.f;

    if constexpr (USE_32X2) {
        // ---- old gemv_kernel_v2 body ----
        #pragma unroll
        for (int idx = 0; idx < 2; ++idx) {   // 8192-special: 2 iters of 32 * 128 = 8192
            int block_base = idx * THREADS_PER_ROW + lane;
            int elem_base  = block_base * 32;

            uint64_t a_regs[4], b_regs[4];
            uint16_t sfa_regs[2], sfb_regs[2];

            load_block_32x2fp4(
                rowA, vecB,
                rowS_u16, vecS_u16,
                elem_base, block_base,
                a_regs, b_regs,
                sfa_regs, sfb_regs);

            __half h = block_scaled_fma_32x2fp4(a_regs, b_regs, sfa_regs, sfb_regs);
            sum += __half2float(h);
        }

        __shared__ float sdata[THREADS_PER_ROW];
        sdata[lane] = sum;
        __syncthreads();

        // THREADS_PER_ROW == 128 here
        if (tid < 64) sdata[lane] += sdata[lane + 64];
        __syncthreads();

        if (lane < 32) {
            float val = sdata[lane] + sdata[lane + 32];
            val += __shfl_down_sync(0xffffffff, val, 16);
            val += __shfl_down_sync(0xffffffff, val, 8);
            val += __shfl_down_sync(0xffffffff, val, 4);
            val += __shfl_down_sync(0xffffffff, val, 2);
            val += __shfl_down_sync(0xffffffff, val, 1);

            if (lane == 0) {
                __half* out = (__half*)params.o_ptr + C_batch_base + row;
                out[0] = __float2half(val);
            }
        }
    } else {
        // ---- old gemv_kernel body ----
        auto body = [&](int idx) {
            int block_base = idx * THREADS_PER_ROW + lane;
            int elem_base  = block_base * 16;

            uint64_t a_regs[2], b_regs[2];
            uint16_t sfa_regs, sfb_regs;

            load_block_16x2fp4<K_SPECIAL>(
                rowA, vecB,
                rowS_u16, vecS_u16,
                elem_base, block_base,
                a_regs, b_regs,
                sfa_regs, sfb_regs);
            sum += block_scaled_fma_16x2fp4(a_regs, b_regs, sfa_regs, sfb_regs);
        };

        if constexpr (ITERS > 0) {
            #pragma unroll
            for (int idx = 0; idx < ITERS; ++idx) {
                body(idx);
            }
        } else {
            int iters = params.k / (THREADS_PER_ROW * 16);
            for (int idx = 0; idx < iters; ++idx) {
                body(idx);
            }
        }

        #pragma unroll
        for (int offset = THREADS_PER_ROW / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset, THREADS_PER_ROW);
        }

        if (lane == 0) {
            __half* out = (__half*)params.o_ptr + C_batch_base + row;
            out[0] = __float2half(sum);
        }
    }
}


torch::Tensor cuda_nvfp4_gemv(torch::Tensor A,
                            torch::Tensor B,
                            torch::Tensor C,
                            torch::Tensor SFA,
                            torch::Tensor SFB)
{

    const auto sizes = A.sizes();
    const int M = sizes[0];
    const int K = sizes[1];
    const int L = sizes[2];

    Gemv_params params{};
    params.b = L;
    params.m = M;
    params.k = K;

    params.a_ptr  = A.data_ptr();
    params.b_ptr  = B.data_ptr();
    params.sfa_ptr= SFA.data_ptr();
    params.sfb_ptr= SFB.data_ptr();
    params.o_ptr  = C.data_ptr();

    params.a_batch_stride  = A.stride(2);
    params.b_batch_stride  = B.stride(2);
    params.sfa_batch_stride= SFA.stride(2);
    params.sfb_batch_stride= SFB.stride(2);
    params.o_batch_stride  = C.stride(2);

    params.a_row_stride  = A.stride(0);
    params.b_row_stride  = B.stride(0);
    params.sfa_row_stride= SFA.stride(0);
    params.sfb_row_stride= SFB.stride(0);
    params.o_row_stride  = C.stride(0);

    if (params.k <= 256) {  // <= 512 FP4 values
        dim3 grid(params.m / 16, 1, params.b);
        dim3 block(128, 1, 1);
        // generic loader (K_SPECIAL = 0), 16x2 path
        gemv_kernel<16, 8, 0, 0, false><<<grid, block>>>(params);
    } else if (params.k == 3584) {
        dim3 block(128, 1, 1);
        dim3 grid(params.m / 4, 1, params.b);
        // K_SPECIAL = 3584 -> uses k3584 loader, 16x2 path
        gemv_kernel<4, 32, 7, 3584, false><<<grid, block>>>(params);
    } else if (params.k == 8192) {
        dim3 block(128, 1, 1);
        dim3 grid(params.m, 1, params.b);
        // 8192 special case -> 32x2 path (former v2)
        gemv_kernel<1, 128, 0, 8192, true><<<grid, block>>>(params);
    } else if (params.k == 1024) {
        dim3 block(128, 1, 1);
        dim3 grid(params.m / 8, 1, params.b);
        // K_SPECIAL = 1024 -> uses k1024 loader, 16x2 path
        gemv_kernel<8, 16, 4, 1024, false><<<grid, block>>>(params);
    } else {
        dim3 block(128, 1, 1);
        dim3 grid(params.m / 8, 1, params.b);
        // generic loader again, 16x2 path
        gemv_kernel<8, 16, 0, 0, false><<<grid, block>>>(params);
    }

    return C;
}
"""

# ---- build the module ----
nvfp4_module = load_inline(
    name="nvfp4_gemv",
    cpp_sources=[gemv_cpp],
    cuda_sources=[gemv_cuda],
    functions=["cuda_nvfp4_gemv"],  # this exposes the function to Python
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3",
        "-w",
        "-maxrregcount=32",
        "--use_fast_math",
        "-allow-unsupported-compiler",
    ],
    extra_ldflags=["-lcuda", "-lcublas"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    return nvfp4_module.cuda_nvfp4_gemv(data[0], data[1], data[6], data[2], data[3])
