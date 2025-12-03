import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# C++ stub to expose the CUDA launcher
cpp_src = """
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_gemm_simple(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C);
"""

cuda_src = """
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <cstdio>

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

static constexpr int K_WORKERS = 16;
static constexpr int M_TILE = 8;
static constexpr int N_TILE = 4;

struct Gemm_params {
    using index_t = uint64_t;
    int m, n, k;
    const __nv_fp4x2_e2m1* __restrict__ a_ptr;
    const __nv_fp4x2_e2m1* __restrict__ b_ptr;
    const __nv_fp8_e4m3* __restrict__ sfa_ptr;
    const __nv_fp8_e4m3* __restrict__ sfb_ptr;
    __half* __restrict__ c_ptr;
    index_t row_stride;
    index_t sf_row_stride;
};

__device__ __forceinline__ void load_row_block(
    const __nv_fp4x2_e2m1* row_ptr,
    const __nv_fp8_e4m3*   row_scale_ptr,
    int                    elem_base,
    int                    block_base,
    __nv_fp4x2_e2m1 (&a_regs)[2],
    __nv_fp8_e4m3 &sfa_regs)
{
    a_regs[0] = row_ptr[elem_base];
    a_regs[1] = row_ptr[elem_base + 1];
    sfa_regs = row_scale_ptr[block_base];
}

__device__ __forceinline__ void load_col_block(
    const __nv_fp4x2_e2m1* col_ptr,
    const __nv_fp8_e4m3*   col_scale_ptr,
    int                    elem_base,
    int                    block_base,
    __nv_fp4x2_e2m1 (&b_regs)[2],
    __nv_fp8_e4m3 &sfb_regs)
{
    b_regs[0] = col_ptr[elem_base];
    b_regs[1] = col_ptr[elem_base + 1];
    sfb_regs = col_scale_ptr[block_base];
}

__device__ __forceinline__ float block_scaled_fma_16x2fp4(
    const __nv_fp4x2_e2m1 (&a_regs)[2],
    const __nv_fp4x2_e2m1 (&b_regs)[2],
    __nv_fp8_e4m3 sfa_regs,
    __nv_fp8_e4m3 sfb_regs)
{
    float accum = 0.0f;

    // Convert scales from fp8 to half then to float (fp8 structs have cast operators)
    __half sfa_half = static_cast<__half>(sfa_regs);
    __half sfb_half = static_cast<__half>(sfb_regs);
    float sfa = __half2float(sfa_half);
    float sfb = __half2float(sfb_half);
    float scale = sfa * sfb;

    // Each fp4x2 holds two fp4 values; convert once per register.
    __half2 a_h2_0 = __half2(__nv_cvt_fp4x2_to_halfraw2(a_regs[0].__x, __NV_E2M1));
    __half2 b_h2_0 = __half2(__nv_cvt_fp4x2_to_halfraw2(b_regs[0].__x, __NV_E2M1));
    __half2 prod_0 = __hmul2(a_h2_0, b_h2_0);
    accum += __half2float(prod_0.x) + __half2float(prod_0.y);

    __half2 a_h2_1 = __half2(__nv_cvt_fp4x2_to_halfraw2(a_regs[1].__x, __NV_E2M1));
    __half2 b_h2_1 = __half2(__nv_cvt_fp4x2_to_halfraw2(b_regs[1].__x, __NV_E2M1));
    __half2 prod_1 = __hmul2(a_h2_1, b_h2_1);
    accum += __half2float(prod_1.x) + __half2float(prod_1.y);

    return accum * scale;
}

template <int M_TILE, int K_WORKERS, int N_TILE>
__global__ void __launch_bounds__(M_TILE * K_WORKERS)
gemm_kernel(const __grid_constant__ Gemm_params params)
{
    const int tid  = threadIdx.x;
    const int m_idx = tid / K_WORKERS;
    const int k_lane = tid % K_WORKERS;

    const int row   = blockIdx.y * M_TILE + m_idx;
    const int n_tile = blockIdx.x;

    if (row >= params.m) {
        return;
    }

    const int col_start = n_tile * N_TILE;
    if (col_start >= params.n) {
        return;
    }

    const __nv_fp4x2_e2m1* rowA = params.a_ptr + row * params.row_stride;
    const __nv_fp8_e4m3* rowS = params.sfa_ptr + row * params.sf_row_stride;

    const __nv_fp4x2_e2m1* colB_ptrs[N_TILE];
    const __nv_fp8_e4m3* colS_ptrs[N_TILE];
    bool col_active[N_TILE];

    #pragma unroll
    for (int ci = 0; ci < N_TILE; ++ci) {
        int col = col_start + ci;
        if (col < params.n) {
            col_active[ci] = true;
            colB_ptrs[ci] = params.b_ptr + col * params.row_stride;
            colS_ptrs[ci] = params.sfb_ptr + col * params.sf_row_stride;
        } else {
            col_active[ci] = false;
            colB_ptrs[ci] = nullptr;
            colS_ptrs[ci] = nullptr;
        }
    }

    float accum[N_TILE] = {0.f};

    const int bytes_per_iter = 16;
    const int iters = params.k / (K_WORKERS * bytes_per_iter);

    #pragma unroll 4
    for (int iter = 0; iter < iters; ++iter) {
        int block_base = iter * K_WORKERS + k_lane;
        int elem_base = block_base * bytes_per_iter / 2;  // Divided by 2 since fp4x2 is 2 elements
        int scale_block_base = block_base;

        __nv_fp4x2_e2m1 a_regs[2];
        __nv_fp8_e4m3 sfa_reg;
        load_row_block(rowA, rowS, elem_base, scale_block_base, a_regs, sfa_reg);

        #pragma unroll
        for (int ci = 0; ci < N_TILE; ++ci) {
            if (!col_active[ci]) {
                continue;
            }
            __nv_fp4x2_e2m1 b_regs[2];
            __nv_fp8_e4m3 sfb_reg;
            load_col_block(colB_ptrs[ci], colS_ptrs[ci], elem_base, scale_block_base, b_regs, sfb_reg);
            float result = block_scaled_fma_16x2fp4(a_regs, b_regs, sfa_reg, sfb_reg);
            accum[ci] += result;
        }
    }

    __half* row_out = params.c_ptr + row * params.n;

    #pragma unroll
    for (int ci = 0; ci < N_TILE; ++ci) {
        if (!col_active[ci]) {
            continue;
        }
        float value = accum[ci];
        for (int offset = K_WORKERS / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xFFFF'FFFF, value, offset, K_WORKERS);
        }
        if (k_lane == 0) {
            int col = col_start + ci;
            __half* out_ptr = row_out + col;
            out_ptr[0] = __float2half(value);
        }
    }
}

torch::Tensor cuda_nvfp4_gemm_simple(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C)
{
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    dim3 block_dim(M_TILE * K_WORKERS, 1, 1);
    dim3 grid_dim(
        (N + N_TILE - 1) / N_TILE,
        (M + M_TILE - 1) / M_TILE,
        1);

    Gemm_params params;
    params.m = M;
    params.n = N;
    params.k = K;
    params.a_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(A.data_ptr());
    params.b_ptr = reinterpret_cast<const __nv_fp4x2_e2m1*>(B.data_ptr());
    params.sfa_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(SFA.data_ptr());
    params.sfb_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(SFB.data_ptr());
    params.c_ptr = reinterpret_cast<__half*>(C.data_ptr());
    params.row_stride = A.stride(0);
    params.sf_row_stride = SFA.stride(0);

    gemm_kernel<M_TILE, K_WORKERS, N_TILE><<<grid_dim, block_dim, 0>>>(params);
    return C;
}
"""

nvfp4_gemm_simple_module = load_inline(
    name="nvfp4_blockscaled_gemm_simple",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["cuda_nvfp4_gemm_simple"],
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3",
        "-w",
        "-maxrregcount=64",
        "--use_fast_math",
        "-allow-unsupported-compiler",
    ],
    extra_ldflags=["-lcuda", "-lcublas"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    return nvfp4_gemm_simple_module.cuda_nvfp4_gemm_simple(data[0], data[1], data[2], data[3], data[6])
