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
    const uint8_t* __restrict__ a_ptr;
    const uint8_t* __restrict__ b_ptr;
    const uint8_t* __restrict__ sfa_ptr;
    const uint8_t* __restrict__ sfb_ptr;
    __half* __restrict__ c_ptr;
    index_t row_stride;
    index_t sf_row_stride;
};

__device__ __forceinline__ __half2 decode_fp4x2(uint8_t byte) {
    __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
        static_cast<__nv_fp4x2_storage_t>(byte),
        __NV_E2M1
    );
    return *reinterpret_cast<__half2*>(&raw);
}

__device__ __forceinline__ float decode_fp8(int8_t byte) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(byte);
    __half_raw raw = __nv_cvt_fp8_to_halfraw(storage, __NV_E4M3);
    return __half2float(__ushort_as_half(raw.x));
}

__device__ __forceinline__ float block_scaled_fma_16x2fp4(
    const uint8_t (&a_bytes)[2],
    const uint8_t (&b_bytes)[2],
    int8_t sfa_byte,
    int8_t sfb_byte)
{
    float scale = decode_fp8(sfa_byte) * decode_fp8(sfb_byte);
    __half2 scale_h2 = __halves2half2(__float2half(scale), __float2half(scale));

    float accum = 0.0f;

    // Decode and multiply first pair of bytes
    __half2 a_h2_0 = decode_fp4x2(a_bytes[0]);
    __half2 b_h2_0 = decode_fp4x2(b_bytes[0]);
    __half2 prod_0 = __hmul2(a_h2_0, __hmul2(b_h2_0, scale_h2));

    // Decode and multiply second pair of bytes
    __half2 a_h2_1 = decode_fp4x2(a_bytes[1]);
    __half2 b_h2_1 = decode_fp4x2(b_bytes[1]);
    __half2 prod_1 = __hfma2(a_h2_1, __hmul2(b_h2_1, scale_h2), prod_0);

    float2 res = __half22float2(prod_1);
    return res.x + res.y;
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

    const uint8_t* rowA = params.a_ptr + row * params.row_stride;
    const uint8_t* rowS = params.sfa_ptr + row * params.sf_row_stride;

    const uint8_t* colB_ptrs[N_TILE];
    const uint8_t* colS_ptrs[N_TILE];
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

    const int K_sf = params.k / 16;  // Scale factors (1 per 16 FP4 values = 8 bytes)
    const int STRIDE = K_WORKERS;

    #pragma unroll 4
    for (int sf_idx = k_lane; sf_idx < K_sf; sf_idx += STRIDE) {
        int byte_base = sf_idx * 8;  // Each SF covers 8 bytes of FP4 data

        int8_t sfa_byte = rowS[sf_idx];

        uint8_t a_bytes[2];
        a_bytes[0] = rowA[byte_base];
        a_bytes[1] = rowA[byte_base + 1];

        #pragma unroll
        for (int ci = 0; ci < N_TILE; ++ci) {
            if (!col_active[ci]) {
                continue;
            }

            int8_t sfb_byte = colS_ptrs[ci][sf_idx];

            uint8_t b_bytes[2];
            b_bytes[0] = colB_ptrs[ci][byte_base];
            b_bytes[1] = colB_ptrs[ci][byte_base + 1];

            float result = block_scaled_fma_16x2fp4(a_bytes, b_bytes, sfa_byte, sfb_byte);
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
    params.a_ptr = reinterpret_cast<const uint8_t*>(A.data_ptr());
    params.b_ptr = reinterpret_cast<const uint8_t*>(B.data_ptr());
    params.sfa_ptr = reinterpret_cast<const uint8_t*>(SFA.data_ptr());
    params.sfb_ptr = reinterpret_cast<const uint8_t*>(SFB.data_ptr());
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
