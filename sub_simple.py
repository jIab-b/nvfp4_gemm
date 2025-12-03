import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# C++ stub to expose the CUDA launcher
cpp_src = """
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_gemm(
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

// Decode packed fp4x2 to two floats using CUDA built-in
__device__ __forceinline__ void decode_fp4x2(__nv_fp4x2_e2m1 packed, float &lo, float &hi) {
    __half2 h2 = __half2(packed);
    lo = __half2float(h2.x);
    hi = __half2float(h2.y);
}

// Decode fp8 to float using CUDA built-in
__device__ __forceinline__ float decode_fp8(__nv_fp8_e4m3 fp8_val) {
    return float(fp8_val);
}

template <int M_TILE, int K_WORKERS, int N_TILE>
__global__ void __launch_bounds__(M_TILE * K_WORKERS)
gemm_kernel(const __grid_constant__ Gemm_params params)
{
    const int tid = threadIdx.x;
    const int m_idx = tid / K_WORKERS;
    const int k_lane = tid % K_WORKERS;

    const int row = blockIdx.y * M_TILE + m_idx;
    const int n_tile = blockIdx.x;

    if (row >= params.m) {
        return;
    }

    const int col_start = n_tile * N_TILE;
    if (col_start >= params.n) {
        return;
    }

    // Pointers to row A data
    const __nv_fp4x2_e2m1* rowA = params.a_ptr + row * params.row_stride;
    const __nv_fp8_e4m3* rowSA = params.sfa_ptr + row * params.sf_row_stride;

    // Setup column pointers for B
    const __nv_fp4x2_e2m1* colB_ptrs[N_TILE];
    const __nv_fp8_e4m3* colSB_ptrs[N_TILE];
    bool col_active[N_TILE];

    #pragma unroll
    for (int ci = 0; ci < N_TILE; ++ci) {
        int col = col_start + ci;
        if (col < params.n) {
            col_active[ci] = true;
            colB_ptrs[ci] = params.b_ptr + col * params.row_stride;
            colSB_ptrs[ci] = params.sfb_ptr + col * params.sf_row_stride;
        } else {
            col_active[ci] = false;
            colB_ptrs[ci] = nullptr;
            colSB_ptrs[ci] = nullptr;
        }
    }

    float accum[N_TILE] = {0.f};

    // K dimension in tensor is K/2 (packed fp4x2), so total fp4 elements = k * 2
    // Each scale factor covers 16 fp4 elements = 8 bytes
    // Total bytes = params.k (which is K/2, and each is a packed fp4x2 = 1 byte)
    const int total_bytes = params.k;  // k is already K/2, each byte has 2 fp4
    const int bytes_per_worker = total_bytes / K_WORKERS;
    const int my_start = k_lane * bytes_per_worker;

    // Iterate over each packed fp4x2 byte
    for (int byte_idx = my_start; byte_idx < my_start + bytes_per_worker; ++byte_idx) {
        // Scale factor index: 1 scale per 16 fp4 elements = 1 scale per 8 bytes
        int scale_idx = byte_idx / 8;

        float scale_a = decode_fp8(rowSA[scale_idx]);

        float fa_lo, fa_hi;
        decode_fp4x2(rowA[byte_idx], fa_lo, fa_hi);

        #pragma unroll
        for (int ci = 0; ci < N_TILE; ++ci) {
            if (!col_active[ci]) {
                continue;
            }

            float scale_b = decode_fp8(colSB_ptrs[ci][scale_idx]);

            float fb_lo, fb_hi;
            decode_fp4x2(colB_ptrs[ci][byte_idx], fb_lo, fb_hi);

            accum[ci] += (fa_lo * fb_lo + fa_hi * fb_hi) * scale_a * scale_b;
        }
    }

    // Warp reduction across K_WORKERS
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

torch::Tensor cuda_nvfp4_gemm(
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

nvfp4_gemm_module = load_inline(
    name="nvfp4_blockscaled_gemm_simple",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["cuda_nvfp4_gemm"],
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
    return nvfp4_gemm_module.cuda_nvfp4_gemm(data[0], data[1], data[2], data[3], data[6])
