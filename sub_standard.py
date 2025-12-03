import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

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

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

// One thread computes one output element C[row, col]

__global__ void gemm_kernel_simple(
    const __nv_fp4x2_e2m1* __restrict__ A,  // [M, K/2]
    const __nv_fp4x2_e2m1* __restrict__ B,  // [N, K/2]
    const __nv_fp8_e4m3* __restrict__ SFA,  // [M, K/16]
    const __nv_fp8_e4m3* __restrict__ SFB,  // [N, K/16]
    __half* __restrict__ C,                  // [M, N]
    int M, int N, int K_packed,              // K_packed = K/2
    int a_row_stride, int b_row_stride,
    int sfa_row_stride, int sfb_row_stride)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    const __nv_fp4x2_e2m1* a_row = A + row * a_row_stride;
    const __nv_fp4x2_e2m1* b_row = B + col * b_row_stride;
    const __nv_fp8_e4m3* sfa_row = SFA + row * sfa_row_stride;
    const __nv_fp8_e4m3* sfb_row = SFB + col * sfb_row_stride;

    float acc = 0.0f;

    for (int k = 0; k < K_packed; k++) {
        int scale_idx = k / 8;

        float sa = float(sfa_row[scale_idx]);
        float sb = float(sfb_row[scale_idx]);
        float scale = sa * sb;

        __half2 a_h2 = __half2(a_row[k]);
        __half2 b_h2 = __half2(b_row[k]);

        float a0 = __half2float(a_h2.x);
        float a1 = __half2float(a_h2.y);
        float b0 = __half2float(b_h2.x);
        float b1 = __half2float(b_h2.y);

        acc += (a0 * b0 + a1 * b1) * scale;
    }

    C[row * N + col] = __float2half(acc);
}

torch::Tensor cuda_nvfp4_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C)
{
    int M = A.size(0);
    int K_packed = A.size(1);
    int N = B.size(0);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    gemm_kernel_simple<<<grid, block>>>(
        reinterpret_cast<const __nv_fp4x2_e2m1*>(A.data_ptr()),
        reinterpret_cast<const __nv_fp4x2_e2m1*>(B.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(SFA.data_ptr()),
        reinterpret_cast<const __nv_fp8_e4m3*>(SFB.data_ptr()),
        reinterpret_cast<__half*>(C.data_ptr()),
        M, N, K_packed,
        A.stride(0), B.stride(0),
        SFA.stride(0), SFB.stride(0)
    );

    return C;
}
"""

module = load_inline(
    name="nvfp4_gemm_standard",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["cuda_nvfp4_gemm"],
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3",
        "-w",
        "--use_fast_math",
        "-allow-unsupported-compiler",
    ],
    extra_ldflags=["-lcuda"],
    verbose=False,
)

_debug_printed = False


def custom_kernel(data: input_t) -> output_t:
    global _debug_printed

    if not _debug_printed:
        _debug_printed = True
        try:
            a, b, sfa, sfb, sfa_perm, sfb_perm, c = data
            print("[dbg] A shape/stride/dtype:", a.shape, a.stride(), a.dtype)
            print("[dbg] B shape/stride/dtype:", b.shape, b.stride(), b.dtype)
            print("[dbg] SFA shape/stride/dtype:", sfa.shape, sfa.stride(), sfa.dtype)
            print("[dbg] SFB shape/stride/dtype:", sfb.shape, sfb.stride(), sfb.dtype)
            # Tiny samples to avoid log spam
            print("[dbg] SFA[0,0:4,0]:", sfa[0, 0:4, 0].cpu().tolist())
            print("[dbg] SFB[0,0:4,0]:", sfb[0, 0:4, 0].cpu().tolist())
            if sfa_perm is not None:
                print("[dbg] SFA_perm shape/dtype:", sfa_perm.shape, sfa_perm.dtype)
                print("[dbg] SFA_perm[0,0,0,0,0,0]:", sfa_perm[0, 0, 0, 0, 0, 0].item())
            if sfb_perm is not None:
                print("[dbg] SFB_perm shape/dtype:", sfb_perm.shape, sfb_perm.dtype)
                print("[dbg] SFB_perm[0,0,0,0,0,0]:", sfb_perm[0, 0, 0, 0, 0, 0].item())
        except Exception as e:
            print("[dbg] print failed:", e)

    return module.cuda_nvfp4_gemm(data[0], data[1], data[2], data[3], data[6])
