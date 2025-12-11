"""Minimal cuBLASLt FP4 block‑scaled GEMM fallback."""

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t


cpp_src = """
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_gemm_cublaslt(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor SFA_perm,
    torch::Tensor SFB_perm,
    torch::Tensor C);
"""


cuda_src = """
#include <torch/extension.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>

namespace {

inline cublasLtHandle_t get_handle() {
    static cublasLtHandle_t h = [] {
        cublasLtHandle_t t;
        cublasLtCreate(&t);
        return t;
    }();
    return h;
}

inline void check(cublasStatus_t s, const char* m) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(m);
}

} // namespace

torch::Tensor cuda_nvfp4_gemm_cublaslt(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor SFA_perm,
    torch::Tensor SFB_perm,
    torch::Tensor C)
{
    const int64_t M = A.size(0);
    const int64_t K = A.size(1) * 2;
    const int64_t N = B.size(0);

    const int64_t lda = K;
    const int64_t ldb = K;
    const int64_t ldc = N;

    cublasLtHandle_t handle = get_handle();

    cublasLtMatmulDesc_t opDesc;
    check(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc");

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &transA, sizeof(transA)), "ta");
    check(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &transB, sizeof(transB)), "tb");

    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    check(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                         &scale_mode, sizeof(scale_mode)), "asm");
    check(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                         &scale_mode, sizeof(scale_mode)), "bsm");

    const void* a_scale_ptr = SFA_perm.data_ptr();
    const void* b_scale_ptr = SFB_perm.data_ptr();
    check(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                         &a_scale_ptr, sizeof(a_scale_ptr)), "asp");
    check(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                         &b_scale_ptr, sizeof(b_scale_ptr)), "bsp");

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    check(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, M, K, lda), "Adesc");
    check(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, N, K, ldb), "Bdesc");
    check(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, ldc), "Cdesc");
    check(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16F, M, N, ldc), "Ddesc");

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check(cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &order, sizeof(order)), "orderA");
    check(cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &order, sizeof(order)), "orderB");
    check(cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &order, sizeof(order)), "orderC");
    check(cublasLtMatrixLayoutSetAttribute(Ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                           &order, sizeof(order)), "orderD");

    cublasLtMatmulPreference_t pref;
    check(cublasLtMatmulPreferenceCreate(&pref), "pref");
    size_t ws = 0;
    check(cublasLtMatmulPreferenceSetAttribute(
              pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
          "pref_ws");

    cublasLtMatmulHeuristicResult_t hres{};
    int ret = 0;
    check(cublasLtMatmulAlgoGetHeuristic(
              handle, opDesc, Adesc, Bdesc, Cdesc, Ddesc, pref, 1, &hres, &ret),
          "heuristic");
    TORCH_CHECK(ret > 0, "no algo");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    check(cublasLtMatmul(
              handle, opDesc, &alpha,
              A.data_ptr(), Adesc,
              B.data_ptr(), Bdesc,
              &beta,
              C.data_ptr(), Cdesc,
              C.data_ptr(), Ddesc,
              &hres.algo,
              nullptr, 0,
              nullptr),
          "matmul");

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(opDesc);

    return C;
}
"""


nvfp4_tcgen05_module = load_inline(
    name="nvfp4_tcgen05_gemm_tma_all",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["cuda_nvfp4_gemm_cublaslt"],
    extra_cuda_cflags=[
        "-std=c++17",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--ptxas-options=--gpu-name=sm_100a",
        "-O3",
        "-w",
        "-allow-unsupported-compiler",
    ],
    extra_ldflags=["-lcuda"],
    verbose=False,
)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa, sfb, sfa_perm, sfb_perm, c = data
    return nvfp4_tcgen05_module.cuda_nvfp4_gemm_cublaslt(
        a, b, sfa, sfb, sfa_perm, sfb_perm, c
    )
