import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_source = """


#include <torch/extension.h>


#include <cuda_runtime.h>





torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c);


"""
cuda_source = """


#include <torch/extension.h>


#include <cuda_fp16.h>


#include <cuda_fp4.h>


#include <cuda_fp8.h>





// ============================================================================


// ======================== CONFIGURATION CONSTANTS ===========================


// ============================================================================


#define BLOCK_SIZE 64


#define ROWS_PER_BLOCK 2


#define K_TILE 4096  // Tunable parameter (unused in direct load version)





// ============================================================================


// ======================== TYPE CONVERSION HELPERS ===========================


// ============================================================================


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





// ============================================================================


// ================ SCALED DOT PRODUCT FOR 4 PACKED BYTES =====================


// ============================================================================


__device__ __forceinline__ __half2 dot_scaled_4bytes(


    uchar4 a4,


    uchar4 b4,


    __half2 scale_h2


) {


    // Two independent FMA chains to increase ILP


    __half2 acc0 = __hmul2(decode_fp4x2(a4.x), __hmul2(decode_fp4x2(b4.x), scale_h2));


    __half2 acc1 = __hmul2(decode_fp4x2(a4.y), __hmul2(decode_fp4x2(b4.y), scale_h2));


    acc0 = __hfma2(decode_fp4x2(a4.z), __hmul2(decode_fp4x2(b4.z), scale_h2), acc0);


    acc1 = __hfma2(decode_fp4x2(a4.w), __hmul2(decode_fp4x2(b4.w), scale_h2), acc1);





    return __hadd2(acc0, acc1);


}





// ============================================================================


// ==================== DIRECT LOAD COMPUTE FUNCTION ==========================


// ============================================================================


__device__ __forceinline__ float compute_direct(


    const uint8_t* row_a,


    const uint8_t* batch_b,


    const uint8_t* row_sfa,


    const uint8_t* batch_sfb,


    int K_sf,


    int tid


) {


    // Process 1 scale factor (16 FP4 values) per chain, 2 chains for ILP


    float acc0 = 0.0f;


    const int THREADS_PER_ROW = BLOCK_SIZE / ROWS_PER_BLOCK;


    const int STRIDE = THREADS_PER_ROW * 1;  // Process 2 sf per iteration





#pragma unroll 4


    for (int sf_base = tid * 1; sf_base < K_sf; sf_base += STRIDE) {


        // Chain 0: process sf_base (16 FP4 values = 8 bytes)


        if (sf_base < K_sf) {


            float scale0 = decode_fp8(static_cast<int8_t>(__ldg(&row_sfa[sf_base]))) *


                          decode_fp8(static_cast<int8_t>(__ldg(&batch_sfb[sf_base])));


            __half2 scale_h2_0 = __halves2half2(__float2half(scale0), __float2half(scale0));





            int byte_base0 = sf_base << 3;


            uchar4 a4_0 = *reinterpret_cast<const uchar4*>(&row_a[byte_base0]);


            uchar4 b4_0 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base0]);


            uchar4 a4_1 = *reinterpret_cast<const uchar4*>(&row_a[byte_base0 + 4]);


            uchar4 b4_1 = *reinterpret_cast<const uchar4*>(&batch_b[byte_base0 + 4]);





            __half2 res0_0 = dot_scaled_4bytes(a4_0, b4_0, scale_h2_0);


            __half2 res0_1 = dot_scaled_4bytes(a4_1, b4_1, scale_h2_0);





            float2 f0_0 = __half22float2(res0_0);


            float2 f0_1 = __half22float2(res0_1);


            acc0 += f0_0.x + f0_0.y + f0_1.x + f0_1.y;


        }


    }





    return acc0;


}





// ============================================================================


// ========================== MAIN KERNEL FUNCTION ============================


// ============================================================================


__global__ void gemv_nvfp4_kernel(


    const uint8_t* __restrict__ a,


    const uint8_t* __restrict__ b,


    const uint8_t* __restrict__ sfa,


    const uint8_t* __restrict__ sfb,


    half* __restrict__ c,


    int M, int K, int L,


    int N_rows


) {


    int m_base = blockIdx.x * ROWS_PER_BLOCK;


    int l = blockIdx.y;


    int tid = threadIdx.x;


    int row_in_block = tid / (BLOCK_SIZE / ROWS_PER_BLOCK);


    int tid_in_row = tid % (BLOCK_SIZE / ROWS_PER_BLOCK);





    if (m_base >= M || l >= L) return;





// ============================================================================


// ===================== PER-CTA BASE POINTER SETUP ===========================


// ============================================================================


    const int K_sf = K / 16;


    const int K_fp4 = K / 2;


    const size_t batch_stride_a = static_cast<size_t>(M) * K_fp4;


    const size_t batch_stride_b = static_cast<size_t>(N_rows) * K_fp4;


    const size_t batch_stride_sfa = static_cast<size_t>(M) * K_sf;


    const size_t batch_stride_sfb = static_cast<size_t>(N_rows) * K_sf;





    const uint8_t* batch_a = a + l * batch_stride_a;


    const uint8_t* batch_b = b + l * batch_stride_b;


    const uint8_t* batch_sfa = sfa + l * batch_stride_sfa;


    const uint8_t* batch_sfb = sfb + l * batch_stride_sfb;





    // Calculate row index for this thread


    int m = m_base + row_in_block;


    if (m >= M) return;





    const uint8_t* row_a = batch_a + static_cast<size_t>(m) * K_fp4;


    const uint8_t* row_sfa = batch_sfa + static_cast<size_t>(m) * K_sf;





// ============================================================================


// ===================== DIRECT LOAD COMPUTE (NO BARRIERS) ===================


// ============================================================================


    float acc = compute_direct(


        row_a,


        batch_b,


        row_sfa,


        batch_sfb,


        K_sf,


        tid_in_row


    );





// ============================================================================


// ===================== ROW-GROUP REDUCTION ==================================


// ============================================================================


    // Each row group (32 threads) reduces independently


    const int lane = tid_in_row;


    float row_sum = acc;





    // Warp reduction within each row group


    row_sum += __shfl_down_sync(0xffffffff, row_sum, 16);


    row_sum += __shfl_down_sync(0xffffffff, row_sum, 8);


    row_sum += __shfl_down_sync(0xffffffff, row_sum, 4);


    row_sum += __shfl_down_sync(0xffffffff, row_sum, 2);


    row_sum += __shfl_down_sync(0xffffffff, row_sum, 1);





// ============================================================================


// ===================== FINAL OUTPUT WRITE ===================================


// ============================================================================


    // Thread 0 of each row group writes the result


    if (lane == 0) {


        size_t c_idx = static_cast<size_t>(m) + static_cast<size_t>(l) * M;


        c[c_idx] = __float2half(row_sum);


    }


}





// ============================================================================


// ========================== HOST WRAPPER FUNCTION ===========================


// ============================================================================


torch::Tensor batched_scaled_gemv_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb, torch::Tensor c) {


    int M = a.size(0);


    int K = a.size(1) * 2;


    int L = a.size(2);


    int N_rows = b.size(0);





    // Launch grid with M/ROWS_PER_BLOCK blocks


    int grid_m = (M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;


    dim3 grid(grid_m, L);


    dim3 block(BLOCK_SIZE);





    auto* a_ptr = reinterpret_cast<const uint8_t*>(a.data_ptr());


    auto* b_ptr = reinterpret_cast<const uint8_t*>(b.data_ptr());


    auto* sfa_ptr = reinterpret_cast<const uint8_t*>(sfa.data_ptr());


    auto* sfb_ptr = reinterpret_cast<const uint8_t*>(sfb.data_ptr());


    auto* c_ptr = reinterpret_cast<half*>(c.data_ptr());





    gemv_nvfp4_kernel<<<grid, block>>>(


        a_ptr,


        b_ptr,


        sfa_ptr,


        sfb_ptr,


        c_ptr,


        M, K, L,


        N_rows


    );
    return c;
}
"""
module = load_inline(
    name='batched_scaled_gemv_v3',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['batched_scaled_gemv_cuda'],
    extra_cuda_cflags=[
        '-O3',
        '--use_fast_math',
        '-std=c++17',
        '-gencode=arch=compute_100a,code=sm_100a',
        '-maxrregcount=56'
    ],
    with_cuda=True,
    verbose=False
)


def custom_kernel(data: input_t) -> output_t:

    a, b, sfa_ref, sfb_ref, _, _, c = data
    device = a.device
    return module.batched_scaled_gemv_cuda(
        a,
        b,
        sfa_ref,
        sfb_ref,
        c
    )
