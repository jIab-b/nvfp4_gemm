import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

cpp_src = """
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_gemm_tcgen05(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor SFA_perm,
    torch::Tensor SFB_perm,
    torch::Tensor C);
"""

cuda_src = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cstdio>

// ============================================================================
// Constants - same as sub_tc2.py baseline
// ============================================================================
static constexpr int MMA_M = 128;
static constexpr int MMA_N = 64;
static constexpr int MMA_K_MMA = 64;
static constexpr int MMA_K_TILE = 256;
static constexpr int K_BLOCKS = 4;
static constexpr int WARPS_PER_CTA = 4;
static constexpr int THREADS_PER_WARP = 32;
static constexpr int TMEM_COLS = 144;

#define SMEM_A_TILE      (MMA_M * MMA_K_TILE / 2)
#define SMEM_B_TILE      (MMA_N * MMA_K_TILE / 2)
#define SMEM_SFA_TILE    (MMA_M * 16)
#define SMEM_SFB_TILE    (MMA_N * 16)
#define SCALE_K_TILES    4
#define SMEM_OFF_MBAR_MMA     8
#define SMEM_OFF_MBAR_TMA_A   16
#define SMEM_OFF_MBAR_TMA_B   24
#define SMEM_OFF_MBAR_TMA_SFA 32
#define SMEM_OFF_MBAR_TMA_SFB 40
#define SMEM_OFF_TILES   256
#define SMEM_OFF_SFA     (SMEM_OFF_TILES + SMEM_A_TILE + SMEM_B_TILE)
#define SMEM_OFF_SFB     (SMEM_OFF_SFA + SMEM_SFA_TILE)

// Extra SMEM for tcgen05.cp testing - 8KB buffer for various configurations
#define SMEM_OFF_CP_TEST (SMEM_OFF_SFB + SMEM_SFB_TILE)
#define SMEM_CP_TEST_SIZE 8192
#define SMEM_TOTAL       (SMEM_OFF_CP_TEST + SMEM_CP_TEST_SIZE)

struct Gemm_params {
    int M, N, K;
    CUtensorMap tensormap_a;
    CUtensorMap tensormap_b;
    CUtensorMap tensormap_sfa;
    CUtensorMap tensormap_sfb;
    __half* __restrict__ c_ptr;
};

// ============================================================================
// SMEM descriptor helper - same as baseline
// ============================================================================
__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr, int leading_dim_bytes, int stride_dim_bytes, int swizzle_mode)
{
    uint64_t addr = reinterpret_cast<uint64_t>(smem_ptr);
    uint64_t start_addr_enc = (addr & 0x3FFFF) >> 4;
    uint64_t lead_dim_enc = (leading_dim_bytes & 0x3FFFF) >> 4;
    uint64_t stride_dim_enc = (stride_dim_bytes & 0x3FFFF) >> 4;
    uint64_t desc = 0;
    desc |= (start_addr_enc & 0x3FFF);
    desc |= (lead_dim_enc & 0x3FFF) << 16;
    desc |= (stride_dim_enc & 0x3FFF) << 32;
    desc |= (0x1ULL) << 46;
    desc |= ((uint64_t)swizzle_mode) << 61;
    return desc;
}

__device__ __forceinline__ uint32_t make_mxf4_idesc(int M_tile, int N_tile)
{
    uint32_t idesc = 0;
    idesc |= (1U << 7);
    idesc |= (1U << 10);
    idesc |= (0U << 16);
    idesc |= ((N_tile >> 3) & 0x3F) << 17;
    idesc |= (0U) << 23;
    idesc |= ((M_tile >> 7) & 0x3) << 27;
    return idesc;
}

// ============================================================================
// tcgen05.cp TEST CONFIGURATIONS
// Tests various SMEM descriptor / TMEM offset combinations
// All use: tcgen05.cp.cta_group::1.32x128b.warpx4
// ============================================================================

// Debug print helper - DISABLED
#define DEBUG_PRINT(msg)
#define DEBUG_PRINT_DONE(n)

// Test config 1: cuBLAS-style - stride=128, lead=128, no swizzle
__device__ __noinline__ void test_cp_config_1(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 1 START: stride=128, lead=128, swizzle=0\\n");
    if (threadIdx.x == 0) {
        // Stride=128, Lead=128, Swizzle=0 (matches cuBLAS 0x4008 high bits)
        uint64_t desc = make_smem_desc(smem_base, 128, 128, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(1);
}

// Test config 2: stride=128, lead=128, tmem offset +4
__device__ __noinline__ void test_cp_config_2(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 2 START: tmem offset +4\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 128, 128, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base + 4), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(2);
}

// Test config 3: stride=128, lead=128, tmem offset +8
__device__ __noinline__ void test_cp_config_3(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 3 START: tmem offset +8\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 128, 128, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base + 8), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(3);
}

// Test config 4: stride=128, lead=128, tmem offset +12
__device__ __noinline__ void test_cp_config_4(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 4 START: tmem offset +12\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 128, 128, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base + 12), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(4);
}

// Test config 5: stride=64, lead=64, no swizzle (smaller stride)
__device__ __noinline__ void test_cp_config_5(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 5 START: stride=64, lead=64\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 64, 64, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(5);
}

// Test config 6: stride=256, lead=256, no swizzle (larger stride)
__device__ __noinline__ void test_cp_config_6(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 6 START: stride=256, lead=256\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 256, 256, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(6);
}

// Test config 7: stride=32, lead=32, no swizzle (minimum stride)
__device__ __noinline__ void test_cp_config_7(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 7 START: stride=32, lead=32\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 32, 32, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(7);
}

// Test config 8-10: REMOVED - swizzle modes cause illegal instruction

// Test config 11: 4 consecutive copies with +4 tmem spacing (cuBLAS pattern)
__device__ __noinline__ void test_cp_config_11_cublas_pattern(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 11 START: cuBLAS pattern 4x +4 spacing\\n");
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int blk = 0; blk < 4; blk++) {
            uint8_t* src = smem_base + blk * 512;  // 512 bytes per k-block
            uint64_t desc = make_smem_desc(src, 128, 128, 0);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                :: "r"(tmem_base + blk * 4), "l"(desc) : "memory"
            );
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(11);
}

// Test config 12: 4 consecutive copies with +16 tmem spacing (alternative)
__device__ __noinline__ void test_cp_config_12_wide_spacing(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 12 START: 4x +16 spacing\\n");
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int blk = 0; blk < 4; blk++) {
            uint8_t* src = smem_base + blk * 512;
            uint64_t desc = make_smem_desc(src, 128, 128, 0);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                :: "r"(tmem_base + blk * 16), "l"(desc) : "memory"
            );
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(12);
}

// Test config 13: Different smem base offsets (aligned to 128B)
__device__ __noinline__ void test_cp_config_13_smem_offset(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 13 START: smem +128B offset\\n");
    if (threadIdx.x == 0) {
        // Test with 128-byte aligned smem offset
        uint64_t desc = make_smem_desc(smem_base + 128, 128, 128, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(13);
}

// Test config 14: Different smem base offsets (aligned to 256B)
__device__ __noinline__ void test_cp_config_14_smem_offset_256(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 14 START: smem +256B offset\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base + 256, 128, 128, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(14);
}

// Test config 15: stride=128, lead=64 (asymmetric)
__device__ __noinline__ void test_cp_config_15_asym_lead(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 15 START: lead=64, stride=128\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 64, 128, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(15);
}

// Test config 16: stride=64, lead=128 (asymmetric reversed)
__device__ __noinline__ void test_cp_config_16_asym_stride(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 16 START: lead=128, stride=64\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 128, 64, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(16);
}

// Test config 17: stride=512, lead=512 (very large)
__device__ __noinline__ void test_cp_config_17_large_stride(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 17 START: stride=512, lead=512\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 512, 512, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(17);
}

// Test config 18: stride=16, lead=16 (very small)
__device__ __noinline__ void test_cp_config_18_small_stride(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 18 START: stride=16, lead=16\\n");
    if (threadIdx.x == 0) {
        uint64_t desc = make_smem_desc(smem_base, 16, 16, 0);
        asm volatile(
            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
            :: "r"(tmem_base), "l"(desc) : "memory"
        );
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(18);
}

// Test config 19: 8 copies with +4 spacing (2x cuBLAS)
__device__ __noinline__ void test_cp_config_19_8copies(uint8_t* smem_base, uint32_t tmem_base)
{
    DEBUG_PRINT("CP config 19 START: 8x +4 spacing\\n");
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int blk = 0; blk < 8; blk++) {
            uint8_t* src = smem_base + blk * 512;
            uint64_t desc = make_smem_desc(src, 128, 128, 0);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                :: "r"(tmem_base + blk * 4), "l"(desc) : "memory"
            );
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(19);
}

// Test config 20: 2 copies SFA style + 2 copies SFB style (dual region)
__device__ __noinline__ void test_cp_config_20_dual_region(uint8_t* smem_base, uint32_t tmem_sfa, uint32_t tmem_sfb)
{
    DEBUG_PRINT("CP config 20 START: dual region SFA+SFB\\n");
    if (threadIdx.x == 0) {
        // SFA copies (2 copies, +4 spacing)
        #pragma unroll
        for (int blk = 0; blk < 2; blk++) {
            uint8_t* src = smem_base + blk * 512;
            uint64_t desc = make_smem_desc(src, 128, 128, 0);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                :: "r"(tmem_sfa + blk * 4), "l"(desc) : "memory"
            );
        }
        // SFB copies (2 copies, +4 spacing) from different smem region
        #pragma unroll
        for (int blk = 0; blk < 2; blk++) {
            uint8_t* src = smem_base + 2048 + blk * 512;  // Offset by 2KB
            uint64_t desc = make_smem_desc(src, 128, 128, 0);
            asm volatile(
                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                :: "r"(tmem_sfb + blk * 4), "l"(desc) : "memory"
            );
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    DEBUG_PRINT_DONE(20);
}

// ============================================================================
// Master test function - runs all configurations
// ============================================================================
__device__ __noinline__ void run_all_cp_tests(uint8_t* cp_test_smem, uint32_t tmem_d)
{
    DEBUG_PRINT("=== STARTING ALL CP TESTS ===\\n");

    // TMEM layout: accumulator uses columns 0-63, scales start at 64
    uint32_t tmem_test_base = tmem_d + MMA_N;  // Start after accumulator
    uint32_t tmem_sfa = tmem_test_base;
    uint32_t tmem_sfb = tmem_test_base + 16;   // 16-column gap like cuBLAS

    // Run all test configurations (8-10 removed - swizzle causes illegal instruction)
    test_cp_config_1(cp_test_smem, tmem_test_base);
    test_cp_config_2(cp_test_smem, tmem_test_base);
    test_cp_config_3(cp_test_smem, tmem_test_base);
    test_cp_config_4(cp_test_smem, tmem_test_base);
    test_cp_config_5(cp_test_smem, tmem_test_base);
    test_cp_config_6(cp_test_smem, tmem_test_base);
    test_cp_config_7(cp_test_smem, tmem_test_base);
    // configs 8-10 removed (swizzle modes)
    test_cp_config_11_cublas_pattern(cp_test_smem, tmem_sfa);
    test_cp_config_12_wide_spacing(cp_test_smem, tmem_test_base);
    test_cp_config_13_smem_offset(cp_test_smem, tmem_test_base);
    test_cp_config_14_smem_offset_256(cp_test_smem, tmem_test_base);
    test_cp_config_15_asym_lead(cp_test_smem, tmem_test_base);
    test_cp_config_16_asym_stride(cp_test_smem, tmem_test_base);
    test_cp_config_17_large_stride(cp_test_smem, tmem_test_base);
    test_cp_config_18_small_stride(cp_test_smem, tmem_test_base);
    test_cp_config_19_8copies(cp_test_smem, tmem_test_base);
    test_cp_config_20_dual_region(cp_test_smem, tmem_sfa, tmem_sfb);

    DEBUG_PRINT("=== ALL CP TESTS COMPLETED ===\\n");
}

// ============================================================================
// Baseline functions from sub_tc2.py (unchanged)
// ============================================================================
__device__ __forceinline__ void tma_load_4x(
    const CUtensorMap* tensormap, int row_block, int k_byte_offset_base,
    uint32_t smem_base_addr, uint32_t mbar_smem, int rows)
{
    if (threadIdx.x == 0) {
        const int bytes_per_chunk = 32 * rows;
        const int total_bytes = 4 * bytes_per_chunk;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :: "r"(mbar_smem), "r"(total_bytes) : "memory"
        );
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int k_byte_offset = k_byte_offset_base + i * 32;
            uint32_t smem_addr = smem_base_addr + i * bytes_per_chunk;
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                :: "r"(smem_addr), "l"(tensormap), "r"(k_byte_offset), "r"(row_block), "r"(mbar_smem)
                : "memory"
            );
        }
    }
}

__device__ __forceinline__ void wait_tma_all_four(
    uint32_t mbar_a, int& phase_a, uint32_t mbar_b, int& phase_b,
    uint32_t mbar_sfa, int& phase_sfa, uint32_t mbar_sfb, int& phase_sfb)
{
    uint32_t done_a = 0, done_b = 0, done_sfa = 0, done_sfb = 0;
    while (!done_a || !done_b || !done_sfa || !done_sfb) {
        if (!done_a) {
            asm volatile(
                "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                : "=r"(done_a) : "r"(mbar_a), "r"(phase_a) : "memory"
            );
        }
        if (!done_b) {
            asm volatile(
                "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                : "=r"(done_b) : "r"(mbar_b), "r"(phase_b) : "memory"
            );
        }
        if (!done_sfa) {
            asm volatile(
                "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                : "=r"(done_sfa) : "r"(mbar_sfa), "r"(phase_sfa) : "memory"
            );
        }
        if (!done_sfb) {
            asm volatile(
                "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                : "=r"(done_sfb) : "r"(mbar_sfb), "r"(phase_sfb) : "memory"
            );
        }
    }
    phase_a ^= 1; phase_b ^= 1; phase_sfa ^= 1; phase_sfb ^= 1;
}

__device__ __forceinline__ void init_tmem_and_mbars(
    char* smem, uint32_t smem_base, uint32_t mbar_mma,
    uint32_t mbar_tma_a, uint32_t mbar_tma_b,
    uint32_t mbar_tma_sfa, uint32_t mbar_tma_sfb,
    uint32_t& tmem_d, uint32_t& tmem_sfa, uint32_t& tmem_sfb)
{
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"(smem_base), "r"(TMEM_COLS) : "memory"
        );
    }
    __syncthreads();
    tmem_d = *reinterpret_cast<uint32_t*>(smem);
    tmem_sfa = tmem_d + MMA_N;
    tmem_sfb = tmem_d + MMA_N + 4;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_mma), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_tma_a), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_tma_b), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_tma_sfa), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_tma_sfb), "r"(1) : "memory");
    }
    __syncthreads();
}

__device__ __forceinline__ void tma_load_scales(
    const CUtensorMap* tensormap_sfa, const CUtensorMap* tensormap_sfb,
    int m_block, int n_block, int scale_k_group,
    uint32_t sfa_smem_addr, uint32_t sfb_smem_addr,
    uint32_t mbar_sfa, uint32_t mbar_sfb)
{
    const int sf_k_bytes = scale_k_group * 16;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mbar_sfa), "r"(SMEM_SFA_TILE) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :: "r"(sfa_smem_addr), "l"(tensormap_sfa), "r"(sf_k_bytes), "r"(m_block), "r"(mbar_sfa) : "memory"
        );
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mbar_sfb), "r"(SMEM_SFB_TILE) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
            :: "r"(sfb_smem_addr), "l"(tensormap_sfb), "r"(sf_k_bytes), "r"(n_block), "r"(mbar_sfb) : "memory"
        );
    }
}

// Original scale copy from sub_tc2.py (uses tcgen05.st, NOT tcgen05.cp)
__device__ __forceinline__ void copy_scales_smem_to_tmem(
    uint32_t sfa_smem_addr, uint32_t sfb_smem_addr,
    uint32_t tmem_sfa, uint32_t tmem_sfb,
    int k_tile_in_group)
{
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int k_byte_offset = k_tile_in_group * 4;

    uint32_t sfa_saddr = sfa_smem_addr + (warp_id * 32 + lane_id) * 16 + k_byte_offset;
    uint32_t sfa_val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfa_val) : "r"(sfa_saddr) : "memory");
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(tmem_sfa + warp_id), "r"(sfa_val) : "memory");

    uint32_t sfb_saddr0 = sfb_smem_addr + lane_id * 16 + k_byte_offset;
    uint32_t sfb_saddr1 = sfb_smem_addr + (lane_id + 32) * 16 + k_byte_offset;
    uint32_t sfb_val0, sfb_val1;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb_val0) : "r"(sfb_saddr0) : "memory");
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb_val1) : "r"(sfb_saddr1) : "memory");
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(tmem_sfb), "r"(sfb_val0) : "memory");
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" :: "r"(tmem_sfb + 1), "r"(sfb_val1) : "memory");
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void issue_mma(
    uint32_t tmem_d, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
    uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t mbar_smem, bool accumulate)
{
    uint32_t enable_d = accumulate ? 1 : 0;
    if (threadIdx.x == 0) {
        asm volatile(
            "{.reg .pred p; setp.ne.u32 p, %6, 0;"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p;}"
            :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(tmem_sfa), "r"(tmem_sfb), "r"(enable_d) : "memory"
        );
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_smem) : "memory");
    }
}

__device__ __forceinline__ void wait_mma(uint32_t mbar_smem, int& mbar_phase)
{
    uint32_t mma_done = 0;
    while (!mma_done) {
        asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
            : "=r"(mma_done) : "r"(mbar_smem), "r"(mbar_phase) : "memory");
    }
    mbar_phase ^= 1;
}

__device__ __forceinline__ void load_accum_and_store(
    const Gemm_params& params, int m_block, int n_block, uint32_t tmem_d)
{
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int out_m = warp_id * 32 + lane_id;

    float acc_regs[MMA_N];
    #pragma unroll
    for (int n_chunk = 0; n_chunk < MMA_N; n_chunk += 8) {
        uint32_t taddr = tmem_d + n_chunk;
        uint32_t r[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
            : "r"(taddr) : "memory"
        );
        #pragma unroll
        for (int i = 0; i < 8; i++) acc_regs[n_chunk + i] = __uint_as_float(r[i]);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    if (m_block + out_m < params.M) {
        #pragma unroll
        for (int n = 0; n < MMA_N; n++) {
            if (n_block + n < params.N) {
                params.c_ptr[(m_block + out_m) * params.N + (n_block + n)] = __float2half(acc_regs[n]);
            }
        }
    }
}

__device__ __forceinline__ void dealloc_tmem(uint32_t tmem_d)
{
    if (threadIdx.x < 32) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_d), "r"(TMEM_COLS) : "memory");
    }
    __syncthreads();
}

// ============================================================================
// MAIN KERNEL - includes tcgen05.cp tests before real computation
// ============================================================================
__global__ void __launch_bounds__(WARPS_PER_CTA * THREADS_PER_WARP)
gemm_kernel_tcgen05(const __grid_constant__ Gemm_params params)
{
    __shared__ __align__(256) char smem[SMEM_TOTAL];

    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t mbar_mma = smem_base + SMEM_OFF_MBAR_MMA;
    const uint32_t mbar_tma_a = smem_base + SMEM_OFF_MBAR_TMA_A;
    const uint32_t mbar_tma_b = smem_base + SMEM_OFF_MBAR_TMA_B;
    const uint32_t mbar_tma_sfa = smem_base + SMEM_OFF_MBAR_TMA_SFA;
    const uint32_t mbar_tma_sfb = smem_base + SMEM_OFF_MBAR_TMA_SFB;

    const int m_block = blockIdx.y * MMA_M;
    const int n_block = blockIdx.x * MMA_N;
    if (m_block >= params.M || n_block >= params.N) return;

    uint32_t tmem_d, tmem_sfa, tmem_sfb;
    init_tmem_and_mbars(smem, smem_base, mbar_mma, mbar_tma_a, mbar_tma_b,
                        mbar_tma_sfa, mbar_tma_sfb, tmem_d, tmem_sfa, tmem_sfb);

    // =========================================================================
    // RUN tcgen05.cp TESTS (only on first block to avoid noise)
    // =========================================================================
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        uint8_t* cp_test_smem = reinterpret_cast<uint8_t*>(smem + SMEM_OFF_CP_TEST);
        run_all_cp_tests(cp_test_smem, tmem_d);
    }
    __syncthreads();

    // =========================================================================
    // NORMAL GEMM COMPUTATION (same as sub_tc2.py baseline)
    // =========================================================================
    uint32_t idesc = make_mxf4_idesc(MMA_M, MMA_N);
    const int num_k_tiles = params.K / MMA_K_TILE;
    int mma_phase = 0;
    int tma_phase_a = 0, tma_phase_b = 0, tma_phase_sfa = 0, tma_phase_sfb = 0;

    uint8_t* a_smem = reinterpret_cast<uint8_t*>(smem + SMEM_OFF_TILES);
    uint8_t* b_smem = reinterpret_cast<uint8_t*>(smem + SMEM_OFF_TILES + SMEM_A_TILE);
    uint32_t a_smem_addr = smem_base + SMEM_OFF_TILES;
    uint32_t b_smem_addr = smem_base + SMEM_OFF_TILES + SMEM_A_TILE;
    uint32_t sfa_smem_addr = smem_base + SMEM_OFF_SFA;
    uint32_t sfb_smem_addr = smem_base + SMEM_OFF_SFB;

    const int a_chunk_bytes = 32 * MMA_M;
    const int b_chunk_bytes = 32 * MMA_N;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_offset = k_tile * MMA_K_TILE;
        const int k_bytes = k_offset / 2;

        tma_load_4x(&params.tensormap_a, m_block, k_bytes, a_smem_addr, mbar_tma_a, MMA_M);
        tma_load_4x(&params.tensormap_b, n_block, k_bytes, b_smem_addr, mbar_tma_b, MMA_N);
        tma_load_scales(&params.tensormap_sfa, &params.tensormap_sfb,
                        m_block, n_block, k_tile,
                        sfa_smem_addr, sfb_smem_addr,
                        mbar_tma_sfa, mbar_tma_sfb);

        wait_tma_all_four(mbar_tma_a, tma_phase_a, mbar_tma_b, tma_phase_b,
                          mbar_tma_sfa, tma_phase_sfa, mbar_tma_sfb, tma_phase_sfb);
        __syncthreads();

        #pragma unroll
        for (int k_block = 0; k_block < K_BLOCKS; k_block++) {
            uint8_t* a_smem_k = a_smem + k_block * a_chunk_bytes;
            uint8_t* b_smem_k = b_smem + k_block * b_chunk_bytes;
            uint64_t a_desc = make_smem_desc(a_smem_k, 16, 256, 6);
            uint64_t b_desc = make_smem_desc(b_smem_k, 16, 256, 6);

            copy_scales_smem_to_tmem(sfa_smem_addr, sfb_smem_addr, tmem_sfa, tmem_sfb, k_block);

            bool accumulate = (k_tile > 0) || (k_block > 0);
            issue_mma(tmem_d, a_desc, b_desc, idesc, tmem_sfa, tmem_sfb, mbar_mma, accumulate);
            wait_mma(mbar_mma, mma_phase);
        }
    }

    load_accum_and_store(params, m_block, n_block, tmem_d);
    dealloc_tmem(tmem_d);
}

// ============================================================================
// HOST CODE (same as sub_tc2.py baseline)
// ============================================================================
void create_tensormap(CUtensorMap* tensormap, const void* data_ptr,
                      int rows, int k_bytes, int row_stride_bytes, int box_rows)
{
    uint64_t globalDim[2] = {(uint64_t)k_bytes, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)row_stride_bytes};
    uint32_t boxDim[2] = {32, (uint32_t)box_rows};
    uint32_t elementStride[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tensormap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        const_cast<void*>(data_ptr), globalDim, globalStride, boxDim, elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + errStr);
    }
}

void create_scale_tensormap(CUtensorMap* tensormap, const void* data_ptr,
                            int rows, int k_scales, int row_stride_bytes, int box_rows)
{
    uint64_t globalDim[2] = {(uint64_t)k_scales, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)row_stride_bytes};
    uint32_t boxDim[2] = {16, (uint32_t)box_rows};
    uint32_t elementStride[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tensormap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2,
        const_cast<void*>(data_ptr), globalDim, globalStride, boxDim, elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled (scales) failed: ") + errStr);
    }
}

torch::Tensor cuda_nvfp4_gemm_tcgen05(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor SFA, torch::Tensor SFB,
    torch::Tensor SFA_perm, torch::Tensor SFB_perm,
    torch::Tensor C)
{
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1)) * 2;
    const int N = static_cast<int>(B.size(0));
    const int K_bytes = K / 2;
    const int K_scales = K / 16;

    Gemm_params params;
    params.M = M; params.N = N; params.K = K;
    params.c_ptr = reinterpret_cast<__half*>(C.data_ptr());

    create_tensormap(&params.tensormap_a, A.data_ptr(), M, K_bytes, A.stride(0), MMA_M);
    create_tensormap(&params.tensormap_b, B.data_ptr(), N, K_bytes, B.stride(0), MMA_N);
    create_scale_tensormap(&params.tensormap_sfa, SFA.data_ptr(), M, K_scales, SFA.stride(0), MMA_M);
    create_scale_tensormap(&params.tensormap_sfb, SFB.data_ptr(), N, K_scales, SFB.stride(0), MMA_N);

    dim3 grid_dim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M, 1);
    dim3 block_dim(WARPS_PER_CTA * THREADS_PER_WARP);

    gemm_kernel_tcgen05<<<grid_dim, block_dim>>>(params);
    cudaDeviceSynchronize();  // Flush printf output
    return C;
}
"""

nvfp4_tcgen05_module = load_inline(
    name="nvfp4_tcgen05_gemm_cp_dump",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["cuda_nvfp4_gemm_tcgen05"],
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
    return nvfp4_tcgen05_module.cuda_nvfp4_gemm_tcgen05(
        a, b, sfa, sfb, sfa_perm, sfb_perm, c
    )
