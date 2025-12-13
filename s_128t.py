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

// ============================================================================
// Constants - MMA_N = 128 version
// ============================================================================
static constexpr int MMA_M = 128;
static constexpr int MMA_N = 128;  // Changed from 64 to 128
static constexpr int MMA_K_MMA = 64;   // 64 FP4 per MMA instruction
static constexpr int MMA_K_TILE = 256; // 256 FP4 loaded per TMA (4 MMAs worth)
static constexpr int K_BLOCKS = 4;     // Number of MMAs per TMA load
static constexpr int WARPS_PER_CTA = 4;
static constexpr int THREADS_PER_WARP = 32;
static constexpr int TMEM_COLS = 160;  // 128 (acc) + 16 (SFA) + 16 (SFB)

// 32B swizzle: 8 rows x 32 bytes = 256 byte atom
// Now loading 256 K elements = 128 bytes per row
#define SMEM_A_TILE      (MMA_M * MMA_K_TILE / 2)   // 128 rows * 128 bytes = 16384
#define SMEM_B_TILE      (MMA_N * MMA_K_TILE / 2)   // 128 rows * 128 bytes = 16384
// 5D permuted scale layout: 512 bytes per k-block in interleaved format [32 rows × 16 bytes]
// Total for 4 k-blocks = 2048 bytes per scale tensor
#define SMEM_SFA_TILE    (512 * K_BLOCKS)   // 512 bytes per k-block * 4 = 2048
#define SMEM_SFB_TILE    (512 * K_BLOCKS)   // 512 bytes per k-block * 4 = 2048
#define SMEM_OFF_MBAR_MMA     8
#define SMEM_OFF_MBAR_TMA_A   16
#define SMEM_OFF_MBAR_TMA_B   24
#define SMEM_OFF_MBAR_TMA_SFA 32
#define SMEM_OFF_MBAR_TMA_SFB 40
#define SMEM_OFF_TILES   256  // 256B alignment for 32B swizzle
#define SMEM_OFF_SFA     (SMEM_OFF_TILES + SMEM_A_TILE + SMEM_B_TILE)
#define SMEM_OFF_SFB     (SMEM_OFF_SFA + SMEM_SFA_TILE)
#define SMEM_TOTAL       (SMEM_OFF_SFB + SMEM_SFB_TILE)

struct Gemm_params {
    int M, N, K;
    CUtensorMap tensormap_a;
    CUtensorMap tensormap_b;
    CUtensorMap tensormap_sfa;
    CUtensorMap tensormap_sfb;
    __half* __restrict__ c_ptr;
};

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
                "{"
                ".reg .pred p;"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                "selp.u32 %0, 1, 0, p;"
                "}"
                : "=r"(done_a)
                : "r"(mbar_a), "r"(phase_a)
                : "memory"
            );
        }
        if (!done_b) {
            asm volatile(
                "{"
                ".reg .pred p;"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                "selp.u32 %0, 1, 0, p;"
                "}"
                : "=r"(done_b)
                : "r"(mbar_b), "r"(phase_b)
                : "memory"
            );
        }
        if (!done_sfa) {
            asm volatile(
                "{"
                ".reg .pred p;"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                "selp.u32 %0, 1, 0, p;"
                "}"
                : "=r"(done_sfa)
                : "r"(mbar_sfa), "r"(phase_sfa)
                : "memory"
            );
        }
        if (!done_sfb) {
            asm volatile(
                "{"
                ".reg .pred p;"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                "selp.u32 %0, 1, 0, p;"
                "}"
                : "=r"(done_sfb)
                : "r"(mbar_sfb), "r"(phase_sfb)
                : "memory"
            );
        }
    }
    phase_a ^= 1;
    phase_b ^= 1;
    phase_sfa ^= 1;
    phase_sfb ^= 1;
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
    tmem_sfa = tmem_d + MMA_N;       // offset 128 for N=128
    tmem_sfb = tmem_d + MMA_N + 16;  // offset 144 (16-column gap like cuBLAS)
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_mma), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_a), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_b), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_sfa), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_sfb), "r"(1) : "memory");
    }
    __syncthreads();
}

// ============================================================================
// TMA SCALE LOADING: 4D layout using cp.async.bulk.tensor.4d
// 4D shape: [16 bytes per row][32 rows][total_k_blocks][num_m_blocks]
// Strides: [16, 512, m_block_stride] — all >= 16 for TMA alignment ✓
// Box: [16, 8, 1, 1] = 128 bytes per load (16 bytes × 8 rows)
// Load each k_block in 4 pieces (rows 0-7, 8-15, 16-23, 24-31), 16 loads per k_tile
// This preserves the [32 rows × 16 bytes] layout required by tcgen05.cp
// ============================================================================
__device__ __forceinline__ void tma_load_scales(
    const CUtensorMap* tensormap_sfa, const CUtensorMap* tensormap_sfb,
    int m_idx, int n_idx, int k_tile,
    uint32_t sfa_smem_addr, uint32_t sfb_smem_addr,
    uint32_t mbar_sfa, uint32_t mbar_sfb)
{
    // Total bytes for 4 k_blocks = 2048 bytes (16 loads of 128 bytes each)
    const int total_bytes = 4 * 512;

    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :: "r"(mbar_sfa), "r"(total_bytes) : "memory"
        );

        // Load 4 k_blocks, 4 TMA loads each (8 rows × 16 bytes = 128 bytes per load)
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            int k_block_idx = k_tile * 4 + b;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                // Each load: 16 bytes × 8 rows = 128 bytes
                // SMEM offset: k_block * 512 + row_chunk * 128
                // This gives contiguous [32 rows × 16 bytes] layout per k_block
                uint32_t smem_addr = sfa_smem_addr + b * 512 + r * 128;
                int row_coord = r * 8;  // Starting row (0, 8, 16, 24)
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];"
                    :: "r"(smem_addr), "l"(tensormap_sfa),
                       "r"(0), "r"(row_coord), "r"(k_block_idx), "r"(m_idx),
                       "r"(mbar_sfa)
                    : "memory"
                );
            }
        }

        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :: "r"(mbar_sfb), "r"(total_bytes) : "memory"
        );

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            int k_block_idx = k_tile * 4 + b;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                uint32_t smem_addr = sfb_smem_addr + b * 512 + r * 128;
                int row_coord = r * 8;
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6];"
                    :: "r"(smem_addr), "l"(tensormap_sfb),
                       "r"(0), "r"(row_coord), "r"(k_block_idx), "r"(n_idx),
                       "r"(mbar_sfb)
                    : "memory"
                );
            }
        }
    }
}

// ============================================================================
// SMEM -> TMEM scale copy using tcgen05.cp (cuBLAS-style precomputed descriptors)
// Permuted layout: each k_block is 512 bytes in interleaved format [32 rows × 16 bytes]
// tcgen05.cp.32x128b.warpx4 copies 512 bytes (32 rows × 16 bytes, warpx4 distributes to 128 M rows)
// SMEM layout: [32, 16] per k_block, use lead=16, stride=16 for contiguous interleaved data
// ============================================================================
__device__ __forceinline__ void copy_scales_smem_to_tmem(
    uint8_t* sfa_smem, uint8_t* sfb_smem,
    uint32_t tmem_sfa, uint32_t tmem_sfb)
{
    if (threadIdx.x == 0) {
        // Precompute all 8 descriptors
        // Permuted layout: 512 bytes per k_block as [32 rows × 16 bytes]
        // lead=16, stride=16 for contiguous interleaved block
        uint64_t sfa_desc0 = make_smem_desc(sfa_smem + 0 * 512, 16, 16, 0);
        uint64_t sfa_desc1 = make_smem_desc(sfa_smem + 1 * 512, 16, 16, 0);
        uint64_t sfa_desc2 = make_smem_desc(sfa_smem + 2 * 512, 16, 16, 0);
        uint64_t sfa_desc3 = make_smem_desc(sfa_smem + 3 * 512, 16, 16, 0);
        uint64_t sfb_desc0 = make_smem_desc(sfb_smem + 0 * 512, 16, 16, 0);
        uint64_t sfb_desc1 = make_smem_desc(sfb_smem + 1 * 512, 16, 16, 0);
        uint64_t sfb_desc2 = make_smem_desc(sfb_smem + 2 * 512, 16, 16, 0);
        uint64_t sfb_desc3 = make_smem_desc(sfb_smem + 3 * 512, 16, 16, 0);

        // Issue all 8 copies (4 SFA + 4 SFB) with +4 TMEM spacing (unchanged)
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfa + 0), "l"(sfa_desc0) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfa + 4), "l"(sfa_desc1) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfa + 8), "l"(sfa_desc2) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfa + 12), "l"(sfa_desc3) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfb + 0), "l"(sfb_desc0) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfb + 4), "l"(sfb_desc1) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfb + 8), "l"(sfb_desc2) : "memory");
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(tmem_sfb + 12), "l"(sfb_desc3) : "memory");
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// ============================================================================
// MAIN COMPUTE (MMA)
// ============================================================================
__device__ __forceinline__ void issue_mma(
    uint32_t tmem_d, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
    uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t mbar_smem, bool accumulate)
{
    uint32_t enable_d = accumulate ? 1 : 0;
    if (threadIdx.x == 0) {
        asm volatile(
            "{"
            ".reg .pred p;"
            "setp.ne.u32 p, %6, 0;"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
            "    [%0], %1, %2, %3, [%4], [%5], p;"
            "}"
            :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc),
               "r"(tmem_sfa), "r"(tmem_sfb), "r"(enable_d) : "memory"
        );
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
            :: "r"(mbar_smem) : "memory"
        );
    }
}

__device__ __forceinline__ void wait_mma(uint32_t mbar_smem, int& mbar_phase)
{
    uint32_t mma_done = 0;
    while (!mma_done) {
        asm volatile(
            "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
            : "=r"(mma_done) : "r"(mbar_smem), "r"(mbar_phase) : "memory"
        );
    }
    mbar_phase ^= 1;
}

// ============================================================================
// ACCUMULATOR LOAD + STORE
// ============================================================================
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
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
              "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
            : "r"(taddr) : "memory"
        );
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc_regs[n_chunk + i] = __uint_as_float(r[i]);
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    if (m_block + out_m < params.M) {
        #pragma unroll
        for (int n = 0; n < MMA_N; n++) {
            if (n_block + n < params.N) {
                params.c_ptr[(m_block + out_m) * params.N + (n_block + n)] =
                    __float2half(acc_regs[n]);
            }
        }
    }
}

__device__ __forceinline__ void dealloc_tmem(uint32_t tmem_d)
{
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_d), "r"(TMEM_COLS) : "memory"
        );
    }
    __syncthreads();
}

// ============================================================================
// MAIN KERNEL
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
    const int m_idx = blockIdx.y;  // Block index for 5D scale TMA
    const int n_idx = blockIdx.x;  // Block index for 5D scale TMA
    if (m_block >= params.M || n_block >= params.N) return;

    uint32_t tmem_d, tmem_sfa, tmem_sfb;
    init_tmem_and_mbars(smem, smem_base, mbar_mma, mbar_tma_a, mbar_tma_b,
                        mbar_tma_sfa, mbar_tma_sfb, tmem_d, tmem_sfa, tmem_sfb);

    uint32_t idesc = make_mxf4_idesc(MMA_M, MMA_N);
    const int num_k_tiles = params.K / MMA_K_TILE;  // Outer loop over 256-K tiles
    int mma_phase = 0;
    int tma_phase_a = 0;
    int tma_phase_b = 0;
    int tma_phase_sfa = 0;
    int tma_phase_sfb = 0;

    uint8_t* a_smem = reinterpret_cast<uint8_t*>(smem + SMEM_OFF_TILES);
    uint8_t* b_smem = reinterpret_cast<uint8_t*>(smem + SMEM_OFF_TILES + SMEM_A_TILE);
    uint8_t* sfa_smem = reinterpret_cast<uint8_t*>(smem + SMEM_OFF_SFA);
    uint8_t* sfb_smem = reinterpret_cast<uint8_t*>(smem + SMEM_OFF_SFB);
    uint32_t a_smem_addr = smem_base + SMEM_OFF_TILES;
    uint32_t b_smem_addr = smem_base + SMEM_OFF_TILES + SMEM_A_TILE;
    uint32_t sfa_smem_addr = smem_base + SMEM_OFF_SFA;
    uint32_t sfb_smem_addr = smem_base + SMEM_OFF_SFB;

    // Size of each K chunk in smem (32 bytes * rows)
    const int a_chunk_bytes = 32 * MMA_M;  // 32 * 128 = 4096
    const int b_chunk_bytes = 32 * MMA_N;  // 32 * 128 = 4096

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_offset = k_tile * MMA_K_TILE;
        const int k_bytes = k_offset / 2;  // 256 FP4 = 128 bytes

        // TMA load A and B (4 x 32-byte chunks each)
        tma_load_4x(&params.tensormap_a, m_block, k_bytes, a_smem_addr, mbar_tma_a, MMA_M);
        tma_load_4x(&params.tensormap_b, n_block, k_bytes, b_smem_addr, mbar_tma_b, MMA_N);

        // TMA load scales using 5D permuted layout (2048 bytes for 4 k_blocks)
        tma_load_scales(&params.tensormap_sfa, &params.tensormap_sfb,
                        m_idx, n_idx, k_tile,
                        sfa_smem_addr, sfb_smem_addr,
                        mbar_tma_sfa, mbar_tma_sfb);

        // Wait for all TMA loads (A, B, SFA, SFB) in parallel
        wait_tma_all_four(mbar_tma_a, tma_phase_a, mbar_tma_b, tma_phase_b,
                          mbar_tma_sfa, tma_phase_sfa, mbar_tma_sfb, tma_phase_sfb);
        __syncthreads();

        // Copy all 4 k-blocks of scales to TMEM once per 256-K tile
        copy_scales_smem_to_tmem(sfa_smem, sfb_smem, tmem_sfa, tmem_sfb);

        // Inner loop: 4 MMAs per TMA load
        #pragma unroll
        for (int k_block = 0; k_block < K_BLOCKS; k_block++) {
            // Each k_block's data is in a separate smem region
            uint8_t* a_smem_k = a_smem + k_block * a_chunk_bytes;
            uint8_t* b_smem_k = b_smem + k_block * b_chunk_bytes;

            // Both A and B use 32B swizzle (mode 6)
            // stride_dim = 256 (8 rows * 32 bytes for 32B swizzle pattern)
            uint64_t a_desc = make_smem_desc(a_smem_k, 16, 256, 6);
            uint64_t b_desc = make_smem_desc(b_smem_k, 16, 256, 6);

            // Issue MMA (accumulate after first MMA of first tile)
            bool accumulate = (k_tile > 0) || (k_block > 0);
            issue_mma(tmem_d, a_desc, b_desc, idesc,
                tmem_sfa + 4 * k_block, tmem_sfb + 4 * k_block, mbar_mma, accumulate);
            wait_mma(mbar_mma, mma_phase);
        }
    }

    load_accum_and_store(params, m_block, n_block, tmem_d);
    dealloc_tmem(tmem_d);
}

// ============================================================================
// HOST: Create TensorMap with 32B swizzle (for A/B)
// Box is 32 bytes - we issue 4 TMA loads per 256-K tile
// ============================================================================
void create_tensormap(CUtensorMap* tensormap, const void* data_ptr,
                      int rows, int k_bytes, int row_stride_bytes, int box_rows)
{
    uint64_t globalDim[2] = {(uint64_t)k_bytes, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)row_stride_bytes};
    uint32_t boxDim[2] = {32, (uint32_t)box_rows};  // 32 bytes per TMA load (SWIZZLE_32B requires box=32)
    uint32_t elementStride[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tensormap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        2,
        const_cast<void*>(data_ptr),
        globalDim,
        globalStride,
        boxDim,
        elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_32B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + errStr);
    }
}

// ============================================================================
// HOST: Create 4D TensorMap for permuted scales
// 4D shape: [16 bytes per row][32 rows][total_k_blocks][num_m_blocks]
// Strides: [16, 512, m_block_stride] — all >= 16 for TMA alignment ✓
// Box: [16, 8, 1, 1] = 128 bytes per load (16 bytes × 8 rows)
// Load 4 times per k_block (rows 0-7, 8-15, 16-23, 24-31)
// ============================================================================
void create_scale_tensormap_4d(CUtensorMap* tensormap, const void* data_ptr,
                               int num_m_blocks, int total_k_blocks, int m_block_stride)
{
    // 4D: [bytes_per_row, rows, k_blocks, m_blocks]
    uint64_t globalDim[4] = {16, 32, (uint64_t)total_k_blocks, (uint64_t)num_m_blocks};
    // Strides for dims 1-3 (dim 0 has implicit stride of 1)
    // All strides >= 16 to satisfy TMA alignment requirements
    uint64_t globalStride[3] = {16, 512, (uint64_t)m_block_stride};
    // Box loads 128 bytes: 16 bytes × 8 rows (preserves row layout for tcgen05.cp)
    uint32_t boxDim[4] = {16, 8, 1, 1};
    uint32_t elementStride[4] = {1, 1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        tensormap,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        4,
        const_cast<void*>(data_ptr),
        globalDim,
        globalStride,
        boxDim,
        elementStride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        throw std::runtime_error(std::string("Scale tensormap 4D failed: ") + errStr);
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

    // Scale dimensions from actual permuted tensor shapes
    // SFA_perm shape: [32, 4, num_m_blocks, 4, total_k_blocks, 1]
    // Strides:        [16, 4, m_stride,     1,  512,           m_stride]
    const int num_m_blocks = static_cast<int>(SFA_perm.size(2));
    const int num_n_blocks = static_cast<int>(SFB_perm.size(2));
    const int total_k_blocks = static_cast<int>(SFA_perm.size(4));

    // m_block_stride from permuted tensor (stride for dim 2 = m_blocks)
    const int sfa_m_block_stride = static_cast<int>(SFA_perm.stride(2));
    const int sfb_m_block_stride = static_cast<int>(SFB_perm.stride(2));

    Gemm_params params;
    params.M = M; params.N = N; params.K = K;
    params.c_ptr = reinterpret_cast<__half*>(C.data_ptr());

    // Create tensormaps for A and B with 32B swizzle
    create_tensormap(&params.tensormap_a, A.data_ptr(), M, K_bytes, A.stride(0), MMA_M);
    create_tensormap(&params.tensormap_b, B.data_ptr(), N, K_bytes, B.stride(0), MMA_N);

    // Use 4D permuted scales (avoids stride < 16 alignment issue)
    // 4D shape: [16 bytes per row][32 rows][total_k_blocks][num_m/n_blocks]
    create_scale_tensormap_4d(&params.tensormap_sfa, SFA_perm.data_ptr(),
                              num_m_blocks, total_k_blocks, sfa_m_block_stride);
    create_scale_tensormap_4d(&params.tensormap_sfb, SFB_perm.data_ptr(),
                              num_n_blocks, total_k_blocks, sfb_m_block_stride);

    dim3 grid_dim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M, 1);
    dim3 block_dim(WARPS_PER_CTA * THREADS_PER_WARP);

    gemm_kernel_tcgen05<<<grid_dim, block_dim>>>(params);
    return C;
}
"""

nvfp4_tcgen05_module = load_inline(
    name="nvfp4_tcgen05_gemm_n128",
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
        a, b, sfa, sfb, sfa_perm[:,:,:,:,:,0], sfb_perm[:,:,:,:,:,0], c
    )
