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
// Constants for tile sizes
// ============================================================================
static constexpr int MMA_M = 128;      // M dimension per MMA
static constexpr int MMA_N = 64;       // N dimension per MMA (can be 8-256)
static constexpr int MMA_K = 64;       // K dimension per MMA for mxf4
static constexpr int WARPS_PER_CTA = 4;
static constexpr int THREADS_PER_WARP = 32;

// Tensor memory allocation size (in columns, must be power of 2, 32-512)
static constexpr int TMEM_COLS = 64;

// ============================================================================
// Kernel parameters
// ============================================================================
struct Gemm_params {
    int M, N, K;
    const void* __restrict__ a_ptr;      // FP4 packed data
    const void* __restrict__ b_ptr;      // FP4 packed data
    const void* __restrict__ sfa_ptr;    // Scale factors for A
    const void* __restrict__ sfb_ptr;    // Scale factors for B
    __half* __restrict__ c_ptr;          // Output
    int64_t a_row_stride;
    int64_t b_row_stride;
    int64_t sfa_row_stride;
    int64_t sfb_row_stride;
};

// ============================================================================
// Helper: Build shared memory descriptor (64-bit)
//
// Descriptor layout:
//   bits 0-13:  matrix start address (encoded)
//   bits 16-29: leading dimension byte offset (stride to next row/col)
//   bits 32-45: stride dimension byte offset
//   bits 46-48: fixed 0b001
//   bits 49-51: base offset for swizzle alignment
//   bit 52:     0=relative offset, 1=absolute address
//   bits 53-60: fixed 0
//   bits 61-63: swizzle mode (0=none, 2=128B, 4=64B, 6=32B)
//
// For K-major data (K contiguous):
//   - leading_dim = stride between rows (in bytes)
//   - stride_dim = element size along K (in bytes)
// ============================================================================
__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr,
    int leading_dim_bytes,
    int stride_dim_bytes,
    int swizzle_mode = 0)
{
    uint64_t addr = reinterpret_cast<uint64_t>(smem_ptr);

    // matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
    uint64_t start_addr_enc = (addr & 0x3FFFF) >> 4;
    uint64_t lead_dim_enc = (leading_dim_bytes & 0x3FFFF) >> 4;
    uint64_t stride_dim_enc = (stride_dim_bytes & 0x3FFFF) >> 4;

    uint64_t desc = 0;
    desc |= (start_addr_enc & 0x3FFF);           // bits 0-13
    desc |= (lead_dim_enc & 0x3FFF) << 16;       // bits 16-29
    desc |= (stride_dim_enc & 0x3FFF) << 32;     // bits 32-45
    desc |= (0x1ULL) << 46;                       // bits 46-48: fixed 0b001
    desc |= (0ULL) << 49;                         // bits 49-51: base offset
    desc |= (0ULL) << 52;                         // bit 52: relative offset mode
    desc |= (0ULL) << 53;                         // bits 53-60: fixed 0
    desc |= ((uint64_t)swizzle_mode) << 61;      // bits 61-63: swizzle mode

    return desc;
}

// ============================================================================
// Helper: Build instruction descriptor for mxf4nvf4 (32-bit)
// ============================================================================
__device__ __forceinline__ uint32_t make_mxf4_idesc(
    int M_tile, int N_tile,
    int sfa_id = 0, int sfb_id = 0,
    bool use_ue4m3_scale = true)  // true for nvfp4 scales
{
    // For .kind::mxf4nvf4:
    // Bits 0-1:   Reserved (0)
    // Bit 2:      Sparsity = 0 (Dense)
    // Bit 3:      Reserved (0)
    // Bits 4-5:   SFB_ID (0 or 2)
    // Bit 6:      Reserved (0)
    // Bits 7-9:   atype = 1 (E2M1)
    // Bits 10-11: btype = 1 (E2M1)
    // Bit 12:     Reserved (0)
    // Bit 13:     Negate A = 0
    // Bit 14:     Negate B = 0
    // Bit 15:     Transpose A = 0
    // Bit 16:     Transpose B = 0
    // Bits 17-22: N >> 3
    // Bit 23:     Scale type (0=UE4M3, 1=UE8M0)
    // Bits 24-26: Reserved (0)
    // Bits 27-28: M >> 7
    // Bits 29-30: SFA_ID (0 or 2)
    // Bit 31:     K dimension (0=64/128)

    uint32_t idesc = 0;
    idesc |= (1U << 7);                              // atype = E2M1 = 1
    idesc |= (1U << 10);                             // btype = E2M1 = 1
    idesc |= ((N_tile >> 3) & 0x3F) << 17;           // N >> 3
    idesc |= (use_ue4m3_scale ? 0U : 1U) << 23;      // scale type
    idesc |= ((M_tile >> 7) & 0x3) << 27;            // M >> 7
    idesc |= ((sfa_id & 0x3)) << 29;                 // SFA_ID
    idesc |= ((sfb_id & 0x3)) << 4;                  // SFB_ID

    return idesc;
}

// ============================================================================
// Helper: Allocate tensor memory (returns base address via shared mem)
// ============================================================================
__device__ __forceinline__ uint32_t tmem_alloc(uint32_t* smem_addr, int num_cols)
{
    if (threadIdx.x < 32) {  // Only one warp does allocation
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\\n"
            :
            : "r"((uint32_t)(uint64_t)smem_addr), "r"(num_cols)
            : "memory"
        );
    }
    __syncthreads();
    return *smem_addr;
}

// ============================================================================
// Helper: Deallocate tensor memory
// ============================================================================
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, int num_cols)
{
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\\n"
            :
            : "r"(taddr), "r"(num_cols)
            : "memory"
        );
    }
    __syncthreads();
}

// ============================================================================
// Helper: Copy from shared memory to tensor memory
// ============================================================================
__device__ __forceinline__ void tmem_copy_128x256b(uint32_t taddr, uint64_t smem_desc)
{
    asm volatile(
        "tcgen05.cp.cta_group::1.128x256b [%0], %1;\\n"
        :
        : "r"(taddr), "l"(smem_desc)
        : "memory"
    );
}

__device__ __forceinline__ void tmem_copy_128x128b(uint32_t taddr, uint64_t smem_desc)
{
    asm volatile(
        "tcgen05.cp.cta_group::1.128x128b [%0], %1;\\n"
        :
        : "r"(taddr), "l"(smem_desc)
        : "memory"
    );
}

// Copy scale factors to TMEM - 4x256b copies 4 lanes × 256 bits (32 bytes per lane)
__device__ __forceinline__ void tmem_copy_4x256b(uint32_t taddr, uint64_t smem_desc)
{
    asm volatile(
        "tcgen05.cp.cta_group::1.4x256b [%0], %1;\\n"
        :
        : "r"(taddr), "l"(smem_desc)
        : "memory"
    );
}

// ============================================================================
// Helper: Issue tcgen05.mma with block scaling
// ============================================================================
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4(
    uint32_t d_taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t idesc,
    uint32_t sfa_taddr,
    uint32_t sfb_taddr,
    bool enable_input_d)
{
    uint32_t enable_d = enable_input_d ? 1 : 0;

    asm volatile(
        "{\\n"
        ".reg .pred p;\\n"
        "setp.ne.u32 p, %6, 0;\\n"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X\\n"
        "    [%0], %1, %2, %3, [%4], [%5], p;\\n"
        "}\\n"
        :
        : "r"(d_taddr), "l"(a_desc), "l"(b_desc), "r"(idesc),
          "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_d)
        : "memory"
    );
}

// ============================================================================
// Helper: Commit and wait for MMA completion
// ============================================================================
__device__ __forceinline__ void tcgen05_commit(uint64_t* mbar)
{
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\\n"
        :
        : "l"((uint64_t)mbar)
        : "memory"
    );
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase)
{
    // Use numbered local labels for safe branching
    uint32_t done = 0;
    while (!done) {
        asm volatile(
            "{\\n"
            ".reg .pred p;\\n"
            "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\\n"
            "selp.u32 %0, 1, 0, p;\\n"
            "}\\n"
            : "=r"(done)
            : "l"((uint64_t)mbar), "r"(phase)
            : "memory"
        );
    }
}

// ============================================================================
// Helper: Wait for pending tcgen05 operations
// ============================================================================
__device__ __forceinline__ void tcgen05_wait_ld()
{
    asm volatile("tcgen05.wait::ld.sync.aligned;\\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st()
{
    asm volatile("tcgen05.wait::st.sync.aligned;\\n" ::: "memory");
}

// ============================================================================
// Helper: Load from tensor memory to registers
// ============================================================================
__device__ __forceinline__ void tmem_load_32x32b_x2(float* dst, uint32_t taddr)
{
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];\\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(taddr)
        : "memory"
    );
}

__device__ __forceinline__ void tmem_load_16x128b_x4(float* dst, uint32_t taddr)
{
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    asm volatile(
        "tcgen05.ld.sync.aligned.16x128b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]),
          "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7])
        : "r"(taddr)
        : "memory"
    );
}

// ============================================================================
// Helper: Initialize mbarrier
// ============================================================================
__device__ __forceinline__ void mbar_init(uint64_t* mbar, int arrive_count)
{
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.init.shared.b64 [%0], %1;\\n"
            :
            : "l"((uint64_t)mbar), "r"(arrive_count)
            : "memory"
        );
    }
    __syncthreads();
}

// ============================================================================
// Main GEMM Kernel using tcgen05
//
// Computes: C[m,n] = sum_k (A[m,k] * SFA[m,k/16]) * (B[n,k] * SFB[n,k/16])
//
// Input layouts (from task.yml):
//   A: [M, K] K-major (K contiguous), FP4 e2m1 packed (2 values/byte)
//   B: [N, K] K-major (K contiguous), FP4 e2m1 packed (2 values/byte)
//   SFA: [M, K/16] K-major, FP8 e4m3
//   SFB: [N, K/16] K-major, FP8 e4m3
//   C: [M, N] output in FP16
//
// tcgen05.mma.mxf4nvf4 expects (TN layout, transpose flags must be 0):
//   Matrix A: M x K (no transpose)
//   Matrix B: K x N (no transpose, but our B is N x K)
//
// Solution: Use smem descriptor to present B's [N,K] data as [K,N] to the MMA
// by swapping leading/stride dimensions. Zero runtime cost - just metadata.
// ============================================================================
__global__ void __launch_bounds__(WARPS_PER_CTA * THREADS_PER_WARP)
gemm_kernel_tcgen05(const __grid_constant__ Gemm_params params)
{
    // Shared memory layout:
    // - Tensor memory address storage (4 bytes)
    // - Mbarrier object (8 bytes, aligned)
    // - Tile buffers for A, B, and scale factors
    extern __shared__ char smem_raw[];

    uint32_t* tmem_addr_storage = reinterpret_cast<uint32_t*>(smem_raw);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_raw + 16);

    // Tile data starts after mbarrier (aligned to 128 bytes)
    char* tile_smem = smem_raw + 128;

    const int m_block = blockIdx.y * MMA_M;
    const int n_block = blockIdx.x * MMA_N;

    // Early exit if out of bounds
    if (m_block >= params.M || n_block >= params.N) return;

    // ========================================================================
    // Step 1: Allocate tensor memory
    // ========================================================================
    uint32_t tmem_base = tmem_alloc(tmem_addr_storage, TMEM_COLS);

    // Partition tensor memory:
    // - D accumulator: needs M x N x 4 bytes (float32)
    // - Scale factors need to be copied too
    uint32_t tmem_d = tmem_base;
    uint32_t tmem_sfa = tmem_base + 16;  // Offset for scale factors A
    uint32_t tmem_sfb = tmem_base + 20;  // Offset for scale factors B

    // ========================================================================
    // Step 2: Initialize mbarrier
    // ========================================================================
    mbar_init(mbar, 1);  // Single arrival expected

    // ========================================================================
    // Step 3: Build instruction descriptor
    // ========================================================================
    // Task uses fp8_e4m3fnuz scales, which maps to UE4M3 in PTX
    uint32_t idesc = make_mxf4_idesc(MMA_M, MMA_N, 0, 0, true);  // UE4M3 scale

    // ========================================================================
    // Step 4: Main K-loop
    // ========================================================================
    const int num_k_tiles = params.K / MMA_K;
    int mbar_phase = 0;  // Track mbarrier phase

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_offset = k_tile * MMA_K;

        // ------------------------------------------------------------------
        // Load A tile to shared memory (collaborative load across threads)
        // A is M x K in K-major, FP4 packed (2 values per byte)
        // ------------------------------------------------------------------
        const uint8_t* a_global = reinterpret_cast<const uint8_t*>(params.a_ptr)
                                  + m_block * params.a_row_stride + k_offset / 2;
        uint8_t* a_smem = reinterpret_cast<uint8_t*>(tile_smem);

        // Simple collaborative load (can be optimized with async copy)
        for (int i = threadIdx.x; i < MMA_M * MMA_K / 2; i += blockDim.x) {
            int m = i / (MMA_K / 2);
            int k = i % (MMA_K / 2);
            a_smem[m * (MMA_K / 2) + k] = a_global[m * params.a_row_stride + k];
        }

        // ------------------------------------------------------------------
        // Load B tile to shared memory
        // B is N x K in K-major, FP4 packed
        // ------------------------------------------------------------------
        const uint8_t* b_global = reinterpret_cast<const uint8_t*>(params.b_ptr)
                                  + n_block * params.b_row_stride + k_offset / 2;
        uint8_t* b_smem = reinterpret_cast<uint8_t*>(tile_smem + MMA_M * MMA_K / 2);

        for (int i = threadIdx.x; i < MMA_N * MMA_K / 2; i += blockDim.x) {
            int n = i / (MMA_K / 2);
            int k = i % (MMA_K / 2);
            b_smem[n * (MMA_K / 2) + k] = b_global[n * params.b_row_stride + k];
        }

        // ------------------------------------------------------------------
        // Load scale factors to shared memory
        // For scale_vec::4X with K=64: 4 scale factors per row (one per 16 elements)
        // SFA: M x 4 (128 x 4 bytes = 512 bytes)
        // SFB: N x 4 (64 x 4 bytes = 256 bytes)
        // Note: Scale factors must be replicated across all 32 lane partitions in TMEM
        // ------------------------------------------------------------------
        const int sf_k_idx = k_offset / 16;  // 4 scale factors for K=64
        const uint8_t* sfa_global = reinterpret_cast<const uint8_t*>(params.sfa_ptr)
                                    + m_block * params.sfa_row_stride + sf_k_idx;
        const uint8_t* sfb_global = reinterpret_cast<const uint8_t*>(params.sfb_ptr)
                                    + n_block * params.sfb_row_stride + sf_k_idx;

        uint8_t* sfa_smem = reinterpret_cast<uint8_t*>(tile_smem + (MMA_M + MMA_N) * MMA_K / 2);
        uint8_t* sfb_smem = sfa_smem + MMA_M * 4;  // 4 scale factors per row

        // Load SFA: 128 rows × 4 bytes each
        for (int i = threadIdx.x; i < MMA_M * 4; i += blockDim.x) {
            int m = i / 4;
            int s = i % 4;
            sfa_smem[m * 4 + s] = sfa_global[m * params.sfa_row_stride + s];
        }

        // Load SFB: 64 rows × 4 bytes each
        for (int i = threadIdx.x; i < MMA_N * 4; i += blockDim.x) {
            int n = i / 4;
            int s = i % 4;
            sfb_smem[n * 4 + s] = sfb_global[n * params.sfb_row_stride + s];
        }

        __syncthreads();

        // ------------------------------------------------------------------
        // Build shared memory descriptors for A and B
        // ------------------------------------------------------------------
        //
        // For tcgen05.mma with mxf4nvf4 (TN layout, no transpose flags):
        //   Matrix A (M x K): read directly from smem via descriptor
        //   Matrix B (K x N): read from smem, descriptor presents [N,K] as [K,N]
        //
        // A descriptor (M x K, K-major in smem):
        //   - leading_dim = K/2 bytes (stride to next M row)
        //   - stride_dim = K/2 bytes (contiguous K access)
        //
        uint64_t a_desc = make_smem_desc(
            a_smem,
            MMA_K / 2,    // leading_dim: stride to next M row (K/2 bytes per row)
            MMA_K / 2,    // stride_dim: contiguous K
            0
        );

        // B descriptor: present [N,K] as [K,N] for MMA
        //   - leading_dim: stride between K rows (1 byte for FP4 pairs)
        //   - stride_dim: stride between N cols (row stride K/2)
        uint64_t b_desc = make_smem_desc(
            b_smem,
            1,            // leading_dim: 1 byte (transpose effect)
            MMA_K / 2,    // stride_dim: K/2 bytes (each N row becomes a column)
            0
        );

        // ------------------------------------------------------------------
        // Copy scale factors from SMEM to TMEM
        // Scale factors MUST be in TMEM for block_scale MMA
        // For scale_vec::4X: scales are 4-byte aligned in TMEM columns
        // Need to replicate to all 32 lane partitions (handled by tcgen05.cp)
        // ------------------------------------------------------------------

        // SFA descriptor: 128 rows x 4 bytes (32 bits per row)
        // Using 4x256b shape: copies 4 lanes × 256 bits = 4 lanes × 32 bytes
        // Need multiple copies to fill all 128 lanes (128/4 = 32 iterations)
        uint64_t sfa_desc = make_smem_desc(
            sfa_smem,
            4,            // leading_dim: 4 bytes per row
            4,            // stride_dim: 4 bytes
            0
        );

        // SFB descriptor: 64 rows x 4 bytes
        uint64_t sfb_desc = make_smem_desc(
            sfb_smem,
            4,            // leading_dim: 4 bytes per row
            4,            // stride_dim: 4 bytes
            0
        );

        // Copy scale factors to TMEM (all threads participate)
        // tcgen05.cp copies with replication across lane partitions
        if (threadIdx.x == 0) {
            // Copy SFA to tmem_sfa
            tmem_copy_4x256b(tmem_sfa, sfa_desc);
            // Copy SFB to tmem_sfb
            tmem_copy_4x256b(tmem_sfb, sfb_desc);

            // Commit and wait for scale factor copies to complete
            tcgen05_commit(mbar);
        }
        __syncthreads();
        mbar_wait(mbar, mbar_phase);
        mbar_phase ^= 1;  // Toggle phase

        // ------------------------------------------------------------------
        // Issue MMA instruction
        // MMA reads A, B from smem descriptors, scales from TMEM, writes D to TMEM
        // ------------------------------------------------------------------
        bool accumulate = (k_tile > 0);  // Accumulate after first tile

        if (threadIdx.x == 0) {
            tcgen05_mma_mxf4nvf4(
                tmem_d, a_desc, b_desc, idesc,
                tmem_sfa, tmem_sfb, accumulate
            );

            // Commit MMA operation
            tcgen05_commit(mbar);
        }
        __syncthreads();
        mbar_wait(mbar, mbar_phase);
        mbar_phase ^= 1;  // Toggle phase

        __syncthreads();
    }

    // ========================================================================
    // Step 5: Load results from tensor memory and store to global memory
    // ========================================================================
    // Results are in f32 in tensor memory, need to convert to f16
    //
    // Tensor memory layout for D (M=128 x N=64):
    //   - 128 lanes (one per M row)
    //   - Each lane has N/32 = 2 columns of 32-bit values
    //   - Total: 128 * 64 * 4 bytes = 32KB
    //
    // With 4 warps (128 threads):
    //   - Each warp covers 32 lanes (M rows)
    //   - Each thread in warp owns 1 lane
    //   - Each thread loads N values = 64 floats

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Output M row this thread is responsible for
    const int out_m = warp_id * 32 + lane_id;

    // Load results from TMEM
    // Using 32x32b.x64 would be ideal but let's use available shapes
    // tcgen05.ld.32x32b.xN loads N registers per lane across 32 lanes
    // For N=64 outputs, we need x64 -> 64 regs per thread

    // Output register array - 64 floats per thread (for N=64)
    float acc_regs[MMA_N];

    // Load using multiple tcgen05.ld calls
    // 32x32b.x2 loads 2 regs per lane, need 32 loads for 64 values
    // Better: use 16x128b.x4 which loads 8 regs per lane for 16 lanes
    // But we have 32 lanes per warp, so each warp does 2 iterations

    // Simpler approach: use 32x32b.x2 repeatedly
    // tmem_d is base, each column is 128 lanes × 4 bytes = 512 bytes
    // TMEM column offset = taddr offset

    #pragma unroll
    for (int n = 0; n < MMA_N; n += 2) {
        // Load 2 values at a time using 32x32b.x2
        // Each load covers 32 lanes (M dimension) and 2 columns (N dimension)
        float tmp[2];
        uint32_t* d = reinterpret_cast<uint32_t*>(tmp);

        // Column offset in tmem: n values * (sizeof float) / 32 lanes = n/8 columns
        // Actually TMEM addressing: each column is 128 lanes
        // For warp warp_id accessing lanes [warp_id*32, warp_id*32+31]:
        //   tcgen05.ld addresses the warp's portion automatically
        uint32_t col_offset = n / 8;  // Each 32x32b accesses 8 N values across lanes

        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];\\n"
            : "=r"(d[0]), "=r"(d[1])
            : "r"(tmem_d + col_offset + (n % 8) / 2)
            : "memory"
        );

        acc_regs[n] = tmp[0];
        acc_regs[n + 1] = tmp[1];
    }

    // Wait for loads to complete
    tcgen05_wait_ld();

    // Convert to fp16 and store to global memory
    if (m_block + out_m < params.M) {
        #pragma unroll
        for (int n = 0; n < MMA_N; n++) {
            if (n_block + n < params.N) {
                params.c_ptr[(m_block + out_m) * params.N + (n_block + n)] =
                    __float2half(acc_regs[n]);
            }
        }
    }

    // ========================================================================
    // Step 6: Deallocate tensor memory
    // ========================================================================
    tmem_dealloc(tmem_base, TMEM_COLS);
}

// ============================================================================
// Host-side kernel launcher
// ============================================================================
torch::Tensor cuda_nvfp4_gemm_tcgen05(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor SFA_perm,
    torch::Tensor SFB_perm,
    torch::Tensor C)
{
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1)) * 2;  // FP4 packed, so actual K is 2x
    const int N = static_cast<int>(B.size(0));

    Gemm_params params;
    params.M = M;
    params.N = N;
    params.K = K;
    params.a_ptr = A.data_ptr();
    params.b_ptr = B.data_ptr();
    params.sfa_ptr = SFA.data_ptr();
    params.sfb_ptr = SFB.data_ptr();
    params.c_ptr = reinterpret_cast<__half*>(C.data_ptr());
    params.a_row_stride = A.stride(0);
    params.b_row_stride = B.stride(0);
    params.sfa_row_stride = SFA.stride(0);
    params.sfb_row_stride = SFB.stride(0);

    // Grid: one block per (MMA_M x MMA_N) tile
    dim3 grid_dim(
        (N + MMA_N - 1) / MMA_N,
        (M + MMA_M - 1) / MMA_M,
        1
    );
    dim3 block_dim(WARPS_PER_CTA * THREADS_PER_WARP);

    // Shared memory size
    int smem_size = 128 +                           // tmem addr + mbarrier
                    (MMA_M * MMA_K / 2) +           // A tile
                    (MMA_N * MMA_K / 2) +           // B tile
                    (MMA_M * 4) +                   // SFA
                    (MMA_N * 4);                    // SFB

    gemm_kernel_tcgen05<<<grid_dim, block_dim, smem_size>>>(params);

    return C;
}
"""

nvfp4_tcgen05_module = load_inline(
    name="nvfp4_tcgen05_gemm",
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
