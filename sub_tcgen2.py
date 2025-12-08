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
// Constants
// ============================================================================
static constexpr int MMA_M = 128;
static constexpr int MMA_N = 64;
static constexpr int MMA_K = 64;
static constexpr int WARPS_PER_CTA = 4;
static constexpr int THREADS_PER_WARP = 32;
// Layout D (4x1): D uses MMA_N columns (64), accessed via lane offsets for M-blocks
// SFA needs 1 col (4 packed bytes per lane), SFB needs 2 cols (64 values in 32 lanes)
// SFB is broadcast to all 4 lane partitions for accessibility
// Total: 64 + 1 + 2 = 67 (with padding at 68-69), TMEM_COLS=128 is sufficient
static constexpr int TMEM_COLS = 128;
static constexpr int SF_ROW_STRIDE = 16;  // 16-byte aligned for descriptors

// ============================================================================
// Shared Memory Layout (static allocation)
// ============================================================================
#define SMEM_TMEM_ADDR   4                                    // uint32_t for tmem base addr
#define SMEM_MBAR        8                                    // uint64_t mbarrier (8-byte aligned)
#define SMEM_A_TILE      (MMA_M * MMA_K / 2)                  // A tile: 128*64/2 = 4096 bytes
#define SMEM_B_TILE      (MMA_N * MMA_K / 2)                  // B tile: 64*64/2 = 2048 bytes
// Scale factors now loaded directly GMEM->RMEM->TMEM, no SMEM needed

#define SMEM_OFF_MBAR    8                                    // mbar at offset 8 (8-byte aligned)
#define SMEM_OFF_TILES   16                                   // tiles start after mbar
#define SMEM_TOTAL       (SMEM_OFF_TILES + SMEM_A_TILE + SMEM_B_TILE)

struct Gemm_params {
    int M, N, K;
    const void* __restrict__ a_ptr;
    const void* __restrict__ b_ptr;
    const void* __restrict__ sfa_ptr;
    const void* __restrict__ sfb_ptr;
    __half* __restrict__ c_ptr;
    int64_t a_row_stride;
    int64_t b_row_stride;
    int64_t sfa_row_stride;
    int64_t sfb_row_stride;
    int debug_block_x;
    int debug_block_y;
};

#define DEBUG_PRINT(fmt, ...) do { \
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) \
        printf(fmt, ##__VA_ARGS__); \
} while(0)

__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr, int leading_dim_bytes, int stride_dim_bytes, int swizzle_mode = 0)
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

// ----------------------------------------------------------------------------
// SETUP / INIT
// ----------------------------------------------------------------------------
__device__ __forceinline__ void init_tmem_and_mbar(
    char* smem, uint32_t smem_base, uint32_t mbar_smem,
    uint32_t& tmem_d, uint32_t& tmem_sfa, uint32_t& tmem_sfb)
{
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\\n"
            :: "r"(smem_base), "r"(TMEM_COLS) : "memory"
        );
    }
    __syncthreads();

    tmem_d = *reinterpret_cast<uint32_t*>(smem);
    tmem_sfa = tmem_d + MMA_N;
    tmem_sfb = tmem_d + MMA_N + 4;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n"
            :: "r"(mbar_smem), "r"(1) : "memory");
    }
    __syncthreads();
}

// ----------------------------------------------------------------------------
// SMEM LOAD A
// ----------------------------------------------------------------------------
__device__ __forceinline__ void load_a_to_smem(
    const Gemm_params& params, int m_block, int k_offset, uint8_t* a_smem)
{
    const uint8_t* a_global = reinterpret_cast<const uint8_t*>(params.a_ptr)
                              + m_block * params.a_row_stride + k_offset / 2;

    for (int i = threadIdx.x; i < MMA_M * MMA_K / 2; i += blockDim.x) {
        int m = i / (MMA_K / 2);
        int k_byte = i % (MMA_K / 2);
        uint8_t val = a_global[m * params.a_row_stride + k_byte];
        int block_row = m / 8;
        int inner_row = m % 8;
        int k_group = k_byte / 16;
        int k_inner = k_byte % 16;
        int smem_offset = inner_row * 16 + k_group * 128 + block_row * 256 + k_inner;
        a_smem[smem_offset] = val;
    }
}

// ----------------------------------------------------------------------------
// SMEM LOAD B
// ----------------------------------------------------------------------------
__device__ __forceinline__ void load_b_to_smem(
    const Gemm_params& params, int n_block, int k_offset, uint8_t* b_smem)
{
    const uint8_t* b_global = reinterpret_cast<const uint8_t*>(params.b_ptr)
                              + n_block * params.b_row_stride + k_offset / 2;

    for (int i = threadIdx.x; i < MMA_N * MMA_K / 2; i += blockDim.x) {
        int n = i / (MMA_K / 2);
        int k_byte = i % (MMA_K / 2);
        uint8_t val = b_global[n * params.b_row_stride + k_byte];
        int block_row = n / 8;
        int inner_row = n % 8;
        int k_group = k_byte / 16;
        int k_inner = k_byte % 16;
        int smem_offset = inner_row * 16 + k_group * 128 + block_row * 256 + k_inner;
        b_smem[smem_offset] = val;
    }
}

// ----------------------------------------------------------------------------
// TMEM LOAD SCALE FACTORS
// ----------------------------------------------------------------------------
__device__ __forceinline__ void load_scales_to_tmem(
    const Gemm_params& params, int m_block, int n_block, int k_offset,
    uint32_t tmem_sfa, uint32_t tmem_sfb)
{
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    // sf_k_idx = which group of 4 scale factors (for scale_vec::4X with MMA_K=64)
    // k_offset=0 -> sf_k_idx=0, k_offset=64 -> sf_k_idx=4, etc.
    const int sf_k_idx = k_offset / 16;

    // === SFA: Each warp loads its 32 M-rows ===
    // Warp 0 -> M-rows 0-31, Warp 1 -> M-rows 32-63, etc.
    int m_row = m_block + warp_id * 32 + lane_id;

    const uint8_t* sfa_base = reinterpret_cast<const uint8_t*>(params.sfa_ptr);
    uint32_t sfa_val = *reinterpret_cast<const uint32_t*>(
        sfa_base + m_row * params.sfa_row_stride + sf_k_idx);

    // tcgen05.st.32x32b is a collective warp operation
    // Hardware automatically routes to the correct 32-lane partition based on warp ID
    // Use column-only addressing (no lane_base << 16)
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\\n"
        :: "r"(tmem_sfa), "r"(sfa_val) : "memory"
    );

    // === SFB: All warps store same data (replication to all lane partitions) ===
    const uint8_t* sfb_base = reinterpret_cast<const uint8_t*>(params.sfb_ptr);

    uint32_t sfb_val0 = *reinterpret_cast<const uint32_t*>(
        sfb_base + (n_block + lane_id) * params.sfb_row_stride + sf_k_idx);
    uint32_t sfb_val1 = *reinterpret_cast<const uint32_t*>(
        sfb_base + (n_block + lane_id + 32) * params.sfb_row_stride + sf_k_idx);

    // All warps write the same SFB data - hardware routes each warp to its partition
    // This replicates SFB across all 4 partitions (128 lanes total)
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\\n"
        :: "r"(tmem_sfb), "r"(sfb_val0) : "memory"
    );
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\\n"
        :: "r"(tmem_sfb + 1), "r"(sfb_val1) : "memory"
    );

    asm volatile("tcgen05.wait::st.sync.aligned;\\n" ::: "memory");
}

// ----------------------------------------------------------------------------
// DEBUG: Fill SMEM A with 0x00 pattern - expect all zeros from MMA
// ----------------------------------------------------------------------------
__device__ __forceinline__ void debug_fill_smem_a(uint8_t* a_smem)
{
    for (int i = threadIdx.x; i < MMA_M * MMA_K / 2; i += blockDim.x) {
        a_smem[i] = 0x00;
    }
    __syncthreads();
}

// ----------------------------------------------------------------------------
// DEBUG: Print SMEM A data for rows 0,32,64,96 and SFA from TMEM
// ----------------------------------------------------------------------------
__device__ __forceinline__ void debug_print_pre_mma(
    uint32_t tmem_sfa, uint32_t tmem_sfb,
    const uint8_t* a_smem, const uint8_t* b_smem,
    int k_tile, int m_block, int n_block)
{
    if (blockIdx.x != 0 || blockIdx.y != 0) return;
    if (k_tile != 0) return;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    if (threadIdx.x == 0) {
        printf("\\n===== DEBUG k_tile=0 =====\\n");
        printf("A_SMEM row0  (offset 0):    ");
        for (int i = 0; i < 8; i++) printf("%02x ", a_smem[0 + i]);
        printf("\\n");
        printf("A_SMEM row32 (offset 1024): ");
        for (int i = 0; i < 8; i++) printf("%02x ", a_smem[1024 + i]);
        printf("\\n");
        printf("A_SMEM row64 (offset 2048): ");
        for (int i = 0; i < 8; i++) printf("%02x ", a_smem[2048 + i]);
        printf("\\n");
        printf("A_SMEM row96 (offset 3072): ");
        for (int i = 0; i < 8; i++) printf("%02x ", a_smem[3072 + i]);
        printf("\\n");
    }
    __syncthreads();
}

// ----------------------------------------------------------------------------
// DEBUG: Print non-zero accumulator values from loaded registers
// Called from load_accum_and_store with the final accumulator values
// ----------------------------------------------------------------------------
__device__ __forceinline__ void debug_print_accum(const float* acc_regs, int out_m, int m_block, int n_block)
{
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Print any non-zero accumulator values from any block
    for (int n = 0; n < MMA_N; n++) {
        if (acc_regs[n] != 0.0f) {
            printf("NON-ZERO blk(%d,%d) warp%d lane%02d M=%3d N=%2d: %f\\n",
                   blockIdx.x, blockIdx.y, warp_id, lane_id, m_block + out_m, n_block + n, acc_regs[n]);
        }
    }
}

// ----------------------------------------------------------------------------
// MAIN COMPUTE (MMA)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void issue_mma(
    uint32_t tmem_d, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
    uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t mbar_smem, int k_tile)
{
    uint32_t enable_d = (k_tile > 0) ? 1 : 0;

    if (threadIdx.x == 0) {
        asm volatile(
            "{\\n"
            ".reg .pred p;\\n"
            "setp.ne.u32 p, %6, 0;\\n"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X\\n"
            "    [%0], %1, %2, %3, [%4], [%5], p;\\n"
            "}\\n"
            :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc),
               "r"(tmem_sfa), "r"(tmem_sfb), "r"(enable_d) : "memory"
        );
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\\n"
            :: "r"(mbar_smem) : "memory"
        );
    }
}

__device__ __forceinline__ void wait_mma(uint32_t mbar_smem, int& mbar_phase)
{
    uint32_t mma_done = 0;
    while (!mma_done) {
        asm volatile(
            "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}\\n"
            : "=r"(mma_done) : "r"(mbar_smem), "r"(mbar_phase) : "memory"
        );
    }
    mbar_phase ^= 1;
}

// ----------------------------------------------------------------------------
// ACCUMULATOR LOAD + STORE
// ----------------------------------------------------------------------------
__device__ __forceinline__ void load_accum_and_store(
    const Gemm_params& params, int m_block, int n_block, uint32_t tmem_d)
{
    asm volatile("tcgen05.fence::after_thread_sync;\\n" ::: "memory");

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int out_m = warp_id * 32 + lane_id;

    float acc_regs[MMA_N];

    // tcgen05.ld.32x32b automatically routes to correct partition based on warp ID
    // Use column-only addressing (consistent with stores)
    #pragma unroll
    for (int n_chunk = 0; n_chunk < MMA_N; n_chunk += 8) {
        uint32_t taddr = tmem_d + n_chunk;
        uint32_t r[8];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\\n"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
              "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
            : "r"(taddr) : "memory"
        );
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            acc_regs[n_chunk + i] = __uint_as_float(r[i]);
        }
    }

    asm volatile("tcgen05.wait::ld.sync.aligned;\\n" ::: "memory");

    // Debug: print any non-zero accumulator values
    debug_print_accum(acc_regs, out_m, m_block, n_block);

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
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\\n"
            :: "r"(tmem_d), "r"(TMEM_COLS) : "memory"
        );
    }
    __syncthreads();
}

// ----------------------------------------------------------------------------
// MAIN KERNEL
// ----------------------------------------------------------------------------
__global__ void __launch_bounds__(WARPS_PER_CTA * THREADS_PER_WARP)
gemm_kernel_tcgen05(const __grid_constant__ Gemm_params params)
{
    // --- SHARED MEMORY SETUP ---
    __shared__ char smem[SMEM_TOTAL];
    char* tile_smem = smem + SMEM_OFF_TILES;
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t mbar_smem = smem_base + SMEM_OFF_MBAR;

    const int m_block = blockIdx.y * MMA_M;
    const int n_block = blockIdx.x * MMA_N;
    if (m_block >= params.M || n_block >= params.N) return;

    // --- INIT ---
    uint32_t tmem_d, tmem_sfa, tmem_sfb;
    init_tmem_and_mbar(smem, smem_base, mbar_smem, tmem_d, tmem_sfa, tmem_sfb);

    uint32_t idesc = make_mxf4_idesc(MMA_M, MMA_N);
   // --- K-LOOP ---
    const int num_k_tiles = params.K / MMA_K;
    int mbar_phase = 0;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_offset = k_tile * MMA_K;

        // --- SMEM LOAD A ---
        uint8_t* a_smem = reinterpret_cast<uint8_t*>(tile_smem);
        load_a_to_smem(params, m_block, k_offset, a_smem);

        // --- SMEM LOAD B ---
        uint8_t* b_smem = reinterpret_cast<uint8_t*>(tile_smem + MMA_M * MMA_K / 2);
        load_b_to_smem(params, n_block, k_offset, b_smem);

        __syncthreads();

        // --- TEST: Fill SMEM A with 0x77 to test if MMA reads all rows ---
        // If rows 32+ are still zero after MMA, MMA isn't computing them
        // If rows 32+ are non-zero, MMA is computing but reading wrong data normally
        debug_fill_smem_a(a_smem);

        // --- BUILD DESCRIPTORS ---
        uint64_t a_desc = make_smem_desc(a_smem, 128, 256, 2);
        uint64_t b_desc = make_smem_desc(b_smem, 128, 256, 0);

        // --- TMEM LOAD SCALE FACTORS ---
        load_scales_to_tmem(params, m_block, n_block, k_offset, tmem_sfa, tmem_sfb);
        __syncthreads();

        // --- DEBUG: Print data before MMA (prints only once) ---
        //debug_print_pre_mma(tmem_sfa, tmem_sfb, a_smem, b_smem, k_tile, m_block, n_block);

        // --- MAIN COMPUTE ---
        issue_mma(tmem_d, a_desc, b_desc, idesc, tmem_sfa, tmem_sfb, mbar_smem, k_tile);
        wait_mma(mbar_smem, mbar_phase);

        __syncthreads();
    }

    // --- ACCUMULATOR LOAD + STORE ---
    load_accum_and_store(params, m_block, n_block, tmem_d);

    // --- TMEM DEALLOC ---
    dealloc_tmem(tmem_d);
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

    Gemm_params params;
    params.M = M; params.N = N; params.K = K;
    params.a_ptr = A.data_ptr();
    params.b_ptr = B.data_ptr();
    params.sfa_ptr = SFA.data_ptr();
    params.sfb_ptr = SFB.data_ptr();
    params.c_ptr = reinterpret_cast<__half*>(C.data_ptr());
    params.a_row_stride = A.stride(0);
    params.b_row_stride = B.stride(0);
    params.sfa_row_stride = SFA.stride(0);
    params.sfb_row_stride = SFB.stride(0);
    params.debug_block_x = 0;
    params.debug_block_y = 0;

    dim3 grid_dim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M, 1);
    dim3 block_dim(WARPS_PER_CTA * THREADS_PER_WARP);

    gemm_kernel_tcgen05<<<grid_dim, block_dim>>>(params);
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
