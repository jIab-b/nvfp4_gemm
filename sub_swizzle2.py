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
static constexpr int TMEM_COLS = 128;

// For 128B swizzle: each 8-row block needs 8*128=1024 bytes, MMA_M/8=16 blocks = 16384 bytes for A
#define SMEM_A_TILE      (16 * 1024)  // 128B swizzle layout: 16 blocks of 1024 bytes each
#define SMEM_B_TILE      (MMA_N * MMA_K / 2)  // B uses interleaved layout (no swizzle)
#define SMEM_OFF_MBAR    8
#define SMEM_OFF_TILES   1024  // 1024B alignment covers swizzle boundaries
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

// ============================================================================
// SMEM LOAD A - 128B swizzle (mode 2)
// ============================================================================
__device__ __forceinline__ void load_a_swizzle128B(
    const Gemm_params& params, int m_block, int k_offset, uint8_t* a_smem)
{
    const uint8_t* a_global = reinterpret_cast<const uint8_t*>(params.a_ptr)
                              + m_block * params.a_row_stride + k_offset / 2;

    // MMA_M=128, MMA_K=64, so 128*64/2 = 4096 bytes total in global
    // Each row: 32 bytes (64 FP4 elements / 2)
    // Layout: 8 rows form one swizzle block, 128 bytes per row in the atom

    for (int i = threadIdx.x; i < MMA_M * MMA_K / 2; i += blockDim.x) {
        int m = i / (MMA_K / 2);           // row 0-127
        int k_byte = i % (MMA_K / 2);      // byte 0-31 within row

        uint8_t val = a_global[m * params.a_row_stride + k_byte];

        // Decompose m into block_row (which 8-row group) and inner_row (0-7)
        int block_row = m / 8;             // 0-15
        int inner_row = m % 8;             // 0-7

        // Decompose k_byte into 16B chunk index and offset within chunk
        int chunk_idx = k_byte / 16;       // 0-1 (since 32 bytes = 2 chunks)
        int chunk_off = k_byte % 16;       // 0-15

        // Apply 128B swizzle: XOR chunk_idx with inner_row (3 bits each)
        int swizzled_chunk = chunk_idx ^ inner_row;

        // Each swizzle atom is 8 rows × 128 bytes = 1024 bytes
        // Within atom: row stride = 128 bytes
        int smem_offset = block_row * 1024 +      // which 8-row block
                          inner_row * 128 +        // row within block (128B stride)
                          swizzled_chunk * 16 +    // swizzled chunk position
                          chunk_off;               // byte within chunk

        a_smem[smem_offset] = val;
    }
}
__device__ __forceinline__ void load_b_swizzle0_interleaved(
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


// ============================================================================
// SETUP / INIT
// ============================================================================
__device__ __forceinline__ void init_tmem_and_mbar(
    char* smem, uint32_t smem_base, uint32_t mbar_smem,
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
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_smem), "r"(1) : "memory");
    }
    __syncthreads();
}

// ============================================================================
// TMEM LOAD SCALE FACTORS
// ============================================================================
__device__ __forceinline__ void load_scales_to_tmem(
    const Gemm_params& params, int m_block, int n_block, int k_offset,
    uint32_t tmem_sfa, uint32_t tmem_sfb)
{
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int sf_k_idx = k_offset / 16;
    int m_row = m_block + warp_id * 32 + lane_id;

    const uint8_t* sfa_base = reinterpret_cast<const uint8_t*>(params.sfa_ptr);
    uint32_t sfa_val = *reinterpret_cast<const uint32_t*>(
        sfa_base + m_row * params.sfa_row_stride + sf_k_idx);

    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
        :: "r"(tmem_sfa), "r"(sfa_val) : "memory"
    );

    const uint8_t* sfb_base = reinterpret_cast<const uint8_t*>(params.sfb_ptr);
    uint32_t sfb_val0 = *reinterpret_cast<const uint32_t*>(
        sfb_base + (n_block + lane_id) * params.sfb_row_stride + sf_k_idx);
    uint32_t sfb_val1 = *reinterpret_cast<const uint32_t*>(
        sfb_base + (n_block + lane_id + 32) * params.sfb_row_stride + sf_k_idx);

    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
        :: "r"(tmem_sfb), "r"(sfb_val0) : "memory"
    );
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
        :: "r"(tmem_sfb + 1), "r"(sfb_val1) : "memory"
    );
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

// ============================================================================
// MAIN COMPUTE (MMA)
// ============================================================================
__device__ __forceinline__ void issue_mma(
    uint32_t tmem_d, uint64_t a_desc, uint64_t b_desc, uint32_t idesc,
    uint32_t tmem_sfa, uint32_t tmem_sfb, uint32_t mbar_smem, int k_tile)
{
    uint32_t enable_d = (k_tile > 0) ? 1 : 0;
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
    __shared__ __align__(1024) char smem[SMEM_TOTAL];
    char* tile_smem = smem + SMEM_OFF_TILES;
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t mbar_smem = smem_base + SMEM_OFF_MBAR;

    const int m_block = blockIdx.y * MMA_M;
    const int n_block = blockIdx.x * MMA_N;
    if (m_block >= params.M || n_block >= params.N) return;

    uint32_t tmem_d, tmem_sfa, tmem_sfb;
    init_tmem_and_mbar(smem, smem_base, mbar_smem, tmem_d, tmem_sfa, tmem_sfb);

    uint32_t idesc = make_mxf4_idesc(MMA_M, MMA_N);
    const int num_k_tiles = params.K / MMA_K;
    int mbar_phase = 0;

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_offset = k_tile * MMA_K;

        uint8_t* a_smem = reinterpret_cast<uint8_t*>(tile_smem);
        uint8_t* b_smem = reinterpret_cast<uint8_t*>(tile_smem + SMEM_A_TILE);

        load_a_swizzle128B(params, m_block, k_offset, a_smem);
        load_b_swizzle0_interleaved(params, n_block, k_offset, b_smem);
        __syncthreads();

        // 128B swizzle: leading 16B, stride 1024B (8 rows * 128B), swizzle mode 2
        uint64_t a_desc = make_smem_desc(a_smem, 16, 1024, 2);
        uint64_t b_desc = make_smem_desc(b_smem, 128, 256, 0);

        load_scales_to_tmem(params, m_block, n_block, k_offset, tmem_sfa, tmem_sfb);
        __syncthreads();

        issue_mma(tmem_d, a_desc, b_desc, idesc, tmem_sfa, tmem_sfb, mbar_smem, k_tile);
        wait_mma(mbar_smem, mbar_phase);
        __syncthreads();
    }

    load_accum_and_store(params, m_block, n_block, tmem_d);
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

    dim3 grid_dim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M, 1);
    dim3 block_dim(WARPS_PER_CTA * THREADS_PER_WARP);

    gemm_kernel_tcgen05<<<grid_dim, block_dim>>>(params);
    return C;
}
"""

nvfp4_tcgen05_module = load_inline(
    name="nvfp4_tcgen05_gemm_swizzle2",
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
