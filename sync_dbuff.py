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
static constexpr int MMA_K = 64;  // 64 FP4 = 32 bytes
static constexpr int WARPS_PER_CTA = 4;
static constexpr int THREADS_PER_WARP = 32;
static constexpr int TMEM_COLS = 128;

// 32B swizzle: 8 rows x 32 bytes = 256 byte atom, MMA_M/8 = 16 atoms
#define SMEM_A_TILE      (MMA_M * 32)   // 128 rows * 32 bytes = 4096
#define SMEM_B_TILE      (MMA_N * MMA_K / 2)  // 64 * 32 = 2048
#define SMEM_SFA_TILE    (MMA_M * 4)    // 128 rows * 4 bytes = 512 (16B aligned)
#define SMEM_SFB_TILE    (MMA_N * 4)    // 64 rows * 4 bytes = 256 (16B aligned)
#define NUM_BUFFERS      2              // Double buffering

// Mbarrier offsets (8 bytes each)
#define SMEM_OFF_MBAR_MMA     8
#define SMEM_OFF_MBAR_TMA_A0  16
#define SMEM_OFF_MBAR_TMA_A1  24
#define SMEM_OFF_MBAR_TMA_B0  32
#define SMEM_OFF_MBAR_TMA_B1  40

#define SMEM_OFF_TILES   256  // 256B alignment for 32B swizzle
// Double buffered layout: [A0][A1][B0][B1][SFA0][SFA1][SFB0][SFB1]
#define SMEM_OFF_A(buf)      (SMEM_OFF_TILES + (buf) * SMEM_A_TILE)
#define SMEM_OFF_B(buf)      (SMEM_OFF_TILES + NUM_BUFFERS * SMEM_A_TILE + (buf) * SMEM_B_TILE)
#define SMEM_OFF_SFA(buf)    (SMEM_OFF_TILES + NUM_BUFFERS * SMEM_A_TILE + NUM_BUFFERS * SMEM_B_TILE + (buf) * SMEM_SFA_TILE)
#define SMEM_OFF_SFB(buf)    (SMEM_OFF_TILES + NUM_BUFFERS * SMEM_A_TILE + NUM_BUFFERS * SMEM_B_TILE + NUM_BUFFERS * SMEM_SFA_TILE + (buf) * SMEM_SFB_TILE)
#define SMEM_TOTAL           (SMEM_OFF_TILES + NUM_BUFFERS * (SMEM_A_TILE + SMEM_B_TILE + SMEM_SFA_TILE + SMEM_SFB_TILE))

struct Gemm_params {
    int M, N, K;
    CUtensorMap tensormap_a;
    CUtensorMap tensormap_b;
    const void* __restrict__ sfa_ptr;
    const void* __restrict__ sfb_ptr;
    __half* __restrict__ c_ptr;
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
// TMA BULK LOAD with 32B hardware swizzle
// ============================================================================
__device__ __forceinline__ void tma_load(
    const CUtensorMap* tensormap, int row_block, int k_byte_offset,
    uint32_t smem_addr, uint32_t mbar_smem, int tile_bytes)
{
    if (threadIdx.x == 0) {
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :: "r"(mbar_smem), "r"(tile_bytes) : "memory"
        );
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%2, %3}], [%4];"
            :: "r"(smem_addr), "l"(tensormap), "r"(k_byte_offset), "r"(row_block), "r"(mbar_smem)
            : "memory"
        );
    }
}



__device__ __forceinline__ void wait_tma_both(
    uint32_t mbar_a, int& phase_a, uint32_t mbar_b, int& phase_b)
{
    uint32_t done_a = 0, done_b = 0;
    while (!done_a || !done_b) {
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
    }
    phase_a ^= 1;
    phase_b ^= 1;
}


// ============================================================================
// SETUP / INIT
// ============================================================================
__device__ __forceinline__ void init_tmem_and_mbars(
    char* smem, uint32_t smem_base, uint32_t mbar_mma,
    uint32_t mbar_tma_a0, uint32_t mbar_tma_a1,
    uint32_t mbar_tma_b0, uint32_t mbar_tma_b1,
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
            :: "r"(mbar_mma), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_a0), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_a1), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_b0), "r"(1) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
            :: "r"(mbar_tma_b1), "r"(1) : "memory");
    }
    __syncthreads();
}

// ============================================================================
// ASYNC SCALE LOADING: global -> smem (cp.async) -> tmem (tcgen05.st)
// ============================================================================
__device__ __forceinline__ void load_scales_async_to_smem(
    const Gemm_params& params, int m_block, int n_block, int k_offset,
    uint32_t sfa_smem_addr, uint32_t sfb_smem_addr)
{
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int sf_k_idx = k_offset / 16;

    // SFA: 128 threads load 128 rows * 4 bytes = 512 bytes
    int m_row = m_block + warp_id * 32 + lane_id;
    const uint8_t* sfa_base = reinterpret_cast<const uint8_t*>(params.sfa_ptr);
    uint64_t sfa_gaddr = reinterpret_cast<uint64_t>(sfa_base + m_row * params.sfa_row_stride + sf_k_idx);
    uint32_t sfa_saddr = sfa_smem_addr + (warp_id * 32 + lane_id) * 4;

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;"
        :: "r"(sfa_saddr), "l"(sfa_gaddr) : "memory"
    );

    // SFB: first 64 threads load 64 rows * 4 bytes = 256 bytes
    if (threadIdx.x < 64) {
        int n_row = n_block + threadIdx.x;
        const uint8_t* sfb_base = reinterpret_cast<const uint8_t*>(params.sfb_ptr);
        uint64_t sfb_gaddr = reinterpret_cast<uint64_t>(sfb_base + n_row * params.sfb_row_stride + sf_k_idx);
        uint32_t sfb_saddr = sfb_smem_addr + threadIdx.x * 4;

        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 4;"
            :: "r"(sfb_saddr), "l"(sfb_gaddr) : "memory"
        );
    }

    // Commit the group (don't wait - caller will wait)
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__ void wait_scales_async(int groups_to_wait)
{
    if (groups_to_wait == 0) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
    } else {
        asm volatile("cp.async.wait_group 1;" ::: "memory");
    }
}

__device__ __forceinline__ void copy_scales_smem_to_tmem(
    uint32_t sfa_smem_addr, uint32_t sfb_smem_addr,
    uint32_t tmem_sfa, uint32_t tmem_sfb)
{
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    // Load SFA from smem and store to tmem
    uint32_t sfa_saddr = sfa_smem_addr + (warp_id * 32 + lane_id) * 4;
    uint32_t sfa_val;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfa_val) : "r"(sfa_saddr) : "memory");

    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
        :: "r"(tmem_sfa + warp_id), "r"(sfa_val) : "memory"
    );

    // Load SFB from smem and store to tmem (all 128 threads, 2 loads per 64 threads)
    uint32_t sfb_saddr0 = sfb_smem_addr + lane_id * 4;
    uint32_t sfb_saddr1 = sfb_smem_addr + (lane_id + 32) * 4;
    uint32_t sfb_val0, sfb_val1;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb_val0) : "r"(sfb_saddr0) : "memory");
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(sfb_val1) : "r"(sfb_saddr1) : "memory");

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
// MAIN KERNEL (Double Buffered)
// ============================================================================
__global__ void __launch_bounds__(WARPS_PER_CTA * THREADS_PER_WARP)
gemm_kernel_tcgen05(const __grid_constant__ Gemm_params params)
{
    __shared__ __align__(256) char smem[SMEM_TOTAL];

    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t mbar_mma = smem_base + SMEM_OFF_MBAR_MMA;
    const uint32_t mbar_tma_a[2] = {smem_base + SMEM_OFF_MBAR_TMA_A0, smem_base + SMEM_OFF_MBAR_TMA_A1};
    const uint32_t mbar_tma_b[2] = {smem_base + SMEM_OFF_MBAR_TMA_B0, smem_base + SMEM_OFF_MBAR_TMA_B1};

    const int m_block = blockIdx.y * MMA_M;
    const int n_block = blockIdx.x * MMA_N;
    if (m_block >= params.M || n_block >= params.N) return;

    uint32_t tmem_d, tmem_sfa, tmem_sfb;
    init_tmem_and_mbars(smem, smem_base, mbar_mma,
                        mbar_tma_a[0], mbar_tma_a[1],
                        mbar_tma_b[0], mbar_tma_b[1],
                        tmem_d, tmem_sfa, tmem_sfb);

    uint32_t idesc = make_mxf4_idesc(MMA_M, MMA_N);
    const int num_k_tiles = params.K / MMA_K;
    int mma_phase = 0;
    int tma_phase_a[2] = {0, 0};
    int tma_phase_b[2] = {0, 0};

    // Double buffer addresses
    uint8_t* a_smem[2] = {
        reinterpret_cast<uint8_t*>(smem + SMEM_OFF_A(0)),
        reinterpret_cast<uint8_t*>(smem + SMEM_OFF_A(1))
    };
    uint8_t* b_smem[2] = {
        reinterpret_cast<uint8_t*>(smem + SMEM_OFF_B(0)),
        reinterpret_cast<uint8_t*>(smem + SMEM_OFF_B(1))
    };
    uint32_t a_smem_addr[2] = {smem_base + SMEM_OFF_A(0), smem_base + SMEM_OFF_A(1)};
    uint32_t b_smem_addr[2] = {smem_base + SMEM_OFF_B(0), smem_base + SMEM_OFF_B(1)};
    uint32_t sfa_smem_addr[2] = {smem_base + SMEM_OFF_SFA(0), smem_base + SMEM_OFF_SFA(1)};
    uint32_t sfb_smem_addr[2] = {smem_base + SMEM_OFF_SFB(0), smem_base + SMEM_OFF_SFB(1)};

    // ========== PROLOGUE: Load first tile into buffer 0 ==========
    {
        const int k_bytes = 0;
        tma_load(&params.tensormap_a, m_block, k_bytes, a_smem_addr[0], mbar_tma_a[0], SMEM_A_TILE);
        tma_load(&params.tensormap_b, n_block, k_bytes, b_smem_addr[0], mbar_tma_b[0], SMEM_B_TILE);
        load_scales_async_to_smem(params, m_block, n_block, 0, sfa_smem_addr[0], sfb_smem_addr[0]);
    }

    // ========== MAIN LOOP ==========
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int curr = k_tile % 2;
        const int next = (k_tile + 1) % 2;

        // Prefetch next tile (if not last iteration)
        if (k_tile < num_k_tiles - 1) {
            const int next_k_offset = (k_tile + 1) * MMA_K;
            const int next_k_bytes = next_k_offset / 2;
            tma_load(&params.tensormap_a, m_block, next_k_bytes, a_smem_addr[next], mbar_tma_a[next], SMEM_A_TILE);
            tma_load(&params.tensormap_b, n_block, next_k_bytes, b_smem_addr[next], mbar_tma_b[next], SMEM_B_TILE);
            load_scales_async_to_smem(params, m_block, n_block, next_k_offset, sfa_smem_addr[next], sfb_smem_addr[next]);
        }

        // Wait for current tile's TMA loads
        wait_tma_both(mbar_tma_a[curr], tma_phase_a[curr], mbar_tma_b[curr], tma_phase_b[curr]);

        // Wait for current tile's scale loads (allow 1 group in flight if prefetching)
        wait_scales_async(k_tile < num_k_tiles - 1 ? 1 : 0);
        __syncthreads();

        // Build descriptors for current buffer
        uint64_t a_desc = make_smem_desc(a_smem[curr], 16, 256, 6);
        uint64_t b_desc = make_smem_desc(b_smem[curr], 16, 256, 6);

        // Copy scales from smem to tmem
        copy_scales_smem_to_tmem(sfa_smem_addr[curr], sfb_smem_addr[curr], tmem_sfa, tmem_sfb);
        __syncthreads();
        // Issue MMA and wait
        issue_mma(tmem_d, a_desc, b_desc, idesc, tmem_sfa, tmem_sfb, mbar_mma, k_tile > 0);
        wait_mma(mbar_mma, mma_phase);

    }

    load_accum_and_store(params, m_block, n_block, tmem_d);
    dealloc_tmem(tmem_d);
}

// ============================================================================
// HOST: Create TensorMap with 32B swizzle
// ============================================================================
void create_tensormap(CUtensorMap* tensormap, const void* data_ptr,
                      int rows, int k_bytes, int row_stride_bytes, int box_rows)
{
    uint64_t globalDim[2] = {(uint64_t)k_bytes, (uint64_t)rows};
    uint64_t globalStride[1] = {(uint64_t)row_stride_bytes};
    uint32_t boxDim[2] = {32, (uint32_t)box_rows};  // 32 bytes for 32B swizzle
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
        CU_TENSOR_MAP_SWIZZLE_32B,  // 32B hardware swizzle
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        throw std::runtime_error(std::string("cuTensorMapEncodeTiled failed: ") + errStr);
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

    Gemm_params params;
    params.M = M; params.N = N; params.K = K;
    params.sfa_ptr = SFA.data_ptr();
    params.sfb_ptr = SFB.data_ptr();
    params.c_ptr = reinterpret_cast<__half*>(C.data_ptr());
    params.sfa_row_stride = SFA.stride(0);
    params.sfb_row_stride = SFB.stride(0);

    // Create tensormaps for A and B with 32B swizzle
    create_tensormap(&params.tensormap_a, A.data_ptr(), M, K_bytes, A.stride(0), MMA_M);
    create_tensormap(&params.tensormap_b, B.data_ptr(), N, K_bytes, B.stride(0), MMA_N);

    dim3 grid_dim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M, 1);
    dim3 block_dim(WARPS_PER_CTA * THREADS_PER_WARP);

    gemm_kernel_tcgen05<<<grid_dim, block_dim>>>(params);
    return C;
}
"""

nvfp4_tcgen05_module = load_inline(
    name="nvfp4_tcgen05_gemm_double_buffered",
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
