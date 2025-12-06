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
static constexpr int TMEM_COLS = 128;  // Power of 2, fits D (64 cols) + scales
static constexpr int SF_ROW_STRIDE = 16;  // 16-byte aligned for descriptors

// ============================================================================
// Shared Memory Layout (static allocation)
// ============================================================================
#define SMEM_TMEM_ADDR   4                                    // uint32_t for tmem base addr
#define SMEM_MBAR        8                                    // uint64_t mbarrier (8-byte aligned)
#define SMEM_A_TILE      (MMA_M * MMA_K / 2)                  // A tile: 128*64/2 = 4096 bytes
#define SMEM_B_TILE      (MMA_N * MMA_K / 2)                  // B tile: 64*64/2 = 2048 bytes
#define SMEM_SFA         (MMA_M * SF_ROW_STRIDE)              // SFA: 128*16 = 2048 bytes
#define SMEM_SFB         (MMA_N * SF_ROW_STRIDE)              // SFB: 64*16 = 1024 bytes

#define SMEM_OFF_MBAR    8                                    // mbar at offset 8 (8-byte aligned)
#define SMEM_OFF_TILES   16                                   // tiles start after mbar
#define SMEM_TOTAL       (SMEM_OFF_TILES + SMEM_A_TILE + SMEM_B_TILE + SMEM_SFA + SMEM_SFB)

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

#define DEBUG_PRINT(fmt, ...) \\
    do { \\
        if (blockIdx.x == params.debug_block_x && \\
            blockIdx.y == params.debug_block_y && \\
            threadIdx.x == 0) { \\
            printf(fmt, ##__VA_ARGS__); \\
        } \\
    } while(0)

// ============================================================================
// Descriptor Helpers
// ============================================================================
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
    idesc |= (1U << 7);                           // atype = E2M1
    idesc |= (1U << 10);                          // btype = E2M1
    idesc |= (1U << 16);                          // Transpose B (B is NxK, MMA expects KxN)
    idesc |= ((N_tile >> 3) & 0x3F) << 17;        // N >> 3
    idesc |= (0U) << 23;                          // UE4M3 scale type
    idesc |= ((M_tile >> 7) & 0x3) << 27;         // M >> 7
    return idesc;
}

// ============================================================================
// Main Kernel
// ============================================================================
__global__ void __launch_bounds__(WARPS_PER_CTA * THREADS_PER_WARP)
gemm_kernel_tcgen05(const __grid_constant__ Gemm_params params)
{
    // =========================================================================
    // SHARED MEMORY LAYOUT
    // =========================================================================
    __shared__ char smem[SMEM_TOTAL];
    uint32_t* tmem_addr_storage = reinterpret_cast<uint32_t*>(smem);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + SMEM_OFF_MBAR);
    char* tile_smem = smem + SMEM_OFF_TILES;
    const uint32_t smem_base = __cvta_generic_to_shared(smem);
    const uint32_t mbar_smem = smem_base + SMEM_OFF_MBAR;

    const int m_block = blockIdx.y * MMA_M;
    const int n_block = blockIdx.x * MMA_N;
    if (m_block >= params.M || n_block >= params.N) return;


    // =========================================================================
    // TMEM ALLOCATION
    // =========================================================================
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\\n"
            :: "r"(smem_base), "r"(TMEM_COLS) : "memory"
        );
    }
    __syncthreads();
    uint32_t tmem_d = *tmem_addr_storage;
    uint32_t tmem_sfa = tmem_d + MMA_N;
    uint32_t tmem_sfb = tmem_d + MMA_N + 1;

    DEBUG_PRINT("[INIT] smem_base=0x%x, tmem_d=0x%x, tmem_sfa=0x%x, tmem_sfb=0x%x\\n",
                smem_base, tmem_d, tmem_sfa, tmem_sfb);


    // =========================================================================
    // MBARRIER INIT
    // =========================================================================
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\\n"
            :: "r"(mbar_smem), "r"(1) : "memory");
    }
    __syncthreads();


    // =========================================================================
    // INSTRUCTION DESCRIPTOR
    // =========================================================================
    uint32_t idesc = make_mxf4_idesc(MMA_M, MMA_N);
    DEBUG_PRINT("[INIT] idesc=0x%08x\\n", idesc);


    // =========================================================================
    // K-LOOP
    // =========================================================================
    const int num_k_tiles = params.K / MMA_K;
    int mbar_phase = 0;
    DEBUG_PRINT("[INIT] num_k_tiles=%d\\n", num_k_tiles);

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int k_offset = k_tile * MMA_K;


        // =====================================================================
        // LOAD A TILE TO SMEM
        // =====================================================================
        const uint8_t* a_global = reinterpret_cast<const uint8_t*>(params.a_ptr)
                                  + m_block * params.a_row_stride + k_offset / 2;
        uint8_t* a_smem = reinterpret_cast<uint8_t*>(tile_smem);

        for (int i = threadIdx.x; i < MMA_M * MMA_K / 2; i += blockDim.x) {
            int m = i / (MMA_K / 2);
            int k = i % (MMA_K / 2);
            a_smem[m * (MMA_K / 2) + k] = a_global[m * params.a_row_stride + k];
        }


        // =====================================================================
        // LOAD B TILE TO SMEM
        // =====================================================================
        const uint8_t* b_global = reinterpret_cast<const uint8_t*>(params.b_ptr)
                                  + n_block * params.b_row_stride + k_offset / 2;
        uint8_t* b_smem = reinterpret_cast<uint8_t*>(tile_smem + MMA_M * MMA_K / 2);

        for (int i = threadIdx.x; i < MMA_N * MMA_K / 2; i += blockDim.x) {
            int n = i / (MMA_K / 2);
            int k = i % (MMA_K / 2);
            b_smem[n * (MMA_K / 2) + k] = b_global[n * params.b_row_stride + k];
        }


        // =====================================================================
        // LOAD SCALE FACTORS TO SMEM (16-byte padded rows)
        // =====================================================================
        const int sf_k_idx = k_offset / 16;
        const uint8_t* sfa_global = reinterpret_cast<const uint8_t*>(params.sfa_ptr)
                                    + m_block * params.sfa_row_stride + sf_k_idx;
        const uint8_t* sfb_global = reinterpret_cast<const uint8_t*>(params.sfb_ptr)
                                    + n_block * params.sfb_row_stride + sf_k_idx;

        uint8_t* sfa_smem = reinterpret_cast<uint8_t*>(tile_smem + (MMA_M + MMA_N) * MMA_K / 2);
        uint8_t* sfb_smem = sfa_smem + MMA_M * SF_ROW_STRIDE;

        for (int i = threadIdx.x; i < MMA_M * SF_ROW_STRIDE; i += blockDim.x) {
            int m = i / SF_ROW_STRIDE;
            int s = i % SF_ROW_STRIDE;
            sfa_smem[i] = (s < 4) ? sfa_global[m * params.sfa_row_stride + s] : 0;
        }

        for (int i = threadIdx.x; i < MMA_N * SF_ROW_STRIDE; i += blockDim.x) {
            int n = i / SF_ROW_STRIDE;
            int s = i % SF_ROW_STRIDE;
            sfb_smem[i] = (s < 4) ? sfb_global[n * params.sfb_row_stride + s] : 0;
        }

        __syncthreads();

        if (k_tile == 0) {
            DEBUG_PRINT("[K0] Tiles loaded to SMEM, a_smem=%p, b_smem=%p\\n", a_smem, b_smem);
            DEBUG_PRINT("[K0] sfa_smem=%p, sfb_smem=%p\\n", sfa_smem, sfb_smem);
        }

        // =====================================================================
        // BUILD SMEM DESCRIPTORS
        // =====================================================================
        uint64_t a_desc = make_smem_desc(a_smem, MMA_K / 2, MMA_K / 2, 0);
        uint64_t b_desc = make_smem_desc(b_smem, MMA_K / 2, MMA_K / 2, 0);
        uint64_t sfa_desc = make_smem_desc(sfa_smem, SF_ROW_STRIDE, SF_ROW_STRIDE, 0);
        uint64_t sfb_desc = make_smem_desc(sfb_smem, SF_ROW_STRIDE, SF_ROW_STRIDE, 0);

        if (k_tile == 0) {
            DEBUG_PRINT("[K0] a_desc=0x%016llx, b_desc=0x%016llx\\n",
                        (unsigned long long)a_desc, (unsigned long long)b_desc);
            DEBUG_PRINT("[K0] sfa_desc=0x%016llx, sfb_desc=0x%016llx\\n",
                        (unsigned long long)sfa_desc, (unsigned long long)sfb_desc);
        }


        // =====================================================================
        // COPY SCALE FACTORS: SMEM -> TMEM
        // =====================================================================
        if (k_tile == 0) {
            DEBUG_PRINT("[K0] Before tcgen05.cp SFA: tmem_sfa=0x%x\\n", tmem_sfa);
        }
        if (threadIdx.x == 0) {
            // SFA: 128 lanes via 128x128b
            asm volatile(
                "tcgen05.cp.cta_group::1.128x128b [%0], %1;\\n"
                :: "r"(tmem_sfa), "l"(sfa_desc) : "memory"
            );
        }
        if (k_tile == 0) {
            DEBUG_PRINT("[K0] After tcgen05.cp SFA\\n");
            DEBUG_PRINT("[K0] Before tcgen05.cp SFB: tmem_sfb=0x%x\\n", tmem_sfb);
        }
        if (threadIdx.x == 0) {
            // SFB: 64 lanes via 64x128b with warpx2 multicast
            asm volatile(
                "tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\\n"
                :: "r"(tmem_sfb), "l"(sfb_desc) : "memory"
            );
        }
        if (k_tile == 0) {
            DEBUG_PRINT("[K0] After tcgen05.cp SFB\\n");
            DEBUG_PRINT("[K0] Before tcgen05.commit SF: mbar=%p\\n", mbar);
        }
        if (threadIdx.x == 0) {
            // Commit scale factor copies
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\\n"
                :: "l"((uint64_t)mbar) : "memory"
            );
        }
        __syncthreads();

        if (k_tile == 0) {
            DEBUG_PRINT("[K0] After tcgen05.commit SF, waiting on mbar_smem=0x%x, phase=%d\\n", mbar_smem, mbar_phase);
        }

        // Wait for scale factor copies
        uint32_t done = 0;
        while (!done) {
            asm volatile(
                "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}\\n"
                : "=r"(done) : "r"(mbar_smem), "r"(mbar_phase) : "memory"
            );
        }
        mbar_phase ^= 1;

        if (k_tile == 0) {
            DEBUG_PRINT("[K0] SF copy done, mbar_phase now=%d\\n", mbar_phase);
        }


        // =====================================================================
        // ISSUE MMA
        // =====================================================================
        uint32_t enable_d = (k_tile > 0) ? 1 : 0;

        if (k_tile == 0) {
            DEBUG_PRINT("[K0] Before tcgen05.mma: tmem_d=0x%x, enable_d=%u\\n", tmem_d, enable_d);
        }
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
        }
        if (k_tile == 0) {
            DEBUG_PRINT("[K0] After tcgen05.mma\\n");
            DEBUG_PRINT("[K0] Before tcgen05.commit MMA\\n");
        }
        if (threadIdx.x == 0) {
            // Commit MMA
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\\n"
                :: "l"((uint64_t)mbar) : "memory"
            );
        }
        __syncthreads();

        if (k_tile == 0) {
            DEBUG_PRINT("[K0] After tcgen05.commit MMA, waiting phase=%d\\n", mbar_phase);
        }

        // Wait for MMA
        done = 0;
        while (!done) {
            asm volatile(
                "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}\\n"
                : "=r"(done) : "r"(mbar_smem), "r"(mbar_phase) : "memory"
            );
        }
        mbar_phase ^= 1;

        if (k_tile == 0) {
            DEBUG_PRINT("[K0] MMA done, mbar_phase now=%d\\n", mbar_phase);
        }

        __syncthreads();
    }


    // =========================================================================
    // LOAD RESULTS FROM TMEM TO REGISTERS
    // =========================================================================
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int out_m = warp_id * 32 + lane_id;

    float acc_regs[MMA_N];

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

    DEBUG_PRINT("[LOAD] acc_regs[0]=%.2f, acc_regs[1]=%.2f, acc_regs[2]=%.2f, acc_regs[3]=%.2f\\n",
                acc_regs[0], acc_regs[1], acc_regs[2], acc_regs[3]);


    // =========================================================================
    // STORE RESULTS TO GLOBAL MEMORY
    // =========================================================================
    if (m_block + out_m < params.M) {
        #pragma unroll
        for (int n = 0; n < MMA_N; n++) {
            if (n_block + n < params.N) {
                params.c_ptr[(m_block + out_m) * params.N + (n_block + n)] =
                    __float2half(acc_regs[n]);
            }
        }
    }


    // =========================================================================
    // TMEM DEALLOCATION
    // =========================================================================
    if (threadIdx.x < 32) {
        asm volatile(
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\\n"
            :: "r"(tmem_d), "r"(TMEM_COLS) : "memory"
        );
    }
    __syncthreads();
}

// ============================================================================
// Host Launcher
// ============================================================================
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
