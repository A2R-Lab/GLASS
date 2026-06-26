// bench_gemm.cu — L3 benchmarks: glass plain vs. tiled vs. glass CT vs. cuBLASDx
// Variants per size:
//   glass plain (global)  — reads A,B,C from global memory each call
//   glass plain (shared)  — data pre-loaded in shared; measures pure compute
//   glass tiled (shared)  — tiled gemm with scratch already in shared
//   glass<CT>   (global)  — compile-time M/N/K; reads from global memory
//   glass<CT>   (shared)  — compile-time M/N/K; data pre-loaded in shared
//   cuBLASDx (global)     — full round-trip: global→shared→execute→global per call
//   cuBLASDx (shared)     — data pre-loaded in shared; measures pure execute
//
// Compilation (with cuBLASDx):
//   nvcc -std=c++17 -arch=sm_XX -O3
//        -I.. -I../src
//        -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
//        -DGLASS_BENCH_CUBLASDX -DSMS=XX0
//        -Xptxas -O1
//        bench_gemm.cu -o bench_gemm
// Usage: ./bench_gemm [m [n [k [iters]]]]

#include <cstdio>
#include <cstdlib>
#include <ctime>

// Bring in cuBLASDx FIRST at global scope so glass-nvidia.cuh's namespace block
// doesn't nest its symbols. glass-nvidia.cuh includes glass.cuh transitively.
#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#include "../glass-nvidia.cuh"
#else
#include "../glass.cuh"
#endif

static const int THREADS = 256;
static const int TILE    = 8;

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// ─── GLASS kernels ────────────────────────────────────────────────────────────

// (global): reads A, B, C from global memory each call
__global__ void k_glass_gemm_global(float* A, float* B, float* C, int m, int n, int k, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        glass::gemm<float>(m, n, k, 1.f, A, B, 1.f, C);
    }
}

// plain (shared): data pre-loaded into shared once; measures pure compute
__global__ void k_glass_gemm_shared(float* A, float* B, float* C, int m, int n, int k, int iters) {
    extern __shared__ float smem[];
    float* s_A = smem;               // m*n floats
    float* s_B = smem + m * n;       // n*k floats
    float* s_C = smem + m * n + n * k; // m*k floats
    for (int i = threadIdx.x; i < m * n; i += blockDim.x) s_A[i] = A[i];
    for (int i = threadIdx.x; i < n * k; i += blockDim.x) s_B[i] = B[i];
    for (int i = threadIdx.x; i < m * k; i += blockDim.x) s_C[i] = C[i];
    __syncthreads();
    for (int rep = 0; rep < iters; rep++) {
        glass::gemm<float>(m, n, k, 1.f, s_A, s_B, 1.f, s_C);
    }
    if (threadIdx.x == 0) C[0] = s_C[0]; // prevent DCE
}

// tiled (shared): tiled gemm with scratch buffers in shared (already in shared use case)
__global__ void k_glass_gemm_tiled(float* A, float* B, float* C, int m, int n, int k, int iters) {
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_B = smem + m * TILE;
    for (int rep = 0; rep < iters; rep++) {
        glass::gemm_tiled<float, TILE>(m, n, k, 1.f, A, B, 1.f, C, s_A, s_B);
    }
}

// ─── GLASS compile-time kernels (compile-time M/N/K) ─────────────────────────

#define DEFINE_GLASS_GEMM_CT(M, N, K)                                                        \
    namespace glass_gemm_ct_##M##x##N##x##K {                                               \
        static const int smem_size = (M*N + N*K + M*K) * sizeof(float);                     \
        __global__ void kernel_global(float* A, float* B, float* C, int iters) {             \
            for (int rep = 0; rep < iters; rep++)                                             \
                glass::gemm<float, M, N, K>(1.f, A, B, 1.f, C);                             \
        }                                                                                     \
        __global__ void kernel_smem(float* A, float* B, float* C, int iters) {               \
            extern __shared__ float smem[];                                                   \
            float* s_A = smem;                                                               \
            float* s_B = smem + M*N;                                                         \
            float* s_C = smem + M*N + N*K;                                                   \
            for (int i = threadIdx.x; i < M*N; i += blockDim.x) s_A[i] = A[i];             \
            for (int i = threadIdx.x; i < N*K; i += blockDim.x) s_B[i] = B[i];             \
            for (int i = threadIdx.x; i < M*K; i += blockDim.x) s_C[i] = C[i];             \
            __syncthreads();                                                                  \
            for (int rep = 0; rep < iters; rep++)                                             \
                glass::gemm<float, M, N, K>(1.f, s_A, s_B, 1.f, s_C);                      \
            if (threadIdx.x == 0) C[0] = s_C[0];                                            \
        }                                                                                     \
    }

DEFINE_GLASS_GEMM_CT(4,  4,  4)
DEFINE_GLASS_GEMM_CT(6,  6,  6)
DEFINE_GLASS_GEMM_CT(8,  8,  8)
DEFINE_GLASS_GEMM_CT(12, 12, 12)
DEFINE_GLASS_GEMM_CT(14, 14, 14)
DEFINE_GLASS_GEMM_CT(24, 24, 24)
DEFINE_GLASS_GEMM_CT(64, 64, 64)
// packed: A(4×4) * B(4×K) = C(4×K)
DEFINE_GLASS_GEMM_CT(4,  4,  16)
DEFINE_GLASS_GEMM_CT(4,  4,  32)
DEFINE_GLASS_GEMM_CT(4,  4,  48)
DEFINE_GLASS_GEMM_CT(4,  4,  64)

#define RUN_GLASS_GEMM_CT(M, N, K, dA, dB, dC, iters, t0, t1)                               \
    cudaMemset(dC, 0, (size_t)M*K*sizeof(float));                                             \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                                    \
    glass_gemm_ct_##M##x##N##x##K::kernel_global<<<1, THREADS>>>(dA, dB, dC, iters);        \
    cudaDeviceSynchronize();                                                                  \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                                    \
    printf("glass::gemm<CT> (global) m=%2d n=%2d k=%2d  %.3f us/op\n",                      \
           M, N, K, elapsed_us(t0, t1) / iters);                                              \
    cudaMemset(dC, 0, (size_t)M*K*sizeof(float));                                             \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                                    \
    glass_gemm_ct_##M##x##N##x##K::kernel_smem<<<1, THREADS,                                 \
        glass_gemm_ct_##M##x##N##x##K::smem_size>>>(dA, dB, dC, iters);                     \
    cudaDeviceSynchronize();                                                                  \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                                    \
    printf("glass::gemm<CT> (shared) m=%2d n=%2d k=%2d  %.3f us/op\n",                      \
           M, N, K, elapsed_us(t0, t1) / iters);

// ─── cuBLASDx kernels (compile-time M/N/K) ────────────────────────────────────
#ifdef GLASS_BENCH_CUBLASDX

#ifndef SMS
#define SMS 860
#endif

#define DEFINE_CUBLASDX_GEMM(M, N, K)                                                       \
    namespace cublasdx_gemm_##M##x##N##x##K {                                               \
        using GEMM = decltype(                                                               \
            cublasdx::Size<M, N, K>()                                                        \
            + cublasdx::Precision<float>()                                                   \
            + cublasdx::Type<cublasdx::type::real>()                                         \
            + cublasdx::Function<cublasdx::function::MM>()                                   \
            + cublasdx::SM<SMS>()                                                             \
            + cublasdx::Block());                                                             \
        /* (global): full round-trip global→shared→execute→global per iteration */          \
        __launch_bounds__(GEMM::max_threads_per_block)                                       \
        __global__ void kernel_global(float* A, float* B, float* C, int iters) {            \
            extern __shared__ __align__(16) char smem[];                                     \
            using align = cublasdx::alignment_of<GEMM>;                                     \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);      \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());         \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());         \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());         \
            for (int rep = 0; rep < iters; rep++) {                                          \
                cublasdx::copy<GEMM, align::a>(                                              \
                    cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);           \
                cublasdx::copy<GEMM, align::b>(                                              \
                    cublasdx::make_tensor(B, GEMM::get_layout_gmem_b()), b_smem);           \
                cublasdx::copy<GEMM, align::c>(                                              \
                    cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()), c_smem);           \
                cublasdx::copy_wait();                                                       \
                GEMM().execute(1.f, a_smem, b_smem, 1.f, c_smem);                           \
                cublasdx::copy<GEMM, align::c>(                                              \
                    c_smem, cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()));           \
                __syncthreads();                                                             \
            }                                                                                \
        }                                                                                    \
        /* (shared): data pre-loaded once; measures pure execute */                          \
        __launch_bounds__(GEMM::max_threads_per_block)                                       \
        __global__ void kernel_smem(float* A, float* B, float* C, int iters) {              \
            extern __shared__ __align__(16) char smem[];                                     \
            using align = cublasdx::alignment_of<GEMM>;                                     \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);      \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());         \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());         \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());         \
            cublasdx::copy<GEMM, align::a>(                                                  \
                cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);               \
            cublasdx::copy<GEMM, align::b>(                                                  \
                cublasdx::make_tensor(B, GEMM::get_layout_gmem_b()), b_smem);               \
            cublasdx::copy<GEMM, align::c>(                                                  \
                cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()), c_smem);               \
            cublasdx::copy_wait();                                                           \
            for (int rep = 0; rep < iters; rep++) {                                          \
                GEMM().execute(1.f, a_smem, b_smem, 1.f, c_smem);                           \
                __syncthreads();                                                             \
            }                                                                                \
            cublasdx::copy<GEMM, align::c>(                                                  \
                c_smem, cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()));               \
        }                                                                                    \
        static constexpr auto smem_size = cublasdx::get_shared_storage_size<GEMM>();        \
    }

DEFINE_CUBLASDX_GEMM(4,  4,  4)
DEFINE_CUBLASDX_GEMM(6,  6,  6)
DEFINE_CUBLASDX_GEMM(8,  8,  8)
DEFINE_CUBLASDX_GEMM(12, 12, 12)
DEFINE_CUBLASDX_GEMM(14, 14, 14)
DEFINE_CUBLASDX_GEMM(24, 24, 24)
DEFINE_CUBLASDX_GEMM(64, 64, 64)
// packed: A(4×4) * B(4×K) = C(4×K)
DEFINE_CUBLASDX_GEMM(4,  4,  16)
DEFINE_CUBLASDX_GEMM(4,  4,  32)
DEFINE_CUBLASDX_GEMM(4,  4,  48)
DEFINE_CUBLASDX_GEMM(4,  4,  64)

#define RUN_CUBLASDX_GEMM(M, N, K, dA, dB, dC, iters, t0, t1)                        \
    cudaMemset(dC, 0, (size_t)M*K*sizeof(float));                                     \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                             \
    cublasdx_gemm_##M##x##N##x##K::kernel_global<<<1,                                 \
        cublasdx_gemm_##M##x##N##x##K::GEMM::block_dim,                               \
        cublasdx_gemm_##M##x##N##x##K::smem_size>>>(dA, dB, dC, iters);              \
    cudaDeviceSynchronize();                                                           \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                             \
    printf("cuBLASDx gemm (global)       m=%2d n=%2d k=%2d  %.3f us/op\n",           \
           M, N, K, elapsed_us(t0, t1) / iters);                                      \
    cudaMemset(dC, 0, (size_t)M*K*sizeof(float));                                     \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                             \
    cublasdx_gemm_##M##x##N##x##K::kernel_smem<<<1,                                   \
        cublasdx_gemm_##M##x##N##x##K::GEMM::block_dim,                               \
        cublasdx_gemm_##M##x##N##x##K::smem_size>>>(dA, dB, dC, iters);              \
    cudaDeviceSynchronize();                                                           \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                             \
    printf("cuBLASDx gemm (shared)       m=%2d n=%2d k=%2d  %.3f us/op\n",           \
           M, N, K, elapsed_us(t0, t1) / iters);

// ─── glass::nvidia kernels (compile-time M/N/K, alongside raw cuBLASDx) ──────
// Two variants per size:
//   _nv_default — uses glass::nvidia::gemm<T,M,N,K>(...)            (no BlockDim)
//   _nv_blockdim — uses glass::nvidia::gemm<T,M,N,K,THREADS>(...)   (BlockDim<THREADS>)
//
// Both write to a volatile sink each iteration to defeat dead-store elimination.

namespace glass { namespace nvidia {
    // BlockDim<THREADS> variants for all sizes (default-block_dim variants for
    // the square sizes 4..64 are already pre-instantiated in glass-nvidia.cuh).
    DEFINE_NVIDIA_GEMM_BLOCKDIM(4,  4,  4,  THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(6,  6,  6,  THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(8,  8,  8,  THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(12, 12, 12, THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(14, 14, 14, THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(24, 24, 24, THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(64, 64, 64, THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(4,  4,  16, THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(4,  4,  32, THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(4,  4,  48, THREADS)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(4,  4,  64, THREADS)
    // Default-block_dim variants for the rectangular 4×4×K shapes (square ones
    // already pre-instantiated in glass-nvidia.cuh).
    DEFINE_NVIDIA_GEMM(4, 4, 16)
    DEFINE_NVIDIA_GEMM(4, 4, 32)
    DEFINE_NVIDIA_GEMM(4, 4, 48)
    DEFINE_NVIDIA_GEMM(4, 4, 64)
}} // namespace glass::nvidia

template<int M, int N, int K>
__global__ void k_nv_gemm_default(float* A, float* B, float* C, volatile float* sink, int iters) {
    extern __shared__ __align__(16) char nv_smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm<float, M, N, K>(1.f, A, B, 0.f, C, nv_smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C[0];
        __syncthreads();
    }
}

template<int M, int N, int K, int TC>
__global__ void k_nv_gemm_blockdim(float* A, float* B, float* C, volatile float* sink, int iters) {
    extern __shared__ __align__(16) char nv_smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm<float, M, N, K, TC>(1.f, A, B, 0.f, C, nv_smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C[0];
        __syncthreads();
    }
}

#define RUN_NVIDIA_GEMM(M, N, K, dA, dB, dC, sink, iters, t0, t1)                            \
    {                                                                                         \
        constexpr auto smem_def = glass::nvidia::gemm_scratch_bytes<float, M, N, K>();           \
        constexpr auto thr_def  = glass::nvidia::gemm_threads<float, M, N, K>();             \
        cudaMemset(dC, 0, (size_t)M*K*sizeof(float));                                         \
        clock_gettime(CLOCK_MONOTONIC, &(t0));                                                 \
        k_nv_gemm_default<M, N, K><<<1, thr_def, smem_def>>>(dA, dB, dC, sink, iters);       \
        cudaDeviceSynchronize();                                                               \
        clock_gettime(CLOCK_MONOTONIC, &(t1));                                                 \
        printf("glass::nvidia gemm (default) m=%2d n=%2d k=%2d  %.3f us/op\n",               \
               M, N, K, elapsed_us(t0, t1) / iters);                                          \
        constexpr auto smem_bd = glass::nvidia::gemm_scratch_bytes<float, M, N, K, THREADS>();   \
        cudaMemset(dC, 0, (size_t)M*K*sizeof(float));                                         \
        clock_gettime(CLOCK_MONOTONIC, &(t0));                                                 \
        k_nv_gemm_blockdim<M, N, K, THREADS><<<1, THREADS, smem_bd>>>(dA, dB, dC, sink, iters); \
        cudaDeviceSynchronize();                                                               \
        clock_gettime(CLOCK_MONOTONIC, &(t1));                                                 \
        printf("glass::nvidia gemm (TC=%d)  m=%2d n=%2d k=%2d  %.3f us/op\n",                \
               THREADS, M, N, K, elapsed_us(t0, t1) / iters);                                 \
    }

#endif // GLASS_BENCH_CUBLASDX

// ─── Benchmark runner ─────────────────────────────────────────────────────────

static void bench_size(int m, int n, int k, int iters) {
    float *dA, *dB, *dC;
    float *dSink;
    cudaMalloc(&dA, m * n * sizeof(float));
    cudaMalloc(&dB, n * k * sizeof(float));
    cudaMalloc(&dC, m * k * sizeof(float));
    cudaMalloc(&dSink, 256 * sizeof(float));   // for anti-DSE writes
    (void)dSink;

    struct timespec t0, t1;
    size_t c_bytes = (size_t)m * k * sizeof(float);

    // glass plain (global)
    cudaMemset(dC, 0, c_bytes);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_gemm_global<<<1, THREADS>>>(dA, dB, dC, m, n, k, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::gemm (global) m=%2d n=%2d k=%2d  %.3f us/op\n",
           m, n, k, elapsed_us(t0, t1) / iters);

    // glass plain (shared)
    int glass_smem = (m * n + n * k + m * k) * sizeof(float);
    cudaMemset(dC, 0, c_bytes);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_gemm_shared<<<1, THREADS, glass_smem>>>(dA, dB, dC, m, n, k, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::gemm (shared) m=%2d n=%2d k=%2d  %.3f us/op\n",
           m, n, k, elapsed_us(t0, t1) / iters);

    // glass tiled (only when m*k <= THREADS; already uses shared scratch)
    if (m * k <= THREADS) {
        int tiled_smem = (m * TILE + TILE * k) * sizeof(float);
        cudaMemset(dC, 0, c_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        k_glass_gemm_tiled<<<1, THREADS, tiled_smem>>>(dA, dB, dC, m, n, k, iters);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("glass::gemm (tiled)  m=%2d n=%2d k=%2d  %.3f us/op\n",
               m, n, k, elapsed_us(t0, t1) / iters);
    }

    // glass compile-time variants (only for pre-instantiated sizes)
    #define MAYBE_GLASS_GEMM_CT(M, N, K)                                           \
        if (m == M && n == N && k == K) { RUN_GLASS_GEMM_CT(M, N, K, dA, dB, dC, iters, t0, t1); }
    MAYBE_GLASS_GEMM_CT(4,  4,  4)
    MAYBE_GLASS_GEMM_CT(6,  6,  6)
    MAYBE_GLASS_GEMM_CT(8,  8,  8)
    MAYBE_GLASS_GEMM_CT(12, 12, 12)
    MAYBE_GLASS_GEMM_CT(14, 14, 14)
    MAYBE_GLASS_GEMM_CT(24, 24, 24)
    MAYBE_GLASS_GEMM_CT(64, 64, 64)
    MAYBE_GLASS_GEMM_CT(4,  4,  16)
    MAYBE_GLASS_GEMM_CT(4,  4,  32)
    MAYBE_GLASS_GEMM_CT(4,  4,  48)
    MAYBE_GLASS_GEMM_CT(4,  4,  64)
    #undef MAYBE_GLASS_GEMM_CT

#ifdef GLASS_BENCH_CUBLASDX
    #define MAYBE_CUBLASDX_GEMM(M, N, K)                                           \
        if (m == M && n == N && k == K) { RUN_CUBLASDX_GEMM(M, N, K, dA, dB, dC, iters, t0, t1); }
    MAYBE_CUBLASDX_GEMM(4,  4,  4)
    MAYBE_CUBLASDX_GEMM(6,  6,  6)
    MAYBE_CUBLASDX_GEMM(8,  8,  8)
    MAYBE_CUBLASDX_GEMM(12, 12, 12)
    MAYBE_CUBLASDX_GEMM(14, 14, 14)
    MAYBE_CUBLASDX_GEMM(24, 24, 24)
    MAYBE_CUBLASDX_GEMM(64, 64, 64)
    MAYBE_CUBLASDX_GEMM(4,  4,  16)
    MAYBE_CUBLASDX_GEMM(4,  4,  32)
    MAYBE_CUBLASDX_GEMM(4,  4,  48)
    MAYBE_CUBLASDX_GEMM(4,  4,  64)
    #undef MAYBE_CUBLASDX_GEMM

    // glass::nvidia variants (default block_dim + caller-pinned BlockDim<THREADS>)
    #define MAYBE_NVIDIA_GEMM(M, N, K)                                             \
        if (m == M && n == N && k == K) { RUN_NVIDIA_GEMM(M, N, K, dA, dB, dC, dSink, iters, t0, t1); }
    MAYBE_NVIDIA_GEMM(4,  4,  4)
    MAYBE_NVIDIA_GEMM(6,  6,  6)
    MAYBE_NVIDIA_GEMM(8,  8,  8)
    MAYBE_NVIDIA_GEMM(12, 12, 12)
    MAYBE_NVIDIA_GEMM(14, 14, 14)
    MAYBE_NVIDIA_GEMM(24, 24, 24)
    MAYBE_NVIDIA_GEMM(64, 64, 64)
    MAYBE_NVIDIA_GEMM(4,  4,  16)
    MAYBE_NVIDIA_GEMM(4,  4,  32)
    MAYBE_NVIDIA_GEMM(4,  4,  48)
    MAYBE_NVIDIA_GEMM(4,  4,  64)
    #undef MAYBE_NVIDIA_GEMM
#endif

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dSink);
}

int main(int argc, char** argv) {
    int m     = (argc > 1) ? atoi(argv[1]) : 0;
    int n     = (argc > 2) ? atoi(argv[2]) : 0;
    int k     = (argc > 3) ? atoi(argv[3]) : 0;
    int iters = (argc > 4) ? atoi(argv[4]) : 100000;

    if (m > 0 && n > 0 && k > 0) {
        bench_size(m, n, k, iters);
    } else {
        int sizes[] = {4, 6, 8, 12, 14, 24, 64};
        for (int s : sizes) bench_size(s, s, s, iters);
        // packed rectangular sweep: A(4×4) * B(4×K) = C(4×K)
        int packed_k[] = {16, 32, 48, 64};
        for (int pk : packed_k) bench_size(4, 4, pk, iters);
    }

    return 0;
}
