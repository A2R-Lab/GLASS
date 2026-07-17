// bench_gemv.cu — L2 benchmarks: glass::gemv vs. cuBLASDx
// Variants per size:
//   glass (global)    — reads A,x from global memory each call
//   glass (shared)    — data pre-loaded in shared; measures pure compute
//   glass<CT>(global) — compile-time M/N; reads from global memory
//   glass<CT>(shared) — compile-time M/N; data pre-loaded in shared
//   cuBLASDx (global) — full round-trip: global→shared→execute→global per call
//   cuBLASDx (shared) — data pre-loaded in shared; measures pure execute
//
// Compilation (with cuBLASDx):
//   nvcc -std=c++17 -arch=sm_XX -O3
//        -I.. -I../src
//        -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
//        -DGLASS_BENCH_CUBLASDX -DSMS=XX0
//        -Xptxas -O1
//        bench_gemv.cu -o bench_gemv
// Usage: ./bench_gemv <m> <n> [iters]

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

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// ─── GLASS kernels ────────────────────────────────────────────────────────────

// (global): reads A, x from global memory each call
__global__ void k_glass_gemv_global(float* A, float* x, float* y, int m, int n, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        glass::gemv<float>(m, n, 1.f, A, x, 1.f, y);
    }
}

// (shared): data pre-loaded into shared once; measures pure compute
__global__ void k_glass_gemv_shared(float* A, float* x, float* y, int m, int n, int iters) {
    extern __shared__ float smem[];
    float* s_A = smem;           // m*n floats
    float* s_x = smem + m * n;   // n floats
    float* s_y = smem + m * n + n; // m floats
    for (int i = threadIdx.x; i < m * n; i += blockDim.x) s_A[i] = A[i];
    for (int i = threadIdx.x; i < n;     i += blockDim.x) s_x[i] = x[i];
    for (int i = threadIdx.x; i < m;     i += blockDim.x) s_y[i] = 0.f;
    __syncthreads();
    for (int rep = 0; rep < iters; rep++) {
        glass::gemv<float>(m, n, 1.f, s_A, s_x, 1.f, s_y);
    }
    if (threadIdx.x == 0) y[0] = s_y[0]; // prevent DCE
}

// ─── GLASS compile-time kernels (compile-time M/N) ───────────────────────────

#define DEFINE_GLASS_GEMV_CT(M, N)                                                           \
    namespace glass_gemv_ct_##M##x##N {                                                     \
        static const int smem_size = (M*N + N + M) * sizeof(float);                         \
        __global__ void kernel_global(float* A, float* x, float* y, int iters) {             \
            for (int rep = 0; rep < iters; rep++)                                             \
                glass::gemv<float, M, N>(1.f, A, x, 1.f, y);                                \
        }                                                                                     \
        __global__ void kernel_smem(float* A, float* x, float* y, int iters) {               \
            extern __shared__ float smem[];                                                   \
            float* s_A = smem;                                                               \
            float* s_x = smem + M*N;                                                         \
            float* s_y = smem + M*N + N;                                                     \
            for (int i = threadIdx.x; i < M*N; i += blockDim.x) s_A[i] = A[i];             \
            for (int i = threadIdx.x; i < N;   i += blockDim.x) s_x[i] = x[i];             \
            for (int i = threadIdx.x; i < M;   i += blockDim.x) s_y[i] = 0.f;              \
            __syncthreads();                                                                  \
            for (int rep = 0; rep < iters; rep++)                                             \
                glass::gemv<float, M, N>(1.f, s_A, s_x, 1.f, s_y);                         \
            if (threadIdx.x == 0) y[0] = s_y[0];                                            \
        }                                                                                     \
    }

DEFINE_GLASS_GEMV_CT(4,  4)
DEFINE_GLASS_GEMV_CT(6,  6)
DEFINE_GLASS_GEMV_CT(8,  8)
DEFINE_GLASS_GEMV_CT(12, 12)
DEFINE_GLASS_GEMV_CT(14, 14)
DEFINE_GLASS_GEMV_CT(24, 24)
DEFINE_GLASS_GEMV_CT(64, 64)

// ─── glass::thread:: tier (one problem per THREAD, 32 packed per warp) ─────────
// A block of THREADS threads runs THREADS independent N×N gemv problems at once —
// thread p owns problem p, operands register-resident (no shared, no barriers).
// Time is amortized PER PROBLEM to expose the low-DOF packing win. Compile-time
// N only, worth it below the N<=7 register-residency ceiling (CLAUDE.md), so the
// tier is instantiated for the small square sizes 4 and 6.
#define DEFINE_THREAD_GEMV_CT(N)                                                             \
    namespace thread_gemv_ct_##N {                                                           \
        __global__ void k_gemv(const float* A, const float* x, volatile float* sink, int iters) { \
            int p = blockIdx.x*blockDim.x + threadIdx.x;                                     \
            float a[N*N], xv[N], yv[N];                                                      \
            for (int i = 0; i < N*N; i++) a[i]  = A[(size_t)p*N*N + i];                      \
            for (int i = 0; i < N;   i++) xv[i] = x[(size_t)p*N + i];                        \
            for (int i = 0; i < N;   i++) yv[i] = 0.f;                                       \
            for (int rep = 0; rep < iters; rep++) {                                          \
                glass::thread::gemv<float, N, N>(1.f, a, xv, 1.f, yv);                       \
                xv[0] = yv[0];   /* feed back so the loop can't be hoisted */                \
            }                                                                                 \
            sink[p & 0xFF] = yv[0];                                                          \
        }                                                                                     \
    }

DEFINE_THREAD_GEMV_CT(4)
DEFINE_THREAD_GEMV_CT(6)

#define RUN_THREAD_GEMV_CT(N, iters, t0, t1)                                                 \
    {                                                                                         \
        float *dAt, *dxt;                                                                    \
        cudaMalloc(&dAt, (size_t)THREADS * N*N * sizeof(float));                             \
        cudaMalloc(&dxt, (size_t)THREADS * N   * sizeof(float));                             \
        float* hAt = new float[(size_t)THREADS * N*N];                                       \
        float* hxt = new float[(size_t)THREADS * N];                                         \
        for (int i = 0; i < THREADS*N*N; i++) hAt[i] = (float)((i % (N*N)) + 1) / (N*N);     \
        for (int i = 0; i < THREADS*N;   i++) hxt[i] = (float)((i % N) + 1) / N;             \
        cudaMemcpy(dAt, hAt, (size_t)THREADS*N*N*sizeof(float), cudaMemcpyHostToDevice);     \
        cudaMemcpy(dxt, hxt, (size_t)THREADS*N  *sizeof(float), cudaMemcpyHostToDevice);     \
        delete[] hAt; delete[] hxt;                                                          \
        float* dSinkT; cudaMalloc(&dSinkT, 256*sizeof(float));                               \
        clock_gettime(CLOCK_MONOTONIC, &(t0));                                                \
        thread_gemv_ct_##N::k_gemv<<<1, THREADS>>>(dAt, dxt, dSinkT, iters);                 \
        cudaDeviceSynchronize();                                                              \
        clock_gettime(CLOCK_MONOTONIC, &(t1));                                                \
        printf("glass::thread::gemv<CT>      m=%2d n=%2d  %.4f us/problem (%d-packed)\n",     \
               N, N, elapsed_us(t0, t1) / ((double)iters * THREADS), THREADS);               \
        cudaFree(dAt); cudaFree(dxt); cudaFree(dSinkT);                                       \
    }

#define RUN_GLASS_GEMV_CT(M, N, dA, dx, dy, iters, t0, t1)                                  \
    cudaMemset(dy, 0, (size_t)M*sizeof(float));                                               \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                                    \
    glass_gemv_ct_##M##x##N::kernel_global<<<1, THREADS>>>(dA, dx, dy, iters);              \
    cudaDeviceSynchronize();                                                                  \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                                    \
    printf("glass::gemv<CT> (global) m=%2d n=%2d  %.3f us/op\n",                            \
           M, N, elapsed_us(t0, t1) / iters);                                                \
    cudaMemset(dy, 0, (size_t)M*sizeof(float));                                               \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                                    \
    glass_gemv_ct_##M##x##N::kernel_smem<<<1, THREADS,                                       \
        glass_gemv_ct_##M##x##N::smem_size>>>(dA, dx, dy, iters);                           \
    cudaDeviceSynchronize();                                                                  \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                                    \
    printf("glass::gemv<CT> (shared) m=%2d n=%2d  %.3f us/op\n",                            \
           M, N, elapsed_us(t0, t1) / iters);

// ─── cuBLASDx kernels (compile-time M/N) ─────────────────────────────────────
#ifdef GLASS_BENCH_CUBLASDX

#ifndef SMS
#define SMS 860
#endif

#define DEFINE_CUBLASDX_GEMV(M, N)                                                          \
    namespace cublasdx_gemv_##M##x##N {                                                     \
        using GEMM = decltype(                                                              \
            cublasdx::Size<M, 1, N>()                                                       \
            + cublasdx::Precision<float>()                                                  \
            + cublasdx::Type<cublasdx::type::real>()                                        \
            + cublasdx::Function<cublasdx::function::MM>()                                  \
            + cublasdx::SM<SMS>()                                                            \
            + cublasdx::Block());                                                            \
        /* (global): full round-trip global→shared→execute→global per iteration */         \
        __launch_bounds__(GEMM::max_threads_per_block)                                      \
        __global__ void kernel_global(float* A, float* x, float* y, int iters) {           \
            extern __shared__ __align__(16) char smem[];                                       \
            using align = cublasdx::alignment_of<GEMM>;                                    \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);     \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());        \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());        \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());        \
            for (int rep = 0; rep < iters; rep++) {                                         \
                cublasdx::copy<GEMM, align::a>(                                             \
                    cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);          \
                cublasdx::copy<GEMM, align::b>(                                             \
                    cublasdx::make_tensor(x, GEMM::get_layout_gmem_b()), b_smem);          \
                cublasdx::copy<GEMM, align::c>(                                             \
                    cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()), c_smem);          \
                cublasdx::copy_wait();                                                      \
                GEMM().execute(1.f, a_smem, b_smem, 1.f, c_smem);                          \
                cublasdx::copy<GEMM, align::c>(                                             \
                    c_smem, cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()));          \
                __syncthreads();                                                            \
            }                                                                               \
        }                                                                                   \
        /* (shared): data pre-loaded once; measures pure execute */                         \
        __launch_bounds__(GEMM::max_threads_per_block)                                      \
        __global__ void kernel_smem(float* A, float* x, float* y, int iters) {             \
            extern __shared__ __align__(16) char smem[];                                       \
            using align = cublasdx::alignment_of<GEMM>;                                    \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);     \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());        \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());        \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());        \
            cublasdx::copy<GEMM, align::a>(                                                 \
                cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);              \
            cublasdx::copy<GEMM, align::b>(                                                 \
                cublasdx::make_tensor(x, GEMM::get_layout_gmem_b()), b_smem);              \
            cublasdx::copy<GEMM, align::c>(                                                 \
                cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()), c_smem);              \
            cublasdx::copy_wait();                                                          \
            for (int rep = 0; rep < iters; rep++) {                                         \
                GEMM().execute(1.f, a_smem, b_smem, 1.f, c_smem);                          \
                __syncthreads();                                                            \
            }                                                                               \
            cublasdx::copy<GEMM, align::c>(                                                 \
                c_smem, cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()));              \
        }                                                                                   \
        static constexpr auto smem_size = cublasdx::get_shared_storage_size<GEMM>();       \
    }

DEFINE_CUBLASDX_GEMV(4,  4)
DEFINE_CUBLASDX_GEMV(6,  6)
DEFINE_CUBLASDX_GEMV(8,  8)
DEFINE_CUBLASDX_GEMV(12, 12)
DEFINE_CUBLASDX_GEMV(14, 14)
DEFINE_CUBLASDX_GEMV(24, 24)
DEFINE_CUBLASDX_GEMV(64, 64)

#define RUN_CUBLASDX_GEMV(M, N, dA, dx, dy, iters, t0, t1)                          \
    cudaMemset(dy, 0, (size_t)M*sizeof(float));                                      \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                           \
    cublasdx_gemv_##M##x##N::kernel_global<<<1,                                      \
        cublasdx_gemv_##M##x##N::GEMM::block_dim,                                    \
        cublasdx_gemv_##M##x##N::smem_size>>>(dA, dx, dy, iters);                   \
    cudaDeviceSynchronize();                                                         \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                           \
    printf("cuBLASDx gemv (global)       m=%2d n=%2d  %.3f us/op\n",               \
           M, N, elapsed_us(t0, t1) / iters);                                        \
    cudaMemset(dy, 0, (size_t)M*sizeof(float));                                      \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                           \
    cublasdx_gemv_##M##x##N::kernel_smem<<<1,                                        \
        cublasdx_gemv_##M##x##N::GEMM::block_dim,                                    \
        cublasdx_gemv_##M##x##N::smem_size>>>(dA, dx, dy, iters);                   \
    cudaDeviceSynchronize();                                                         \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                           \
    printf("cuBLASDx gemv (shared)       m=%2d n=%2d  %.3f us/op\n",               \
           M, N, elapsed_us(t0, t1) / iters);

// ─── glass::nvidia gemv kernels (alongside raw cuBLASDx) ─────────────────────
// Two variants: default block_dim, and BlockDim<THREADS> pinned.
// Both write to a volatile sink each iteration to defeat dead-store elimination.

namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMV_BLOCKDIM(4,  4,  THREADS)
    DEFINE_NVIDIA_GEMV_BLOCKDIM(6,  6,  THREADS)
    DEFINE_NVIDIA_GEMV_BLOCKDIM(8,  8,  THREADS)
    DEFINE_NVIDIA_GEMV_BLOCKDIM(12, 12, THREADS)
    DEFINE_NVIDIA_GEMV_BLOCKDIM(14, 14, THREADS)
    DEFINE_NVIDIA_GEMV_BLOCKDIM(24, 24, THREADS)
    DEFINE_NVIDIA_GEMV_BLOCKDIM(64, 64, THREADS)
}} // namespace glass::nvidia

template<int M, int N>
__global__ void k_nv_gemv_default(float* A, float* x, float* y, volatile float* sink, int iters) {
    extern __shared__ __align__(16) char nv_smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemv<float, M, N>(1.f, A, x, 0.f, y, nv_smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = y[0];
        __syncthreads();
    }
}

template<int M, int N, int TC>
__global__ void k_nv_gemv_blockdim(float* A, float* x, float* y, volatile float* sink, int iters) {
    extern __shared__ __align__(16) char nv_smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemv<float, M, N, TC>(1.f, A, x, 0.f, y, nv_smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = y[0];
        __syncthreads();
    }
}

#define RUN_NVIDIA_GEMV(M, N, dA, dx, dy, sink, iters, t0, t1)                          \
    {                                                                                    \
        constexpr auto smem_def = glass::nvidia::gemv_scratch_bytes<float, M, N>();         \
        constexpr auto thr_def  = glass::nvidia::gemv_threads<float, M, N>();           \
        cudaMemset(dy, 0, (size_t)M*sizeof(float));                                      \
        clock_gettime(CLOCK_MONOTONIC, &(t0));                                            \
        k_nv_gemv_default<M, N><<<1, thr_def, smem_def>>>(dA, dx, dy, sink, iters);     \
        cudaDeviceSynchronize();                                                          \
        clock_gettime(CLOCK_MONOTONIC, &(t1));                                            \
        printf("glass::nvidia gemv (default) m=%2d n=%2d  %.3f us/op\n",                \
               M, N, elapsed_us(t0, t1) / iters);                                        \
        constexpr auto smem_bd = glass::nvidia::gemv_scratch_bytes<float, M, N, THREADS>(); \
        cudaMemset(dy, 0, (size_t)M*sizeof(float));                                      \
        clock_gettime(CLOCK_MONOTONIC, &(t0));                                            \
        k_nv_gemv_blockdim<M, N, THREADS><<<1, THREADS, smem_bd>>>(dA, dx, dy, sink, iters); \
        cudaDeviceSynchronize();                                                          \
        clock_gettime(CLOCK_MONOTONIC, &(t1));                                            \
        printf("glass::nvidia gemv (TC=%d)  m=%2d n=%2d  %.3f us/op\n",                 \
               THREADS, M, N, elapsed_us(t0, t1) / iters);                               \
    }

#endif // GLASS_BENCH_CUBLASDX

// ─── Benchmark runner ─────────────────────────────────────────────────────────

static void bench_size(int m, int n, int iters) {
    float *dA, *dx, *dy;
    float *dSink;
    cudaMalloc(&dA, m * n * sizeof(float));
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, m * sizeof(float));
    cudaMalloc(&dSink, 256 * sizeof(float));   // for anti-DSE writes
    (void)dSink;

    struct timespec t0, t1;
    size_t y_bytes = (size_t)m * sizeof(float);

    // glass (global)
    cudaMemset(dy, 0, y_bytes);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_gemv_global<<<1, THREADS>>>(dA, dx, dy, m, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::gemv (global) m=%2d n=%2d  %.3f us/op\n",
           m, n, elapsed_us(t0, t1) / iters);

    // glass (shared)
    int glass_smem = (m * n + n + m) * sizeof(float);
    cudaMemset(dy, 0, y_bytes);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_gemv_shared<<<1, THREADS, glass_smem>>>(dA, dx, dy, m, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::gemv (shared) m=%2d n=%2d  %.3f us/op\n",
           m, n, elapsed_us(t0, t1) / iters);

    // glass compile-time variants (only for pre-instantiated sizes)
    #define MAYBE_GLASS_GEMV_CT(M, N)                                           \
        if (m == M && n == N) { RUN_GLASS_GEMV_CT(M, N, dA, dx, dy, iters, t0, t1); }
    MAYBE_GLASS_GEMV_CT(4,  4)
    MAYBE_GLASS_GEMV_CT(6,  6)
    MAYBE_GLASS_GEMV_CT(8,  8)
    MAYBE_GLASS_GEMV_CT(12, 12)
    MAYBE_GLASS_GEMV_CT(14, 14)
    MAYBE_GLASS_GEMV_CT(24, 24)
    MAYBE_GLASS_GEMV_CT(64, 64)
    #undef MAYBE_GLASS_GEMV_CT

    // glass::thread:: tier (one problem per thread) — only for N in {4,6}
    if (m == n && m == 4) RUN_THREAD_GEMV_CT(4, iters, t0, t1);
    if (m == n && m == 6) RUN_THREAD_GEMV_CT(6, iters, t0, t1);

#ifdef GLASS_BENCH_CUBLASDX
    #define MAYBE_CUBLASDX_GEMV(M, N)                                           \
        if (m == M && n == N) { RUN_CUBLASDX_GEMV(M, N, dA, dx, dy, iters, t0, t1); }
    MAYBE_CUBLASDX_GEMV(4,  4)
    MAYBE_CUBLASDX_GEMV(6,  6)
    MAYBE_CUBLASDX_GEMV(8,  8)
    MAYBE_CUBLASDX_GEMV(12, 12)
    MAYBE_CUBLASDX_GEMV(14, 14)
    MAYBE_CUBLASDX_GEMV(24, 24)
    MAYBE_CUBLASDX_GEMV(64, 64)
    #undef MAYBE_CUBLASDX_GEMV

    // glass::nvidia variants (default block_dim + caller-pinned BlockDim<THREADS>)
    #define MAYBE_NVIDIA_GEMV(M, N)                                             \
        if (m == M && n == N) { RUN_NVIDIA_GEMV(M, N, dA, dx, dy, dSink, iters, t0, t1); }
    MAYBE_NVIDIA_GEMV(4,  4)
    MAYBE_NVIDIA_GEMV(6,  6)
    MAYBE_NVIDIA_GEMV(8,  8)
    MAYBE_NVIDIA_GEMV(12, 12)
    MAYBE_NVIDIA_GEMV(14, 14)
    MAYBE_NVIDIA_GEMV(24, 24)
    MAYBE_NVIDIA_GEMV(64, 64)
    #undef MAYBE_NVIDIA_GEMV
#endif

    cudaFree(dA); cudaFree(dx); cudaFree(dy); cudaFree(dSink);
}

int main(int argc, char** argv) {
    int m     = (argc > 1) ? atoi(argv[1]) : 0;
    int n     = (argc > 2) ? atoi(argv[2]) : 0;
    int iters = (argc > 3) ? atoi(argv[3]) : 100000;

    if (m > 0 && n > 0) {
        bench_size(m, n, iters);
    } else {
        int sizes[] = {4, 6, 8, 12, 14, 24, 64};
        for (int s : sizes) bench_size(s, s, iters);
    }

    return 0;
}
