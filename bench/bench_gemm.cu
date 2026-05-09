// bench_gemm.cu — L3 benchmarks: glass plain vs. tiled vs. cuBLASDx
// Compilation (with cuBLASDx):
//   nvcc -std=c++17 -arch=sm_XX -O3
//        -I.. -I../src
//        -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
//        -DGLASS_BENCH_CUBLASDX -DSMS=XX0
//        -Xptxas -O1
//        bench_gemm.cu -o bench_gemm
// Usage: ./bench_gemm [m [n [k [iters]]]]
//        (no args = sweep robot-dynamics sizes)

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "../glass.cuh"

#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#endif

static const int THREADS = 256;
static const int TILE    = 8;

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// ─── GLASS kernels ────────────────────────────────────────────────────────────

__global__ void k_glass_gemm_plain(float* A, float* B, float* C, int m, int n, int k, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        glass::simple::gemm<float>(m, n, k, 1.f, A, B, 0.f, C);
    }
}

__global__ void k_glass_gemm_tiled(float* A, float* B, float* C, int m, int n, int k, int iters) {
    extern __shared__ float smem[];
    float* s_A = smem;
    float* s_B = smem + m * TILE;
    for (int rep = 0; rep < iters; rep++) {
        glass::simple::gemm_tiled<float, TILE>(m, n, k, 1.f, A, B, 0.f, C, s_A, s_B);
    }
}

// ─── cuBLASDx kernels (compile-time M/N/K) ────────────────────────────────────
#ifdef GLASS_BENCH_CUBLASDX

#ifndef SMS
#define SMS 860
#endif

// cuBLASDx 25.x API: data loaded global→shared before execute(); iters loop
// benchmarks the execute() call with data already resident in shared memory.
#define DEFINE_CUBLASDX_GEMM(M, N, K)                                                       \
    namespace cublasdx_gemm_##M##x##N##x##K {                                               \
        using GEMM = decltype(                                                               \
            cublasdx::Size<M, K, N>()                                                        \
            + cublasdx::Precision<float>()                                                   \
            + cublasdx::Type<cublasdx::type::real>()                                         \
            + cublasdx::Function<cublasdx::function::MM>()                                   \
            + cublasdx::SM<SMS>()                                                             \
            + cublasdx::Block());                                                             \
        __launch_bounds__(GEMM::max_threads_per_block)                                       \
        __global__ void kernel(float* A, float* B, float* C, int iters) {                   \
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
                GEMM().execute(1.f, a_smem, b_smem, 0.f, c_smem);                           \
                __syncthreads();                                                             \
            }                                                                                \
            cublasdx::copy<GEMM, align::c>(                                                  \
                c_smem, cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()));               \
        }                                                                                    \
        static constexpr auto smem_size = cublasdx::get_shared_storage_size<GEMM>();        \
    }

// Square sizes (robot-dynamics motivated: DOF and 2*DOF)
DEFINE_CUBLASDX_GEMM(4,  4,  4)
DEFINE_CUBLASDX_GEMM(6,  6,  6)
DEFINE_CUBLASDX_GEMM(8,  8,  8)
DEFINE_CUBLASDX_GEMM(12, 12, 12)
DEFINE_CUBLASDX_GEMM(14, 14, 14)
DEFINE_CUBLASDX_GEMM(24, 24, 24)

#define RUN_CUBLASDX_GEMM(M, N, K, dA, dB, dC, iters, t0, t1)                   \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                        \
    cublasdx_gemm_##M##x##N##x##K::kernel<<<1,                                   \
        cublasdx_gemm_##M##x##N##x##K::GEMM::block_dim,                          \
        cublasdx_gemm_##M##x##N##x##K::smem_size>>>(dA, dB, dC, iters);         \
    cudaDeviceSynchronize();                                                      \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                        \
    printf("cuBLASDx gemm                m=%2d n=%2d k=%2d  %.3f us/op\n",      \
           M, N, K, elapsed_us(t0, t1) / iters);

#endif // GLASS_BENCH_CUBLASDX

// ─── Benchmark runner ─────────────────────────────────────────────────────────

static void bench_size(int m, int n, int k, int iters) {
    float *dA, *dB, *dC;
    cudaMalloc(&dA, m * n * sizeof(float));
    cudaMalloc(&dB, n * k * sizeof(float));
    cudaMalloc(&dC, m * k * sizeof(float));

    struct timespec t0, t1;

    // ── glass::simple::gemm (plain) ──────────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_gemm_plain<<<1, THREADS>>>(dA, dB, dC, m, n, k, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::simple::gemm (plain)  m=%2d n=%2d k=%2d  %.3f us/op\n",
           m, n, k, elapsed_us(t0, t1) / iters);

    // ── glass::simple::gemm_tiled (only when m*k <= THREADS) ────────────────
    if (m * k <= THREADS) {
        int smem_bytes = (m * TILE + TILE * k) * sizeof(float);
        clock_gettime(CLOCK_MONOTONIC, &t0);
        k_glass_gemm_tiled<<<1, THREADS, smem_bytes>>>(dA, dB, dC, m, n, k, iters);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        printf("glass::simple::gemm (tiled)  m=%2d n=%2d k=%2d  %.3f us/op\n",
               m, n, k, elapsed_us(t0, t1) / iters);
    }

    // ── cuBLASDx (only for pre-compiled sizes) ───────────────────────────────
#ifdef GLASS_BENCH_CUBLASDX
    #define MAYBE_CUBLASDX_GEMM(M, N, K)                                         \
        if (m == M && n == N && k == K) { RUN_CUBLASDX_GEMM(M, N, K, dA, dB, dC, iters, t0, t1); }
    MAYBE_CUBLASDX_GEMM(4,  4,  4)
    MAYBE_CUBLASDX_GEMM(6,  6,  6)
    MAYBE_CUBLASDX_GEMM(8,  8,  8)
    MAYBE_CUBLASDX_GEMM(12, 12, 12)
    MAYBE_CUBLASDX_GEMM(14, 14, 14)
    MAYBE_CUBLASDX_GEMM(24, 24, 24)
    #undef MAYBE_CUBLASDX_GEMM
#endif

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv) {
    int m     = (argc > 1) ? atoi(argv[1]) : 0;
    int n     = (argc > 2) ? atoi(argv[2]) : 0;
    int k     = (argc > 3) ? atoi(argv[3]) : 0;
    int iters = (argc > 4) ? atoi(argv[4]) : 10000;

    if (m > 0 && n > 0 && k > 0) {
        bench_size(m, n, k, iters);
    } else {
        // Sweep robot-dynamics sizes: DOF and 2*DOF for common robots
        int sizes[] = {4, 6, 8, 12, 14, 24};
        for (int s : sizes) bench_size(s, s, s, iters);
    }

    return 0;
}
