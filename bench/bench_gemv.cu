// bench_gemv.cu — L2 benchmarks: glass::simple::gemv vs. cuBLASDx
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
#include "../glass.cuh"

#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#endif

static const int THREADS = 256;

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// ─── GLASS gemv kernel ────────────────────────────────────────────────────────
__global__ void k_glass_gemv(float* A, float* x, float* y, int m, int n, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        glass::simple::gemv<float>(m, n, 1.f, A, x, 0.f, y);
    }
}

// ─── cuBLASDx gemv kernels (one per size, compile-time M/N) ──────────────────
#ifdef GLASS_BENCH_CUBLASDX

#ifndef SMS
#define SMS 860  // default: sm_86 (Ampere A100 / RTX 30xx)
#endif

// cuBLASDx 25.x API: data loaded global→shared before execute(); iters loop
// benchmarks the execute() call with data already resident in shared memory.
#define DEFINE_CUBLASDX_GEMV(M, N)                                                          \
    namespace cublasdx_gemv_##M##x##N {                                                     \
        using GEMM = decltype(                                                              \
            cublasdx::Size<M, 1, N>()                                                       \
            + cublasdx::Precision<float>()                                                  \
            + cublasdx::Type<cublasdx::type::real>()                                        \
            + cublasdx::Function<cublasdx::function::MM>()                                  \
            + cublasdx::SM<SMS>()                                                            \
            + cublasdx::Block());                                                            \
        __launch_bounds__(GEMM::max_threads_per_block)                                      \
        __global__ void kernel(float* A, float* x, float* y, int iters) {                  \
            extern __shared__ __align__(16) char smem[];                                    \
            using align = cublasdx::alignment_of<GEMM>;                                    \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);     \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());        \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());        \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());        \
            cublasdx::copy<GEMM, align::a>(                                                 \
                cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);              \
            cublasdx::copy<GEMM, align::b>(                                                 \
                cublasdx::make_tensor(x, GEMM::get_layout_gmem_b()), b_smem);              \
            cublasdx::copy_wait();                                                          \
            for (int rep = 0; rep < iters; rep++) {                                         \
                GEMM().execute(1.f, a_smem, b_smem, 0.f, c_smem);                          \
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

#define RUN_CUBLASDX_GEMV(M, N, dA, dx, dy, iters, t0, t1)                      \
    clock_gettime(CLOCK_MONOTONIC, &(t0));                                       \
    cublasdx_gemv_##M##x##N::kernel<<<1,                                         \
        cublasdx_gemv_##M##x##N::GEMM::block_dim,                                \
        cublasdx_gemv_##M##x##N::smem_size>>>(dA, dx, dy, iters);               \
    cudaDeviceSynchronize();                                                     \
    clock_gettime(CLOCK_MONOTONIC, &(t1));                                       \
    printf("cuBLASDx gemv                m=%2d n=%2d  %.3f us/op\n",            \
           M, N, elapsed_us(t0, t1) / iters);

#endif // GLASS_BENCH_CUBLASDX

// ─── Benchmark runner ─────────────────────────────────────────────────────────

static void bench_size(int m, int n, int iters) {
    float *dA, *dx, *dy;
    cudaMalloc(&dA, m * n * sizeof(float));
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, m * sizeof(float));

    struct timespec t0, t1;

    // GLASS
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_gemv<<<1, THREADS>>>(dA, dx, dy, m, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::simple::gemv          m=%2d n=%2d  %.3f us/op\n",
           m, n, elapsed_us(t0, t1) / iters);

#ifdef GLASS_BENCH_CUBLASDX
    // cuBLASDx — dispatch by size (only benchmarked sizes are defined above)
    #define MAYBE_CUBLASDX_GEMV(M, N)                                       \
        if (m == M && n == N) { RUN_CUBLASDX_GEMV(M, N, dA, dx, dy, iters, t0, t1); }
    MAYBE_CUBLASDX_GEMV(4,  4)
    MAYBE_CUBLASDX_GEMV(6,  6)
    MAYBE_CUBLASDX_GEMV(8,  8)
    MAYBE_CUBLASDX_GEMV(12, 12)
    MAYBE_CUBLASDX_GEMV(14, 14)
    MAYBE_CUBLASDX_GEMV(24, 24)
    #undef MAYBE_CUBLASDX_GEMV
#endif

    cudaFree(dA); cudaFree(dx); cudaFree(dy);
}

int main(int argc, char** argv) {
    int m     = (argc > 1) ? atoi(argv[1]) : 0;
    int n     = (argc > 2) ? atoi(argv[2]) : 0;
    int iters = (argc > 3) ? atoi(argv[3]) : 10000;

    if (m > 0 && n > 0) {
        bench_size(m, n, iters);
    } else {
        // Sweep robot-dynamics-motivated sizes
        int sizes[] = {4, 6, 8, 12, 14, 24};
        for (int s : sizes) bench_size(s, s, iters);
    }

    return 0;
}
