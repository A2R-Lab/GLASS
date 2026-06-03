// bench_blockdim.cu — demonstrates the P0-1 BlockDim deadlock fix is functional
// and quantifies the cost of running with caller-pinned BlockDim<TC> vs.
// cuBLASDx's natural block_dim choice.
//
// For each (M, N, K), three measurements:
//   default    — glass::nvidia::gemm<...>             launched with cuBLASDx's
//                                                     natural block_dim
//   pinned-128 — glass::nvidia::gemm<...,128>(...)    launched <<<1, 128>>>
//   pinned-352 — glass::nvidia::gemm<...,352>(...)    launched <<<1, 352>>>
//                (matches GRiD's iiwa14 MAX_PERF_LEVEL_THREADS, the value that
//                 deadlocked before P0-1)
//
// Anti-optimization: per-iter sink writes; -Xptxas -O1 (set by run_bench.py);
// --expt-relaxed-constexpr.
//
// Usage: ./bench_blockdim [iters]

#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#endif
#include "../glass-nvidia.cuh"

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// Pre-instantiate at the standard sizes for both pinned thread counts.
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM_BLOCKDIM(4,  4,  4,  128)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(6,  6,  6,  128)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(8,  8,  8,  128)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(12, 12, 12, 128)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(14, 14, 14, 128)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(24, 24, 24, 128)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(64, 64, 64, 128)

    DEFINE_NVIDIA_GEMM_BLOCKDIM(4,  4,  4,  352)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(6,  6,  6,  352)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(8,  8,  8,  352)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(12, 12, 12, 352)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(14, 14, 14, 352)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(24, 24, 24, 352)
    DEFINE_NVIDIA_GEMM_BLOCKDIM(64, 64, 64, 352)
}}

template<int M, int N, int K>
__global__ void k_default(float* A, float* B, float* C, volatile float* sink, int iters) {
    extern __shared__ __align__(16) char smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm<float, M, N, K>(1.f, A, B, 0.f, C, smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C[0];
        __syncthreads();
    }
}

template<int M, int N, int K, int TC>
__global__ void k_pinned(float* A, float* B, float* C, volatile float* sink, int iters) {
    extern __shared__ __align__(16) char smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm<float, M, N, K, TC>(1.f, A, B, 0.f, C, smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C[0];
        __syncthreads();
    }
}

template<int M, int N, int K>
static void bench_one(int iters) {
    float *dA, *dB, *dC, *dSink;
    cudaMalloc(&dA, M*N*sizeof(float));
    cudaMalloc(&dB, N*K*sizeof(float));
    cudaMalloc(&dC, M*K*sizeof(float));
    cudaMalloc(&dSink, 256*sizeof(float));

    struct timespec t0, t1;

    // default (cuBLASDx-chosen block_dim)
    constexpr auto smem_def = glass::nvidia::gemm_smem_size<float, M, N, K>();
    constexpr auto thr_def  = glass::nvidia::gemm_threads<float, M, N, K>();
    cudaMemset(dC, 0, M*K*sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_default<M, N, K><<<1, thr_def, smem_def>>>(dA, dB, dC, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("M=%2d N=%2d K=%2d  default(TC=%-3u)  %.3f us/op\n",
           M, N, K, thr_def, elapsed_us(t0, t1) / iters);

    // pinned 128
    constexpr auto smem_128 = glass::nvidia::gemm_smem_size<float, M, N, K, 128>();
    cudaMemset(dC, 0, M*K*sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_pinned<M, N, K, 128><<<1, 128, smem_128>>>(dA, dB, dC, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("M=%2d N=%2d K=%2d  pinned(TC=128)   %.3f us/op\n",
           M, N, K, elapsed_us(t0, t1) / iters);

    // pinned 352 (GRiD iiwa14 MAX_PERF_LEVEL_THREADS — the value that deadlocked pre-P0-1)
    constexpr auto smem_352 = glass::nvidia::gemm_smem_size<float, M, N, K, 352>();
    cudaMemset(dC, 0, M*K*sizeof(float));
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_pinned<M, N, K, 352><<<1, 352, smem_352>>>(dA, dB, dC, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("M=%2d N=%2d K=%2d  pinned(TC=352)   %.3f us/op  (GRiD iiwa14 launch)\n",
           M, N, K, elapsed_us(t0, t1) / iters);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dSink);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 10000;
    bench_one<4,  4,  4 >(iters);
    bench_one<6,  6,  6 >(iters);
    bench_one<8,  8,  8 >(iters);
    bench_one<12, 12, 12>(iters);
    bench_one<14, 14, 14>(iters);
    bench_one<24, 24, 24>(iters);
    bench_one<64, 64, 64>(iters);
    return 0;
}
