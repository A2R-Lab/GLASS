// bench_gemm_batched.cu — batched GEMM: glass::nvidia::gemm_batched vs. naive loop.
//
// Demonstrates the P2-7 single-block batching speedup. For each (M, N, K, BATCH):
//   naive:    BATCH × glass::nvidia::gemm<...>  in a single block, BlockDim<TC>
//   batched:  one  glass::nvidia::gemm_batched<...,BATCH,TC> call, dim3(TC, BATCH) launch
//
// Anti-optimization: per-iteration sink writes; -Xptxas -O1 (set in run_bench.py);
// --expt-relaxed-constexpr.
//
// Compilation:
//   nvcc -std=c++17 -arch=sm_XX -O3 --expt-relaxed-constexpr -Xptxas -O1
//        -I.. -I../src
//        -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
//        -DGLASS_BENCH_CUBLASDX -DSMS=XX0
//        bench_gemm_batched.cu -o bench_gemm_batched

#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#endif
#include "../glass-nvidia.cuh"

static const int THREADS = 64;   // per-batch thread count
static const int M = 6, N = 6, K = 6;

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// Pre-instantiate the basic GEMM at TC=THREADS (used by the naive loop) and
// the batched variants at the BATCH counts we will benchmark.
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM_BLOCKDIM(M, N, K, THREADS)
    DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(M, N, K, 4,  THREADS)
    DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(M, N, K, 8,  THREADS)
    DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(M, N, K, 16, THREADS)
    DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(M, N, K, 32, THREADS)
}}

// Naive: BATCH sequential gemm calls in one block (one batch active at a time).
template<int BATCH>
__global__ void k_naive_loop(float** A_ptrs, float** B_ptrs, float** C_ptrs,
                              volatile float* sink, int iters) {
    extern __shared__ __align__(16) char smem[];
    for (int rep = 0; rep < iters; rep++) {
        for (int b = 0; b < BATCH; b++) {
            glass::nvidia::gemm<float, M, N, K, THREADS>(
                1.f, A_ptrs[b], B_ptrs[b], 0.f, C_ptrs[b], smem);
            __syncthreads();
        }
        if (threadIdx.x == 0) sink[rep & 0xFF] = C_ptrs[0][0];
        __syncthreads();
    }
}

// Batched: one call processes BATCH items in parallel via threadIdx.y.
template<int BATCH>
__global__ void k_batched(float** A_ptrs, float** B_ptrs, float** C_ptrs,
                           volatile float* sink, int iters) {
    extern __shared__ __align__(16) char smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm_batched<float, M, N, K, BATCH, THREADS>(
            1.f, A_ptrs, B_ptrs, 0.f, C_ptrs, smem);
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) sink[rep & 0xFF] = C_ptrs[0][0];
        __syncthreads();
    }
}

template<int BATCH>
static void bench_batch(int iters) {
    // Allocate BATCH separate (M*N) A's, (N*K) B's, (M*K) C's contiguously.
    float *dA_buf, *dB_buf, *dC_buf, *dSink;
    cudaMalloc(&dA_buf, BATCH * M*N * sizeof(float));
    cudaMalloc(&dB_buf, BATCH * N*K * sizeof(float));
    cudaMalloc(&dC_buf, BATCH * M*K * sizeof(float));
    cudaMalloc(&dSink,  256 * sizeof(float));

    // Build pointer arrays on host then copy to device.
    float* hA_ptrs[BATCH];
    float* hB_ptrs[BATCH];
    float* hC_ptrs[BATCH];
    for (int b = 0; b < BATCH; b++) {
        hA_ptrs[b] = dA_buf + b * M * N;
        hB_ptrs[b] = dB_buf + b * N * K;
        hC_ptrs[b] = dC_buf + b * M * K;
    }
    float **dA_ptrs, **dB_ptrs, **dC_ptrs;
    cudaMalloc(&dA_ptrs, BATCH * sizeof(float*));
    cudaMalloc(&dB_ptrs, BATCH * sizeof(float*));
    cudaMalloc(&dC_ptrs, BATCH * sizeof(float*));
    cudaMemcpy(dA_ptrs, hA_ptrs, BATCH * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dB_ptrs, hB_ptrs, BATCH * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dC_ptrs, hC_ptrs, BATCH * sizeof(float*), cudaMemcpyHostToDevice);

    struct timespec t0, t1;

    // Naive loop (one block, sequential per batch).
    constexpr size_t naive_smem = glass::nvidia::gemm_smem_size<float, M, N, K, THREADS>();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_naive_loop<BATCH><<<1, THREADS, naive_smem>>>(dA_ptrs, dB_ptrs, dC_ptrs, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia naive-loop  M=%d N=%d K=%d BATCH=%-2d  %.3f us/op (= %.3f us/batch)\n",
           M, N, K, BATCH, elapsed_us(t0, t1) / iters,
           elapsed_us(t0, t1) / iters / BATCH);

    // Batched (one block, dim3(TC, BATCH)).
    constexpr size_t batch_smem = glass::nvidia::gemm_batched_smem_size<float, M, N, K, BATCH, THREADS>();
    dim3 batch_block(THREADS, BATCH);
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_batched<BATCH><<<1, batch_block, batch_smem>>>(dA_ptrs, dB_ptrs, dC_ptrs, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia gemm_batched M=%d N=%d K=%d BATCH=%-2d  %.3f us/op (= %.3f us/batch)\n",
           M, N, K, BATCH, elapsed_us(t0, t1) / iters,
           elapsed_us(t0, t1) / iters / BATCH);

    cudaFree(dA_buf); cudaFree(dB_buf); cudaFree(dC_buf);
    cudaFree(dA_ptrs); cudaFree(dB_ptrs); cudaFree(dC_ptrs);
    cudaFree(dSink);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 10000;
    bench_batch<4>(iters);
    bench_batch<8>(iters);
    bench_batch<16>(iters);
    bench_batch<32>(iters);
    return 0;
}
