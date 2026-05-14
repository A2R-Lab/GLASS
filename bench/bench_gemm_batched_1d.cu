// bench_gemm_batched_1d.cu — 1D-launch SIMT batched GEMM (P0-1, P0-2).
//
// Three kernels per BATCH:
//   naive_loop_1d:  BATCH × glass::gemm<T,M,N,K> in a single 1D block.
//   batched_1d:     one glass::nvidia::gemm_batched_1d<...> call.
//   strided_1d:     one glass::nvidia::gemm_strided_batched_1d<...> call
//                   (shared A; strided B/C).
//
// Launch geometry: <<<grid, dim3(TC*BATCH, 1, 1)>>> (no smem needed).
// Anti-DSE: per-iteration sink writes; -Xptxas -O1 (set in run_bench.py).
//
// Compilation:
//   nvcc -std=c++17 -arch=sm_XX -O3 -I.. -I../src
//        bench_gemm_batched_1d.cu -o bench_gemm_batched_1d
// (cuBLASDx is NOT required — l3_simt.cuh has no cuBLASDx dependency.)

#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#endif
#include "../glass-nvidia.cuh"

static const int TC = 32;            // threads per batch element
static const int M = 4, N = 4, K = 4;

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// Naive: BATCH sequential SIMT gemm calls in one block.
template<int BATCH>
__global__ void k_naive_loop_1d(float** A_ptrs, float** B_ptrs, float** C_ptrs,
                                volatile float* sink, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        for (int b = 0; b < BATCH; b++) {
            glass::gemm<float, M, N, K>(1.f, A_ptrs[b], B_ptrs[b], 0.f, C_ptrs[b]);
            __syncthreads();
        }
        if (threadIdx.x == 0) sink[rep & 0xFF] = C_ptrs[0][0];
        __syncthreads();
    }
}

// 1D-launch batched: each batch element gets TC threads.
template<int BATCH>
__global__ void k_batched_1d(float** A_ptrs, float** B_ptrs, float** C_ptrs,
                              volatile float* sink, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm_batched_1d<float, M, N, K, BATCH, TC>(
            1.f, A_ptrs, B_ptrs, 0.f, C_ptrs);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C_ptrs[0][0];
        __syncthreads();
    }
}

// Strided batched: shared A pointer, strided B and C base pointers.
template<int BATCH>
__global__ void k_strided_1d(float* A_shared, float* B_base, float* C_base,
                              volatile float* sink, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm_strided_batched_1d<float, M, N, K, BATCH, TC>(
            1.f, A_shared, B_base, 0.f, C_base);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C_base[0];
        __syncthreads();
    }
}

template<int BATCH>
static void bench_batch(int iters) {
    // Allocate BATCH × (M*N) A's, (N*K) B's, (M*K) C's contiguously, plus a
    // single shared A buffer for the strided variant.
    float *dA_buf, *dB_buf, *dC_buf, *dA_shared, *dSink;
    cudaMalloc(&dA_buf,    BATCH * M*N * sizeof(float));
    cudaMalloc(&dB_buf,    BATCH * N*K * sizeof(float));
    cudaMalloc(&dC_buf,    BATCH * M*K * sizeof(float));
    cudaMalloc(&dA_shared, M*N * sizeof(float));
    cudaMalloc(&dSink,     256 * sizeof(float));

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
    const int TOTAL_THREADS = TC * BATCH;
    const int WARMUP = 100;

    // Warmup all three kernels — first launch on a fresh CUDA context dominates
    // BATCH=1 timings otherwise (~100us cold-start vs ~1us steady-state).
    k_naive_loop_1d<BATCH><<<1, TOTAL_THREADS>>>(dA_ptrs, dB_ptrs, dC_ptrs, dSink, WARMUP);
    k_batched_1d   <BATCH><<<1, TOTAL_THREADS>>>(dA_ptrs, dB_ptrs, dC_ptrs, dSink, WARMUP);
    k_strided_1d   <BATCH><<<1, TOTAL_THREADS>>>(dA_shared, dB_buf, dC_buf, dSink, WARMUP);
    cudaDeviceSynchronize();

    // Naive loop (one 1D block, sequential per batch).
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_naive_loop_1d<BATCH><<<1, TOTAL_THREADS>>>(dA_ptrs, dB_ptrs, dC_ptrs, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia naive_loop_1d         M=%d N=%d K=%d BATCH=%-2d  %.3f us/op\n",
           M, N, K, BATCH, elapsed_us(t0, t1) / iters);

    // 1D-launch batched.
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_batched_1d<BATCH><<<1, TOTAL_THREADS>>>(dA_ptrs, dB_ptrs, dC_ptrs, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia gemm_batched_1d       M=%d N=%d K=%d BATCH=%-2d  %.3f us/op\n",
           M, N, K, BATCH, elapsed_us(t0, t1) / iters);

    // 1D-launch strided batched (shared A).
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_strided_1d<BATCH><<<1, TOTAL_THREADS>>>(dA_shared, dB_buf, dC_buf, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia gemm_strided_batch_1d M=%d N=%d K=%d BATCH=%-2d  %.3f us/op\n",
           M, N, K, BATCH, elapsed_us(t0, t1) / iters);

    cudaFree(dA_buf); cudaFree(dB_buf); cudaFree(dC_buf); cudaFree(dA_shared);
    cudaFree(dA_ptrs); cudaFree(dB_ptrs); cudaFree(dC_ptrs);
    cudaFree(dSink);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 10000;
    bench_batch<1>(iters);
    bench_batch<2>(iters);
    bench_batch<4>(iters);
    bench_batch<8>(iters);
    return 0;
}
