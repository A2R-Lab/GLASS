// bench_gemm_batched_1d.cu — 1D-launch batched GEMM: SIMT vs cuBLASDx.
//
// For each (M, N, K, BATCH), times:
//   simt   — ::glass::gemm_batched_1d (flat 1D parallelism across BATCH*M*K elements)
//   cublas — glass::nvidia::gemm_batched_1d (cuBLASDx, sequential outer loop)
//
// Output labels are formatted so bench/autotune.py can parse them and decide
// which path to write into the local tuning override.

#include <cstdio>
#include <cstdlib>
#include <ctime>

#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#endif
#include "../glass-nvidia.cuh"

static const int THREADS = 64;
static const int M = 6, N = 6, K = 6;

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// Pre-instantiate cuBLASDx specializations for the 1D batched variant at the
// shapes we will benchmark.
namespace glass { namespace nvidia {
    DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM(M, N, K, 4,  THREADS)
    DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM(M, N, K, 8,  THREADS)
    DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM(M, N, K, 16, THREADS)
    DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM(M, N, K, 32, THREADS)
}}

template<int BATCH>
__global__ void k_simt_1d(float** A, float** B, float** C,
                          volatile float* sink, int iters) {
    for (int rep = 0; rep < iters; rep++) {
        ::glass::gemm_batched_1d<float, M, N, K, BATCH>(1.f, A, B, 0.f, C);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C[0][0];
        __syncthreads();
    }
}

template<int BATCH>
__global__ void k_cublas_1d(float** A, float** B, float** C,
                            volatile float* sink, int iters) {
    extern __shared__ __align__(16) char smem[];
    for (int rep = 0; rep < iters; rep++) {
        glass::nvidia::gemm_batched_1d<float, M, N, K, BATCH, THREADS>(
            1.f, A, B, 0.f, C, smem);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = C[0][0];
        __syncthreads();
    }
}

template<int BATCH>
static void bench_batch(int iters) {
    float *dA_buf, *dB_buf, *dC_buf, *dSink;
    cudaMalloc(&dA_buf, BATCH * M*N * sizeof(float));
    cudaMalloc(&dB_buf, BATCH * N*K * sizeof(float));
    cudaMalloc(&dC_buf, BATCH * M*K * sizeof(float));
    cudaMalloc(&dSink,  256 * sizeof(float));

    float* hA[BATCH]; float* hB[BATCH]; float* hC[BATCH];
    for (int b = 0; b < BATCH; b++) {
        hA[b] = dA_buf + b * M * N;
        hB[b] = dB_buf + b * N * K;
        hC[b] = dC_buf + b * M * K;
    }
    float **dA, **dB, **dC;
    cudaMalloc(&dA, BATCH * sizeof(float*));
    cudaMalloc(&dB, BATCH * sizeof(float*));
    cudaMalloc(&dC, BATCH * sizeof(float*));
    cudaMemcpy(dA, hA, BATCH * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, BATCH * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, BATCH * sizeof(float*), cudaMemcpyHostToDevice);

    struct timespec t0, t1;

    // SIMT 1D batched
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_simt_1d<BATCH><<<1, THREADS>>>(dA, dB, dC, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    // Label format autotune.py recognizes (see classify() in autotune.py):
    //   "naive loop ... BATCH=N m= M n= N k= K   ... us/op"
    printf("naive loop  gemm_batched_1d  BATCH=%-2d m=%2d n=%2d k=%2d  %.3f us/op\n",
           BATCH, M, N, K, elapsed_us(t0, t1) / iters);

    // cuBLASDx 1D batched
    constexpr size_t smemsz = glass::nvidia::gemm_batched_1d_smem_size<float, M, N, K, BATCH, THREADS>();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_cublas_1d<BATCH><<<1, THREADS, smemsz>>>(dA, dB, dC, dSink, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nvidia::gemm_batched_1d  BATCH=%-2d m=%2d n=%2d k=%2d  %.3f us/op\n",
           BATCH, M, N, K, elapsed_us(t0, t1) / iters);

    cudaFree(dA_buf); cudaFree(dB_buf); cudaFree(dC_buf);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dSink);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 10000;
    bench_batch<4>(iters);
    bench_batch<8>(iters);
    bench_batch<16>(iters);
    bench_batch<32>(iters);
    return 0;
}
