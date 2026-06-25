// bench_reduced.cu — A/B crossover sweep: serial glass::gemm vs the
// contraction-parallel glass::gemm_reduced, with A/B/C resident in shared
// memory (the DDP-backward-pass regime). Reduced parallelizes the length-N
// contraction across a warp's lanes; total MAC work is identical, so it only
// wins when the output count n_out = M*Ccols is smaller than the block (idle
// threads to soak up) AND N amortizes the ~5-step shuffle tail.
//
// Emits one parseable row per (M,N,K,blockDim):
//     REDUCED  M N K  blockDim  n_out  serial_us  reduced_us  ratio
// where ratio = serial_us / reduced_us  (>1 ⇒ reduced wins). Used to seed
// glass::suggested_use_reduced<>() and the contraction_parallel concepts page.
//
// Build: nvcc -std=c++17 -arch=sm_XX -O3 -I.. -I../src bench_reduced.cu -o bench_reduced
// Usage: ./bench_reduced [iters]

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "../glass.cuh"

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6 + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

// Shapes: square sizes, wide-contraction shapes (N≫n_out, where reduced should
// win), and the DDP consumer dims (14/7/21). Ccols = K (no transpose).
#define SHAPES(_) \
    _(4,4,4)   _(8,8,8)   _(14,14,14) _(21,21,21) \
    _(4,4,64)  _(8,8,64)  _(2,64,2)   _(7,7,64) \
    _(14,14,7) _(14,21,14) _(21,14,21) _(7,14,7)

template <uint32_t M, uint32_t N, uint32_t K, bool REDUCED>
__global__ void k_bench(const float* A, const float* B, float* C, int iters) {
    extern __shared__ float s[];
    float* sA = s; float* sB = s + M*N; float* sC = s + M*N + N*K;
    for (uint32_t i = threadIdx.x; i < M*N; i += blockDim.x) sA[i] = A[i];
    for (uint32_t i = threadIdx.x; i < N*K; i += blockDim.x) sB[i] = B[i];
    for (uint32_t i = threadIdx.x; i < M*K; i += blockDim.x) sC[i] = C[i];
    __syncthreads();
    for (int rep = 0; rep < iters; ++rep) {
        if (REDUCED)
            // TRAILING_SYNC=false: each output is owned by the same warp every
            // iteration, so back-to-back reps have no cross-warp hazard on sC —
            // matches the barrier-free serial gemm for a fair compute compare.
            glass::gemm_reduced<float, M, N, K, false, false, false>(1.f, sA, sB, 1.f, sC);
        else
            glass::gemm<float, M, N, K>(1.f, sA, sB, 1.f, sC);
    }
    __syncthreads();
    if (threadIdx.x == 0) C[0] = sC[0];   // anti-DCE
}

template <uint32_t M, uint32_t N, uint32_t K>
static double time_one(bool reduced, int blockDim, const float* dA, const float* dB,
                       float* dC, int iters) {
    const int smem = (M*N + N*K + M*K) * sizeof(float);
    struct timespec t0, t1;
    // warmup
    if (reduced) k_bench<M,N,K,true ><<<1, blockDim, smem>>>(dA, dB, dC, 256);
    else         k_bench<M,N,K,false><<<1, blockDim, smem>>>(dA, dB, dC, 256);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (reduced) k_bench<M,N,K,true ><<<1, blockDim, smem>>>(dA, dB, dC, iters);
    else         k_bench<M,N,K,false><<<1, blockDim, smem>>>(dA, dB, dC, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return elapsed_us(t0, t1) / iters;
}

template <uint32_t M, uint32_t N, uint32_t K>
static void bench_shape(int iters) {
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*N*sizeof(float));
    cudaMalloc(&dB, N*K*sizeof(float));
    cudaMalloc(&dC, M*K*sizeof(float));
    cudaMemset(dA, 0, M*N*sizeof(float));
    cudaMemset(dB, 0, N*K*sizeof(float));
    cudaMemset(dC, 0, M*K*sizeof(float));
    const int blockdims[] = {32, 64, 128, 256};
    const uint32_t n_out = M * K;
    for (int bd : blockdims) {
        double s = time_one<M,N,K>(false, bd, dA, dB, dC, iters);
        double r = time_one<M,N,K>(true,  bd, dA, dB, dC, iters);
        printf("REDUCED %3u %3u %3u  %4d  %5u  %8.4f  %8.4f  %6.3f\n",
               M, N, K, bd, n_out, s, r, s / r);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200000;
    printf("# REDUCED  M N K  blockDim  n_out  serial_us  reduced_us  ratio(serial/reduced)\n");
    #define RUN(M,N,K) bench_shape<M,N,K>(iters);
    SHAPES(RUN)
    #undef RUN
    return 0;
}
