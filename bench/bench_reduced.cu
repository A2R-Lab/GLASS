// bench_reduced.cu — A/B crossover sweep: serial glass::gemm vs the
// contraction-parallel glass::gemm_reduced, with A/B/C resident in shared
// memory (the DDP-backward-pass regime). Reduced parallelizes the length-N
// contraction across a warp's lanes; total MAC work is identical, so it only
// wins when the output count n_out = M*Ccols is smaller than the block (idle
// threads to soak up) AND N amortizes the ~5-step shuffle tail.
//
// Emits one parseable row per (dtype,M,N,K,blockDim):
//     REDUCED dtype M N K blockDim n_out serial_us reduced_us ratio spreads
// where ratio = serial_us / reduced_us  (>1 ⇒ reduced wins). Used to seed
// the conservative standard policy and the contraction_parallel concepts page.
//
// Build: nvcc -std=c++17 -arch=sm_XX -O3 -I.. -I../src bench_reduced.cu -o bench_reduced
// Usage: ./bench_reduced [iters] [f32|f64|both]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../glass.cuh"
#include "timing_common.cuh"

static double g_spread = 0.0;

// Shapes: square sizes, wide-contraction shapes (N≫n_out, where reduced should
// win), and the DDP consumer dims (14/7/21). Ccols = K (no transpose).
#define SHAPES(_) \
    _(4,4,4)   _(8,8,8)   _(14,14,14) _(21,21,21) \
    _(4,4,64)  _(8,8,64)  _(2,64,2)   _(7,7,64) \
    _(14,14,7) _(14,21,14) _(21,14,21) _(7,14,7)

template <typename T, uint32_t M, uint32_t N, uint32_t K, bool REDUCED>
__global__ void k_bench(const T* A, const T* B, T* C, int iters) {
    extern __shared__ double raw[];
    T* s = reinterpret_cast<T*>(raw);
    T* sA = s; T* sB = s + M*N; T* sC = s + M*N + N*K;
    for (uint32_t i = threadIdx.x; i < M*N; i += blockDim.x) sA[i] = A[i];
    for (uint32_t i = threadIdx.x; i < N*K; i += blockDim.x) sB[i] = B[i];
    for (uint32_t i = threadIdx.x; i < M*K; i += blockDim.x) sC[i] = C[i];
    __syncthreads();
    for (int rep = 0; rep < iters; ++rep) {
        if (REDUCED)
            // TRAILING_SYNC=false: each output is owned by the same warp every
            // iteration, so back-to-back reps have no cross-warp hazard on sC —
            // matches the barrier-free serial gemm for a fair compute compare.
            glass::block::gemm_reduced<T, M, N, K, false, false, false>((T)1, sA, sB, (T)1, sC);
        else
            glass::block::gemm<T, M, N, K>((T)1, sA, sB, (T)1, sC);
    }
    __syncthreads();
    if (threadIdx.x == 0) C[0] = sC[0];   // anti-DCE
}

template <typename T, uint32_t M, uint32_t N, uint32_t K>
static double time_one(bool reduced, int blockDim, const T* dA, const T* dB,
                       T* dC, int iters) {
    const int smem = (M*N + N*K + M*K) * sizeof(T);
    // warmup
    if (reduced) k_bench<T,M,N,K,true ><<<1, blockDim, smem>>>(dA, dB, dC, 256);
    else         k_bench<T,M,N,K,false><<<1, blockDim, smem>>>(dA, dB, dC, 256);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 1e30;
    double best = 1e30, worst = 0.0;
    for (int trial = 0; trial < 3; ++trial) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (reduced) k_bench<T,M,N,K,true ><<<1, blockDim, smem>>>(dA, dB, dC, iters);
        else         k_bench<T,M,N,K,false><<<1, blockDim, smem>>>(dA, dB, dC, iters);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = tc_elapsed_ms(t0, t1) * 1000.0 / iters;
        if (us < best) best = us;
        if (us > worst) worst = us;
    }
    g_spread = (best < 1e29) ? (worst / best - 1.0) * 100.0 : 0.0;
    return best;
}

template <typename T, uint32_t M, uint32_t N, uint32_t K>
static void bench_shape(const char* dtype, int iters) {
    T *dA, *dB, *dC;
    std::vector<T> A(M*N), B(N*K), C(M*K);
    for (size_t i=0;i<A.size();++i) A[i]=(T)(0.01*(1+(i*7)%13));
    for (size_t i=0;i<B.size();++i) B[i]=(T)(0.01*(1+(i*5)%11));
    for (size_t i=0;i<C.size();++i) C[i]=(T)(0.001*(1+i%3));
    cudaMalloc(&dA, A.size()*sizeof(T)); cudaMalloc(&dB, B.size()*sizeof(T));
    cudaMalloc(&dC, C.size()*sizeof(T));
    cudaMemcpy(dA,A.data(),A.size()*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B.data(),B.size()*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(dC,C.data(),C.size()*sizeof(T),cudaMemcpyHostToDevice);
    const int blockdims[] = {32, 64, 128, 256};
    const uint32_t n_out = M * K;
    for (int bd : blockdims) {
        double s = time_one<T,M,N,K>(false, bd, dA, dB, dC, iters);
        double ss = g_spread;
        double r = time_one<T,M,N,K>(true,  bd, dA, dB, dC, iters);
        double rs = g_spread;
        printf("REDUCED %s %3u %3u %3u %4d %5u %8.4f %8.4f %6.3f spread=%.2f%%/%.2f%%\n",
               dtype, M, N, K, bd, n_out, s, r, s / r, ss, rs);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 200000;
    const char* dtype = (argc > 2) ? argv[2] : "both";
    printf("# REDUCED dtype M N K blockDim n_out serial_us reduced_us ratio(serial/reduced) spreads\n");
    tc_warm_gpu();
    if (!strcmp(dtype,"f64") || !strcmp(dtype,"both")) {
        #define RUN(M,N,K) bench_shape<double,M,N,K>("f64",iters);
        SHAPES(RUN)
        #undef RUN
    }
    if (!strcmp(dtype,"f32") || !strcmp(dtype,"both")) {
        #define RUN(M,N,K) bench_shape<float,M,N,K>("f32",iters);
        SHAPES(RUN)
        #undef RUN
    }
    return 0;
}
