// bench_reduce.cu — L1 benchmarks: glass:: reduce/dot/nrm2 vs. CUB BlockReduce
// Compilation: nvcc -std=c++17 -arch=sm_XX -I.. -I../src -O3 bench_reduce.cu -o bench_reduce
// Usage: ./bench_reduce <n> [iters]

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cub/cub.cuh>
#include "../glass.cuh"
#include "../glass-nvidia.cuh"  // pulls in glass::nvidia::reduce/dot/nrm2 (CUB-backed)

static const int THREADS = 256;

// ─── glass::nvidia kernels (CUB-backed via glass-nvidia.cuh) ─────────────────
// One variant: glass::nvidia::reduce<float,N,THREADS>. Writes a per-iter sink
// to defeat dead-store elimination.
template<int N>
__global__ void k_nv_reduce(float* x, volatile float* sink, int iters) {
    extern __shared__ float scratch[];
    for (int rep = 0; rep < iters; rep++) {
        // reload x into a temp inside scratch each iter (don't overwrite x[0])
        for (int i = threadIdx.x; i < N; i += blockDim.x) scratch[N + i] = x[i];
        __syncthreads();
        glass::nvidia::reduce<float, N, THREADS>(scratch + N, scratch);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = scratch[N];
        __syncthreads();
    }
}

template<int N>
__global__ void k_nv_dot(float* x, float* y, volatile float* sink, int iters) {
    extern __shared__ float scratch[];
    for (int rep = 0; rep < iters; rep++) {
        float out;
        glass::nvidia::dot<float, N, THREADS>(x, y, &out, scratch);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = out;
        __syncthreads();
    }
}

template<int N>
__global__ void k_nv_nrm2(float* x, volatile float* sink, int iters) {
    extern __shared__ float scratch[];
    for (int rep = 0; rep < iters; rep++) {
        float out;
        glass::nvidia::nrm2<float, N, THREADS>(x, &out, scratch);
        __syncthreads();
        if (threadIdx.x == 0) sink[rep & 0xFF] = out;
        __syncthreads();
    }
}

// ─── GLASS kernels ────────────────────────────────────────────────────────────

__global__ void k_glass_reduce(float* x, int n, int iters) {
    extern __shared__ float smem[];
    // copy x into shared scratch for each iteration
    for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = x[i];
    __syncthreads();
    for (int rep = 0; rep < iters; rep++) {
        // reload from global each iteration to avoid dead-code elimination
        for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = x[i];
        __syncthreads();
        glass::reduce_fast(n, smem, smem + n);
        __syncthreads();
    }
    if (threadIdx.x == 0) x[0] = smem[0]; // prevent dead-code elimination
}

__global__ void k_glass_dot(float* x, float* y, int n, int iters) {
    extern __shared__ float smem[];
    for (int rep = 0; rep < iters; rep++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = x[i];
        __syncthreads();
        glass::dot_fast(n, smem, y, smem + n, smem + n + 1);
        __syncthreads();
    }
    if (threadIdx.x == 0) x[0] = smem[0];
}

__global__ void k_glass_nrm2(float* x, int n, int iters) {
    extern __shared__ float smem[];
    for (int rep = 0; rep < iters; rep++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = x[i];
        __syncthreads();
        glass::nrm2_fast(n, smem, smem + n);
        __syncthreads();
    }
    if (threadIdx.x == 0) x[0] = smem[0];
}

// glass::reduce_fast_min — block-min over an n-length array. Each thread folds a
// strided slice of x to a per-thread minimum (seeding empty lanes with +inf, the
// min identity) and hands it to reduce_fast_min(partial, scratch), which returns
// the block minimum to every thread. Mirrors test_l1's k_reduce_min_partial_hs.
__global__ void k_glass_reduce_min(float* x, int n, int iters) {
    extern __shared__ float smem[];   // s_scratch: ceil(THREADS/32) floats
    uint32_t rank = threadIdx.x;
    for (int rep = 0; rep < iters; rep++) {
        float partial = INFINITY;
        for (int i = rank; i < n; i += blockDim.x) partial = fminf(partial, x[i]);
        float total = glass::reduce_fast_min(partial, smem);
        __syncthreads();
        if (rank == 0) x[0] = total;   // prevent dead-code elimination
    }
}

// ─── CUB kernels ──────────────────────────────────────────────────────────────

__global__ void k_cub_reduce(float* x, float* out, int n, int iters) {
    typedef cub::BlockReduce<float, THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float val = (threadIdx.x < n) ? x[threadIdx.x] : 0.f;
    for (int rep = 0; rep < iters; rep++) {
        val = (threadIdx.x < n) ? x[threadIdx.x] : 0.f;
        float result = BlockReduce(temp).Sum(val);
        if (threadIdx.x == 0) out[0] = result;
    }
}

// ─── GLASS compile-time kernels ───────────────────────────────────────────────

#define DEFINE_GLASS_REDUCE_CT(N)                                                            \
    namespace glass_reduce_ct_##N {                                                          \
        __global__ void k_reduce(float* x, int iters) {                                      \
            extern __shared__ float smem[];                                                   \
            for (int rep = 0; rep < iters; rep++) {                                           \
                for (int i = threadIdx.x; i < N; i += blockDim.x) smem[i] = x[i];           \
                __syncthreads();                                                              \
                glass::reduce_fast<float, N>(smem, smem + N);                         \
                __syncthreads();                                                              \
            }                                                                                 \
            if (threadIdx.x == 0) x[0] = smem[0];                                            \
        }                                                                                     \
        __global__ void k_dot(float* x, float* y, int iters) {                               \
            extern __shared__ float smem[];                                                   \
            for (int rep = 0; rep < iters; rep++) {                                           \
                for (int i = threadIdx.x; i < N; i += blockDim.x) smem[i] = x[i];           \
                __syncthreads();                                                              \
                glass::dot_fast<float, N>(smem, y, smem + N, smem + N + 1);          \
                __syncthreads();                                                              \
            }                                                                                 \
            if (threadIdx.x == 0) x[0] = smem[0];                                            \
        }                                                                                     \
        __global__ void k_nrm2(float* x, int iters) {                                      \
            extern __shared__ float smem[];                                                   \
            for (int rep = 0; rep < iters; rep++) {                                           \
                for (int i = threadIdx.x; i < N; i += blockDim.x) smem[i] = x[i];           \
                __syncthreads();                                                              \
                glass::nrm2_fast<float, N>(smem, smem + N);                         \
                __syncthreads();                                                              \
            }                                                                                 \
            if (threadIdx.x == 0) x[0] = smem[0];                                            \
        }                                                                                     \
    }

DEFINE_GLASS_REDUCE_CT(4)
DEFINE_GLASS_REDUCE_CT(6)
DEFINE_GLASS_REDUCE_CT(8)
DEFINE_GLASS_REDUCE_CT(12)
DEFINE_GLASS_REDUCE_CT(14)
DEFINE_GLASS_REDUCE_CT(24)
DEFINE_GLASS_REDUCE_CT(64)
DEFINE_GLASS_REDUCE_CT(256)

// ─── glass::thread:: tier (one problem per THREAD, 32 packed per warp) ─────────
// A block of THREADS threads solves THREADS independent length-N dot products
// concurrently — thread p owns problem p, operands live in registers (no shared
// scratch, no shuffles, no barriers). Reported time is amortized PER PROBLEM
// (elapsed / (iters * THREADS)) to expose the low-DOF packing win. Compile-time
// N only, and only worth it below the N<=7 register-residency ceiling (CLAUDE.md),
// so the tier is instantiated for the small sizes 4 and 6.
#define DEFINE_THREAD_DOT_CT(N)                                                              \
    namespace thread_dot_ct_##N {                                                            \
        __global__ void k_dot(const float* x, const float* y, volatile float* sink, int iters) { \
            int p = blockIdx.x*blockDim.x + threadIdx.x;                                     \
            float xv[N], yv[N];                                                              \
            for (int i = 0; i < N; i++) { xv[i] = x[(size_t)p*N + i]; yv[i] = y[(size_t)p*N + i]; } \
            float acc = 0.f;                                                                 \
            for (int rep = 0; rep < iters; rep++) {                                          \
                acc += glass::thread::dot<float, N>(xv, yv);                                 \
                yv[0] = acc;   /* feed back so the loop can't be hoisted */                  \
            }                                                                                 \
            sink[p & 0xFF] = acc;                                                            \
        }                                                                                     \
    }

DEFINE_THREAD_DOT_CT(4)
DEFINE_THREAD_DOT_CT(6)

// ─── Timing helper ────────────────────────────────────────────────────────────

static double elapsed_us(struct timespec a, struct timespec b) {
    return (double)(b.tv_sec - a.tv_sec) * 1e6
         + (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
}

static void bench(const char* name, void(*launch)(float*, float*, int, int, int, struct timespec*, struct timespec*),
                  float* dx, float* dy, int n, int iters) {
    struct timespec t0, t1;
    launch(dx, dy, n, iters, THREADS, &t0, &t1);
    printf("%-28s n=%3d  %.3f us/op\n", name, n, elapsed_us(t0, t1) / iters);
}

int main(int argc, char** argv) {
    int n     = (argc > 1) ? atoi(argv[1]) : 256;
    int iters = (argc > 2) ? atoi(argv[2]) : 100000;

    if (n > THREADS) {
        fprintf(stderr, "n must be <= %d (block size)\n", THREADS);
        return 1;
    }

    // Allocate device memory
    float *dx, *dy, *dout;
    cudaMalloc(&dx,   n * sizeof(float));
    cudaMalloc(&dy,   n * sizeof(float));
    cudaMalloc(&dout, sizeof(float));

    // Initialize with non-zero values
    float* hx = new float[n];
    for (int i = 0; i < n; i++) hx[i] = (float)(i + 1) / n;
    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    delete[] hx;

    // Scratch: n floats for data copy + ceil(THREADS/32) for warp partials
    int smem_bytes = (n + THREADS / 32 + 1) * sizeof(float);

    struct timespec t0, t1;

    // ─── glass::reduce_fast ────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_reduce<<<1, THREADS, smem_bytes>>>(dx, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::reduce_fast    n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── glass::dot_fast ───────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_dot<<<1, THREADS, smem_bytes>>>(dx, dy, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::dot_fast       n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── glass::nrm2_fast ────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_nrm2<<<1, THREADS, smem_bytes>>>(dx, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::nrm2_fast    n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── glass::reduce_fast_min ──────────────────────────────
    // (dx[0] is already clobbered by the sum-reduces above; each op reloads the
    // full array from dx every iter, so an altered dx[0] doesn't affect timing.)
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_reduce_min<<<1, THREADS, smem_bytes>>>(dx, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::reduce_fast_min  n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── cub::BlockReduce ────────────────────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_cub_reduce<<<1, THREADS>>>(dx, dout, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("cub::BlockReduce             n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── glass::high_speed CT (only for pre-instantiated sizes) ──────────────
    #define MAYBE_GLASS_REDUCE_CT(N)                                                     \
        if (n == N) {                                                                     \
            clock_gettime(CLOCK_MONOTONIC, &t0);                                         \
            glass_reduce_ct_##N::k_reduce<<<1, THREADS, smem_bytes>>>(dx, iters);       \
            cudaDeviceSynchronize();                                                      \
            clock_gettime(CLOCK_MONOTONIC, &t1);                                         \
            printf("glass::hs::reduce<CT>        n=%3d  %.3f us/op\n",                  \
                   N, elapsed_us(t0, t1) / iters);                                        \
            clock_gettime(CLOCK_MONOTONIC, &t0);                                         \
            glass_reduce_ct_##N::k_dot<<<1, THREADS, smem_bytes>>>(dx, dy, iters);      \
            cudaDeviceSynchronize();                                                      \
            clock_gettime(CLOCK_MONOTONIC, &t1);                                         \
            printf("glass::hs::dot<CT>           n=%3d  %.3f us/op\n",                  \
                   N, elapsed_us(t0, t1) / iters);                                        \
            clock_gettime(CLOCK_MONOTONIC, &t0);                                         \
            glass_reduce_ct_##N::k_nrm2<<<1, THREADS, smem_bytes>>>(dx, iters);       \
            cudaDeviceSynchronize();                                                      \
            clock_gettime(CLOCK_MONOTONIC, &t1);                                         \
            printf("glass::hs::nrm2<CT>        n=%3d  %.3f us/op\n",                  \
                   N, elapsed_us(t0, t1) / iters);                                        \
        }
    MAYBE_GLASS_REDUCE_CT(4)
    MAYBE_GLASS_REDUCE_CT(6)
    MAYBE_GLASS_REDUCE_CT(8)
    MAYBE_GLASS_REDUCE_CT(12)
    MAYBE_GLASS_REDUCE_CT(14)
    MAYBE_GLASS_REDUCE_CT(24)
    MAYBE_GLASS_REDUCE_CT(64)
    MAYBE_GLASS_REDUCE_CT(256)
    #undef MAYBE_GLASS_REDUCE_CT

    // ─── glass::nvidia (CUB-backed) — only for pre-instantiated sizes ────────
    float *dSink;
    cudaMalloc(&dSink, 256 * sizeof(float));
    constexpr size_t nv_smem = glass::nvidia::reduce_scratch_bytes<float, THREADS>();
    #define MAYBE_NV_REDUCE_CT(N)                                                            \
        if (n == N) {                                                                         \
            constexpr size_t nv_smem_total = nv_smem + 2 * N * sizeof(float);                \
            clock_gettime(CLOCK_MONOTONIC, &t0);                                              \
            k_nv_reduce<N><<<1, THREADS, nv_smem_total>>>(dx, dSink, iters);                 \
            cudaDeviceSynchronize();                                                          \
            clock_gettime(CLOCK_MONOTONIC, &t1);                                              \
            printf("glass::nvidia::reduce<CT>    n=%3d  %.3f us/op\n",                       \
                   N, elapsed_us(t0, t1) / iters);                                            \
            clock_gettime(CLOCK_MONOTONIC, &t0);                                              \
            k_nv_dot<N><<<1, THREADS, nv_smem>>>(dx, dy, dSink, iters);                      \
            cudaDeviceSynchronize();                                                          \
            clock_gettime(CLOCK_MONOTONIC, &t1);                                              \
            printf("glass::nvidia::dot<CT>       n=%3d  %.3f us/op\n",                       \
                   N, elapsed_us(t0, t1) / iters);                                            \
            clock_gettime(CLOCK_MONOTONIC, &t0);                                              \
            k_nv_nrm2<N><<<1, THREADS, nv_smem>>>(dx, dSink, iters);                       \
            cudaDeviceSynchronize();                                                          \
            clock_gettime(CLOCK_MONOTONIC, &t1);                                              \
            printf("glass::nvidia::nrm2<CT>    n=%3d  %.3f us/op\n",                       \
                   N, elapsed_us(t0, t1) / iters);                                            \
        }
    MAYBE_NV_REDUCE_CT(4)
    MAYBE_NV_REDUCE_CT(6)
    MAYBE_NV_REDUCE_CT(8)
    MAYBE_NV_REDUCE_CT(12)
    MAYBE_NV_REDUCE_CT(14)
    MAYBE_NV_REDUCE_CT(24)
    MAYBE_NV_REDUCE_CT(64)
    MAYBE_NV_REDUCE_CT(256)
    #undef MAYBE_NV_REDUCE_CT

    // ─── glass::thread:: dot (one problem per thread) — only for N in {4,6} ───
    // THREADS packed problems; time reported amortized per problem.
    {
        const int TN = (n == 4) ? 4 : (n == 6) ? 6 : 0;
        if (TN) {
            float *dxt, *dyt;
            cudaMalloc(&dxt, (size_t)THREADS * TN * sizeof(float));
            cudaMalloc(&dyt, (size_t)THREADS * TN * sizeof(float));
            float* ht = new float[(size_t)THREADS * TN];
            for (int i = 0; i < THREADS * TN; i++) ht[i] = (float)((i % TN) + 1) / TN;
            cudaMemcpy(dxt, ht, (size_t)THREADS * TN * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dyt, ht, (size_t)THREADS * TN * sizeof(float), cudaMemcpyHostToDevice);
            delete[] ht;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (TN == 4) thread_dot_ct_4::k_dot<<<1, THREADS>>>(dxt, dyt, dSink, iters);
            else         thread_dot_ct_6::k_dot<<<1, THREADS>>>(dxt, dyt, dSink, iters);
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_MONOTONIC, &t1);
            printf("glass::thread::dot<CT>       n=%3d  %.4f us/problem (%d-packed)\n",
                   TN, elapsed_us(t0, t1) / ((double)iters * THREADS), THREADS);
            cudaFree(dxt); cudaFree(dyt);
        }
    }

    cudaFree(dx); cudaFree(dy); cudaFree(dout); cudaFree(dSink);
    return 0;
}
