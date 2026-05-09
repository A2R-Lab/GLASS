// bench_reduce.cu — L1 benchmarks: glass::simple reduce/dot/l2norm vs. CUB BlockReduce
// Compilation: nvcc -std=c++17 -arch=sm_XX -I.. -I../src -O3 bench_reduce.cu -o bench_reduce
// Usage: ./bench_reduce <n> [iters]

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cub/cub.cuh>
#include "../glass.cuh"

static const int THREADS = 256;

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
        glass::simple::high_speed::reduce(n, smem, smem + n);
        __syncthreads();
    }
    if (threadIdx.x == 0) x[0] = smem[0]; // prevent dead-code elimination
}

__global__ void k_glass_dot(float* x, float* y, int n, int iters) {
    extern __shared__ float smem[];
    for (int rep = 0; rep < iters; rep++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = x[i];
        __syncthreads();
        glass::simple::high_speed::dot(n, smem, y, smem + n, smem + n + 1);
        __syncthreads();
    }
    if (threadIdx.x == 0) x[0] = smem[0];
}

__global__ void k_glass_l2norm(float* x, int n, int iters) {
    extern __shared__ float smem[];
    for (int rep = 0; rep < iters; rep++) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) smem[i] = x[i];
        __syncthreads();
        glass::simple::high_speed::l2norm(n, smem, smem + n);
        __syncthreads();
    }
    if (threadIdx.x == 0) x[0] = smem[0];
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
    int iters = (argc > 2) ? atoi(argv[2]) : 10000;

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

    // ─── glass::simple::high_speed::reduce ────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_reduce<<<1, THREADS, smem_bytes>>>(dx, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::simple::hs::reduce    n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── glass::simple::high_speed::dot ───────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_dot<<<1, THREADS, smem_bytes>>>(dx, dy, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::simple::hs::dot       n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── glass::simple::high_speed::l2norm ────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_glass_l2norm<<<1, THREADS, smem_bytes>>>(dx, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("glass::simple::hs::l2norm    n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    // ─── cub::BlockReduce ────────────────────────────────────────────────────
    clock_gettime(CLOCK_MONOTONIC, &t0);
    k_cub_reduce<<<1, THREADS>>>(dx, dout, n, iters);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("cub::BlockReduce             n=%3d  %.3f us/op\n", n, elapsed_us(t0, t1) / iters);

    cudaFree(dx); cudaFree(dy); cudaFree(dout);
    return 0;
}
