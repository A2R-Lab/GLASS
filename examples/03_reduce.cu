// 03_reduce.cu — block reductions (sum + dot + high-speed warp-shuffle), pure SIMT.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 03_reduce.cu -o reduce && ./reduce
//
// glass::reduce sums a vector in place (result lands in x[0]); the high_speed
// variant uses warp-shuffle and needs ceil(blockDim/32)*sizeof(T) scratch.

#include "glass.cuh"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void reduce_kernel(float *x, int n) {
    glass::reduce(static_cast<uint32_t>(n), x);   // x[0] = sum(x)
}

__global__ void reduce_hs_kernel(float *x, int n) {
    // Scratch: one float per warp = ceil(blockDim.x / 32) floats.
    extern __shared__ float scratch[];
    glass::reduce_fast(static_cast<uint32_t>(n), x, scratch);   // x[0] = sum(x)
}

int main() {
    const int n = 8;
    float hx[n];
    for (int i = 0; i < n; ++i) hx[i] = static_cast<float>(i + 1);   // sum = 36

    float *dx;
    cudaMalloc(&dx, n * sizeof(float));

    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    reduce_kernel<<<1, 256>>>(dx, n);
    cudaDeviceSynchronize();
    float out = 0.f;
    cudaMemcpy(&out, dx, sizeof(float), cudaMemcpyDeviceToHost);
    printf("glass::reduce            sum(1..8) = %.0f (expect 36)\n", out);

    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    const int threads = 256;
    size_t smem = ((threads + 31) / 32) * sizeof(float);
    reduce_hs_kernel<<<1, threads, smem>>>(dx, n);
    cudaDeviceSynchronize();
    cudaMemcpy(&out, dx, sizeof(float), cudaMemcpyDeviceToHost);
    printf("glass::reduce_fast sum(1..8) = %.0f (expect 36)\n", out);

    cudaFree(dx);
    return 0;
}
