// 01_axpy_simt.cu — basic L1 op (AXPY, runtime size), pure SIMT (no MathDx).
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 01_axpy_simt.cu -o axpy && ./axpy
//
// Computes y = alpha*x + y for a length-n vector inside a single block.

#include "glass.cuh"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void axpy_kernel(float *x, float *y, int n) {
    // Runtime size: every thread in the block strides over the n elements.
    glass::axpy(static_cast<uint32_t>(n), 1.5f, x, y);   // y = 1.5*x + y
}

int main() {
    const int n = 8;
    float hx[n], hy[n];
    for (int i = 0; i < n; ++i) { hx[i] = static_cast<float>(i); hy[i] = 1.0f; }

    float *dx, *dy;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMalloc(&dy, n * sizeof(float));
    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, n * sizeof(float), cudaMemcpyHostToDevice);

    // One block per independent data item (here: a single vector).
    axpy_kernel<<<1, 256>>>(dx, dy, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("y = 1.5*x + 1  ->");
    for (int i = 0; i < n; ++i) printf(" %.1f", hy[i]);   // expect 1.0 2.5 4.0 ...
    printf("\n");

    cudaFree(dx); cudaFree(dy);
    return 0;
}
