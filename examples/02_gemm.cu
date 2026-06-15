// 02_gemm.cu — single-block GEMM with layout flags, pure SIMT (no MathDx).
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 02_gemm.cu -o gemm && ./gemm
//
// Computes C = alpha*A*B + beta*C, column-major. A is m x n, B is n x k,
// C is m x k. Uses both the runtime-size and compile-time-size overloads.

#include "glass.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// Runtime-size overload: sizes passed as args.
__global__ void gemm_rt(float *A, float *B, float *C, int m, int n, int k) {
    glass::gemm(static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                static_cast<uint32_t>(k), 1.f, A, B, 0.f, C);
}

// Compile-time-size overload: sizes baked in as template params (loop unrolling).
__global__ void gemm_ct(float *A, float *B, float *C) {
    glass::gemm<float, 2, 2, 2>(1.f, A, B, 0.f, C);
}

int main() {
    const int m = 2, n = 2, k = 2;
    // Column-major 2x2 identity-ish inputs: A = [[1,3],[2,4]], B = I  => C = A.
    float hA[] = {1, 2, 3, 4};   // A[row + col*m]
    float hB[] = {1, 0, 0, 1};   // B = 2x2 identity
    float hC[m * k] = {0};

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    gemm_ct<<<1, 256>>>(dA, dB, dC);   // swap for gemm_rt<<<...>>>(dA,dB,dC,m,n,k)
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
    printf("C = A*I (col-major) -> [%.0f %.0f; %.0f %.0f]\n",
           hC[0], hC[2], hC[1], hC[3]);   // expect [1 3; 2 4]

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
