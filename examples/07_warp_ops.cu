// 07_warp_ops.cu — single-warp (glass::warp::) primitives, launched <<<1,32>>>.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 07_warp_ops.cu -o warp_ops && ./warp_ops
//
// The glass::warp:: namespace holds warp-scoped SIMT variants (raw __shfl, one
// 32-lane warp, no shared scratch, no __syncthreads) for warp-per-problem
// kernels — e.g. a block that processes many independent problems, one per warp.
// No cooperative groups, no vendor (cuBLASDx/cuSOLVERDx) deps.

#include "glass.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// Sum a vector within one warp; result in x[0].
__global__ void warp_reduce_kernel(float *x, int n) {
    glass::warp::reduce(static_cast<uint32_t>(n), x);
}

// 4x4 column-major GEMM C = A*B within one warp (the mat4-multiply use case).
__global__ void warp_gemm_kernel(float *A, float *B, float *C) {
    glass::warp::gemm<float, 4, 4, 4>(1.0f, A, B, 0.0f, C);
}

// SPD solve A x = b within one warp: factor A = L Lᵀ, then forward + transpose solve.
__global__ void warp_posv_kernel(float *A, float *b) {
    glass::warp::cholDecomp_InPlace<float, 3>(A);
    glass::warp::trsm<float, 3>(A, b);
    glass::warp::trsm_transpose<float, 3>(A, b);
}

int main() {
    // ── warp::reduce ──
    const int n = 8;
    float hx[n];
    for (int i = 0; i < n; ++i) hx[i] = static_cast<float>(i + 1);   // sum = 36
    float *dx; cudaMalloc(&dx, n * sizeof(float));
    cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice);
    warp_reduce_kernel<<<1, 32>>>(dx, n);
    cudaDeviceSynchronize();
    float s = 0.f; cudaMemcpy(&s, dx, sizeof(float), cudaMemcpyDeviceToHost);
    printf("glass::warp::reduce      sum(1..8) = %.0f (expect 36)\n", s);

    // ── warp::gemm (4x4, column-major; B = identity ⇒ C = A) ──
    float hA[16], hB[16] = {0}, hC[16] = {0};
    for (int i = 0; i < 16; ++i) hA[i] = static_cast<float>(i);
    for (int i = 0; i < 4; ++i) hB[i*4 + i] = 1.0f;                  // identity
    float *dA, *dB, *dC;
    cudaMalloc(&dA, 16*sizeof(float)); cudaMalloc(&dB, 16*sizeof(float)); cudaMalloc(&dC, 16*sizeof(float));
    cudaMemcpy(dA, hA, 16*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 16*sizeof(float), cudaMemcpyHostToDevice);
    warp_gemm_kernel<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, 16*sizeof(float), cudaMemcpyDeviceToHost);
    printf("glass::warp::gemm        C[0,5,10,15] = %.0f %.0f %.0f %.0f (expect 0 5 10 15)\n",
           hC[0], hC[5], hC[10], hC[15]);

    // ── warp:: SPD solve (3x3, column-major) ──
    // A = [[4,1,0],[1,3,1],[0,1,2]] (SPD), b = [1,2,3]; solve A x = b.
    float hAs[9] = {4,1,0, 1,3,1, 0,1,2};   // column-major == symmetric here
    float hb[3]  = {1,2,3};
    float *dAs, *db;
    cudaMalloc(&dAs, 9*sizeof(float)); cudaMalloc(&db, 3*sizeof(float));
    cudaMemcpy(dAs, hAs, 9*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, 3*sizeof(float), cudaMemcpyHostToDevice);
    warp_posv_kernel<<<1, 32>>>(dAs, db);
    cudaDeviceSynchronize();
    float hx3[3]; cudaMemcpy(hx3, db, 3*sizeof(float), cudaMemcpyDeviceToHost);
    printf("glass::warp:: SPD solve  x = %.3f %.3f %.3f\n", hx3[0], hx3[1], hx3[2]);

    cudaFree(dx); cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dAs); cudaFree(db);
    return 0;
}
