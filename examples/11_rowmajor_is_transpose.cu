// 11_rowmajor_is_transpose.cu — "row-major is just a transpose".
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 11_rowmajor_is_transpose.cu -o rmt && ./rmt
//
// GLASS gemm is column-major and has NO per-operand row-major flag (it was
// pruned). You don't need one: a row-major M×K matrix occupies the SAME bytes as
// a column-major K×M matrix, so to multiply by a row-major A you just pass it
// with TRANSPOSE_A=true. This example shows the two paths give BIT-IDENTICAL
// output, which is exactly why the separate ROW_MAJOR_A path was removed.
//
//   row-major A (M×K)  ==  column-major Aᵀ (K×M)   →   read with TRANSPOSE_A=true
//
// (The same holds for B via TRANSPOSE_B, and for a row-major C via ROW_MAJOR_C.)

#include "glass.cuh"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

static constexpr int M = 3, N = 2, K = 4;

__global__ void nn(const float* A, const float* B, float* C) {     // A col-major M×K
    glass::gemm<float, M, N, K, /*TA=*/false, /*TB=*/false>(1.f, const_cast<float*>(A), const_cast<float*>(B), C);
}
__global__ void ta(const float* A, const float* B, float* C) {     // A row-major M×K == col-major K×M
    glass::gemm<float, M, N, K, /*TA=*/true,  /*TB=*/false>(1.f, const_cast<float*>(A), const_cast<float*>(B), C);
}

int main() {
    // The SAME logical A (M×K), laid out two ways.
    float A_logical[M][K];
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) A_logical[m][k] = 0.3f*m - 0.1f*k + 1.f;
    float A_colmajor[M*K];   // A[m + k*M]
    float A_rowmajor[M*K];   // A[m*K + k]  ==  bytes of the K×M col-major transpose
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) {
        A_colmajor[m + k*M] = A_logical[m][k];
        A_rowmajor[m*K + k] = A_logical[m][k];
    }
    float B[K*N]; for (int i = 0; i < K*N; i++) B[i] = 0.2f*i - 0.4f;   // B col-major K×N

    float *dA, *dB, *dC; cudaMalloc(&dA, sizeof(A_colmajor)); cudaMalloc(&dB, sizeof(B)); cudaMalloc(&dC, sizeof(float)*M*N);
    cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice);

    float C_nn[M*N], C_ta[M*N];
    cudaMemcpy(dA, A_colmajor, sizeof(A_colmajor), cudaMemcpyHostToDevice);
    nn<<<1,64>>>(dA, dB, dC); cudaDeviceSynchronize();
    cudaMemcpy(C_nn, dC, sizeof(C_nn), cudaMemcpyDeviceToHost);

    cudaMemcpy(dA, A_rowmajor, sizeof(A_rowmajor), cudaMemcpyHostToDevice);
    ta<<<1,64>>>(dA, dB, dC); cudaDeviceSynchronize();
    cudaMemcpy(C_ta, dC, sizeof(C_ta), cudaMemcpyDeviceToHost);

    bool identical = (memcmp(C_nn, C_ta, sizeof(C_nn)) == 0);
    printf("  col-major NN  vs  row-major-via-TRANSPOSE_A : %s\n",
           identical ? "BIT-IDENTICAL" : "DIFFER");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    printf(identical ? "PASS\n" : "FAIL\n");
    return identical ? 0 : 1;
}
