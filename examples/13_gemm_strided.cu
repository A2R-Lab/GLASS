// 13_gemm_strided.cu — GEMM on column-major sub-blocks with explicit
// leading dimensions (the "strided" GEMM).
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 13_gemm_strided.cu -o rsgemm && ./rsgemm
//
// Same standard convention as glass::gemm (C is M×N, contraction K), but A and B
// live inside larger buffers with custom column strides (leading dimensions):
//
//   A is M×K, leading dim A_RS ≥ M :  A[m][k] = A_buf[m + k*A_RS]
//   B is K×N, leading dim B_RS ≥ K :  B[k][n] = B_buf[k + n*B_RS]
//   C is M×N, standard column-major (LDC = M).
//
//   alpha/beta are at the FRONT, matching every other GLASS op:
//     glass::gemm_strided<T, M, N, K, A_RS, B_RS>(alpha, A, B, beta, C);
//   NumPy:  C = alpha * A @ B + beta * C    (on the strided sub-views)

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static constexpr int M = 5, N = 7, K = 3, A_RS = 8, B_RS = 6;  // A_RS>M, B_RS>K

__global__ void run(const float* A, const float* B, float beta, float* C) {
    glass::gemm_strided<float, M, N, K, A_RS, B_RS>(1.5f, const_cast<float*>(A), const_cast<float*>(B), beta, C);
}

int main() {
    float A[A_RS*K], B[B_RS*N], C[M*N];
    for (int i = 0; i < A_RS*K; i++) A[i] = 0.1f*i - 0.3f;
    for (int i = 0; i < B_RS*N; i++) B[i] = 0.2f*i - 0.5f;
    for (int i = 0; i < M*N;   i++) C[i] = 1.0f;
    const float alpha = 1.5f, beta = 0.25f;

    float ref[M*N];
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float s = 0; for (int k = 0; k < K; k++) s += A[m + k*A_RS] * B[k + n*B_RS];
        ref[m + n*M] = alpha*s + beta*C[m + n*M];
    }

    float *dA, *dB, *dC; cudaMalloc(&dA, sizeof(A)); cudaMalloc(&dB, sizeof(B)); cudaMalloc(&dC, sizeof(C));
    cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, sizeof(C), cudaMemcpyHostToDevice);
    run<<<1, 64>>>(dA, dB, beta, dC); cudaDeviceSynchronize();
    float out[M*N]; cudaMemcpy(out, dC, sizeof(out), cudaMemcpyDeviceToHost);

    float md = 0; for (int i = 0; i < M*N; i++) md = fmaxf(md, fabsf(out[i] - ref[i]));
    printf("  gemm_strided %dx%dx%d  A_RS=%d B_RS=%d  max_err=%.2e\n", M, N, K, A_RS, B_RS, md);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    printf(md < 1e-4 ? "PASS\n" : "FAIL\n");
    return md < 1e-4 ? 0 : 1;
}
