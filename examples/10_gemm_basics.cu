// 10_gemm_basics.cu — the standard-BLAS GEMM convention and its transpose flags.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 10_gemm_basics.cu -o gemm_basics && ./gemm_basics
//
// GLASS gemm follows the standard BLAS / cuBLAS / NumPy / Eigen convention:
//
//   C = alpha * op(A) * op(B) + beta * C    (column-major)
//   C is M×N,  contraction K.
//   op(A) is M×K:  TRANSPOSE_A=false ⇒ A is M×K ;  true ⇒ A is K×M (op(A)=Aᵀ).
//   op(B) is K×N:  TRANSPOSE_B=false ⇒ B is K×N ;  true ⇒ B is N×K (op(B)=Bᵀ).
//
//   NumPy:  C = alpha * opA(A) @ opB(B) + beta * C
//   Eigen:  C.noalias() = alpha * (opA(A) * opB(B)) + beta * C;   // col-major
//
// We deliberately use a NON-SQUARE shape (M=2, N=3, K=4): the dimension order
// matters, and a square example would hide a wrong mapping.

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static constexpr int M = 2, N = 3, K = 4;

template <bool TA, bool TB>
__global__ void run(const float* A, const float* B, float* C) {
    // beta = 0 overload: C is overwritten (never read).
    glass::gemm<float, M, N, K, TA, TB>(1.0f, const_cast<float*>(A), const_cast<float*>(B), C);
}

// Host reference: logical op(A) is M×K, op(B) is K×N, C is M×N (all col-major).
static void ref(const float* opA, const float* opB, float* C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += opA[m + k*M] * opB[k + n*K];
            C[m + n*M] = s;
        }
}

int main() {
    // opA (M×K) and opB (K×N) are the LOGICAL operands.
    float opA[M*K], opB[K*N];
    for (int i = 0; i < M*K; i++) opA[i] = 0.1f * (i + 1);
    for (int i = 0; i < K*N; i++) opB[i] = 0.2f * (i + 1) - 0.5f;

    // Physical storage for each transpose flag: a transposed operand is stored
    // as op(_)ᵀ in column-major (i.e. A is K×M when TRANSPOSE_A).
    float A_n[M*K], A_t[K*M], B_n[K*N], B_t[N*K];
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) { A_n[m + k*M] = opA[m + k*M]; A_t[k + m*K] = opA[m + k*M]; }
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) { B_n[k + n*K] = opB[k + n*K]; B_t[n + k*N] = opB[k + n*K]; }

    float ref_C[M*N]; ref(opA, opB, ref_C);

    float *dA, *dB, *dC; cudaMalloc(&dA, sizeof(float)*K*M); cudaMalloc(&dB, sizeof(float)*N*K); cudaMalloc(&dC, sizeof(float)*M*N);
    const char* names[4] = {"C = A * B     ", "C = AT * B    ", "C = A * BT    ", "C = AT * BT   "};
    int bad = 0;
    for (int combo = 0; combo < 4; combo++) {
        bool ta = combo & 2, tb = combo & 1;
        cudaMemcpy(dA, ta ? A_t : A_n, sizeof(float)*(ta ? K*M : M*K), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, tb ? B_t : B_n, sizeof(float)*(tb ? N*K : K*N), cudaMemcpyHostToDevice);
        if      (!ta && !tb) run<false,false><<<1,64>>>(dA, dB, dC);
        else if ( ta && !tb) run<true ,false><<<1,64>>>(dA, dB, dC);
        else if (!ta &&  tb) run<false,true ><<<1,64>>>(dA, dB, dC);
        else                 run<true ,true ><<<1,64>>>(dA, dB, dC);
        cudaDeviceSynchronize();
        float C[M*N]; cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost);
        float md = 0; for (int i = 0; i < M*N; i++) md = fmaxf(md, fabsf(C[i] - ref_C[i]));
        printf("  %s  max_err=%.2e  %s\n", names[combo], md, md < 1e-5 ? "ok" : "FAIL");
        bad += (md >= 1e-5);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
