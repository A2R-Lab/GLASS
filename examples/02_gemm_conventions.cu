// 02_gemm_conventions.cu — THE single-block GEMM example: the standard-BLAS
// convention, both size overloads, all four transpose combos, why row-major
// needs no flag, and the cooperative-groups spelling of the same call.
// (Merged from the former 02_gemm / 04_cgrps / 10_gemm_basics /
// 11_rowmajor_is_transpose examples, 2026-08-11.)
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 02_gemm_conventions.cu -o gemm && ./gemm
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

#include "glass-cgrps.cuh"   // pulls glass.cuh; also enables the cgrps section
#include <cooperative_groups.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

static constexpr int M = 2, N = 3, K = 4;

// ─── §1  the four transpose combos (compile-time-size overload) ──────────────
template <bool TA, bool TB>
__global__ void run(const float* A, const float* B, float* C) {
    // beta = 0 overload: C is overwritten (never read).
    glass::block::gemm<float, M, N, K, TA, TB>(1.0f, const_cast<float*>(A), const_cast<float*>(B), C);
}

// ─── §2  the runtime-size overload — sizes as arguments, same convention ─────
__global__ void run_rt(const float* A, const float* B, float* C, int m, int n, int k) {
    glass::block::gemm(static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                       static_cast<uint32_t>(k),
                       1.0f, const_cast<float*>(A), const_cast<float*>(B), 0.0f, C);
}

// ─── §3  row-major is just a transpose (why there is no ROW_MAJOR_A flag) ────
// A row-major M×K matrix occupies the SAME bytes as a column-major K×M matrix,
// so a row-major A is read with TRANSPOSE_A=true — bit-identically.
__global__ void run_ta(const float* A, const float* B, float* C) {
    glass::block::gemm<float, M, N, K, /*TA=*/true, /*TB=*/false>(1.f, const_cast<float*>(A), const_cast<float*>(B), C);
}

// ─── §4  the cooperative-groups spelling — same numerics, group-driven ───────
__global__ void run_cgrps(const float* A, const float* B, float* C) {
    // Whole block (default group). A sub-block tile also works:
    //   auto warp = cooperative_groups::tiled_partition<32>(
    //       cooperative_groups::this_thread_block());
    //   glass::cgrps::gemm<float, M, N, K>(1.f, ..., warp);
    glass::cgrps::gemm<float, M, N, K>(1.f, const_cast<float*>(A), const_cast<float*>(B), 0.f, C);
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

    // Physical storage per transpose flag (a transposed operand is op(_)ᵀ col-major).
    float A_n[M*K], A_t[K*M], B_n[K*N], B_t[N*K];
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) { A_n[m + k*M] = opA[m + k*M]; A_t[k + m*K] = opA[m + k*M]; }
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) { B_n[k + n*K] = opB[k + n*K]; B_t[n + k*N] = opB[k + n*K]; }

    float ref_C[M*N]; ref(opA, opB, ref_C);

    float *dA, *dB, *dC; cudaMalloc(&dA, sizeof(float)*K*M); cudaMalloc(&dB, sizeof(float)*N*K); cudaMalloc(&dC, sizeof(float)*M*N);
    const char* names[4] = {"C = A * B     ", "C = AT * B    ", "C = A * BT    ", "C = AT * BT   "};
    int bad = 0;
    float C[M*N];

    // §1 — all four transpose combos vs the host reference.
    for (int combo = 0; combo < 4; combo++) {
        bool ta = combo & 2, tb = combo & 1;
        cudaMemcpy(dA, ta ? A_t : A_n, sizeof(float)*(ta ? K*M : M*K), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, tb ? B_t : B_n, sizeof(float)*(tb ? N*K : K*N), cudaMemcpyHostToDevice);
        if      (!ta && !tb) run<false,false><<<1,64>>>(dA, dB, dC);
        else if ( ta && !tb) run<true ,false><<<1,64>>>(dA, dB, dC);
        else if (!ta &&  tb) run<false,true ><<<1,64>>>(dA, dB, dC);
        else                 run<true ,true ><<<1,64>>>(dA, dB, dC);
        cudaDeviceSynchronize();
        cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost);
        float md = 0; for (int i = 0; i < M*N; i++) md = fmaxf(md, fabsf(C[i] - ref_C[i]));
        printf("  %s  max_err=%.2e  %s\n", names[combo], md, md < 1e-5 ? "ok" : "FAIL");
        bad += (md >= 1e-5);
    }

    // §2 — runtime-size overload matches the compile-time one.
    cudaMemcpy(dA, A_n, sizeof(A_n), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_n, sizeof(B_n), cudaMemcpyHostToDevice);
    run_rt<<<1,64>>>(dA, dB, dC, M, N, K); cudaDeviceSynchronize();
    cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost);
    { float md = 0; for (int i = 0; i < M*N; i++) md = fmaxf(md, fabsf(C[i] - ref_C[i]));
      printf("  runtime-size overload  max_err=%.2e  %s\n", md, md < 1e-5 ? "ok" : "FAIL");
      bad += (md >= 1e-5); }

    // §3 — row-major A via TRANSPOSE_A is BIT-identical to col-major NN.
    float C_nn[M*N], C_ta[M*N];
    float A_rowmajor[M*K];   // A[m*K + k] == bytes of the K×M col-major transpose
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) A_rowmajor[m*K + k] = opA[m + k*M];
    cudaMemcpy(dA, A_n, sizeof(A_n), cudaMemcpyHostToDevice);
    run<false,false><<<1,64>>>(dA, dB, dC); cudaDeviceSynchronize();
    cudaMemcpy(C_nn, dC, sizeof(C_nn), cudaMemcpyDeviceToHost);
    cudaMemcpy(dA, A_rowmajor, sizeof(A_rowmajor), cudaMemcpyHostToDevice);
    run_ta<<<1,64>>>(dA, dB, dC); cudaDeviceSynchronize();
    cudaMemcpy(C_ta, dC, sizeof(C_ta), cudaMemcpyDeviceToHost);
    { bool identical = (memcmp(C_nn, C_ta, sizeof(C_nn)) == 0);
      printf("  row-major-via-TRANSPOSE_A vs col-major NN: %s\n",
             identical ? "BIT-IDENTICAL" : "DIFFER");
      bad += !identical; }

    // §4 — the cgrps spelling produces the same answer.
    cudaMemcpy(dA, A_n, sizeof(A_n), cudaMemcpyHostToDevice);
    run_cgrps<<<1,64>>>(dA, dB, dC); cudaDeviceSynchronize();
    cudaMemcpy(C, dC, sizeof(C), cudaMemcpyDeviceToHost);
    { float md = 0; for (int i = 0; i < M*N; i++) md = fmaxf(md, fabsf(C[i] - ref_C[i]));
      printf("  glass::cgrps::gemm     max_err=%.2e  %s\n", md, md < 1e-5 ? "ok" : "FAIL");
      bad += (md >= 1e-5); }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
