// 10_ldlt_solve.cu — symmetric-INDEFINITE solve via LDLᵀ (no square root).
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 10_ldlt_solve.cu -o ldlt && ./ldlt
//
// Cholesky (potrf/posv) requires SPD. A symmetric matrix with NEGATIVE
// eigenvalues (a KKT / saddle-point system) has no Cholesky factor — but it
// does have A = L·D·Lᵀ with unit-lower L and a signed diagonal D. glass::ldlt
// factors it in place (SciPy: lu, d, _ = scipy.linalg.ldl(A, lower=True)) and
// glass::ldlt_solve runs the three sweeps L y = b, z = y/D, Lᵀ x = z.
//
// Also shown: the compile-out CHECK=true path. Factorizations default to
// CHECK=false and will silently produce NaN/Inf on a non-factorable input;
// with CHECK=true rank 0 reports a zero/NaN pivot via s_fail and the pivot
// sign counts {n_pos, n_neg, n_zero} (the matrix INERTIA) via s_inertia.
// Scratch is sized by glass::ldlt_scratch_bytes<T>(n) (covers the pivot path;
// the non-pivoted path would also accept nullptr).

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static constexpr int N  = 4;   // indefinite 4x4 system
static constexpr int NZ = 2;   // zero-pivot demo matrix

// Factor (checked, non-pivoted) then solve A x = b in place; x lands in b.
__global__ void k_factor_solve(float* A, float* b, int* fail, int* inertia) {
    __shared__ float s_scratch[glass::block::ldlt_scratch_bytes<float>(N) / sizeof(float)];
    glass::block::ldlt<float, N, /*CHECK=*/true>(A, s_scratch, /*pivot=*/false, nullptr, fail, inertia);
    glass::block::ldlt_solve<float, N>(A, b);
}

// CHECK=true on a matrix whose FIRST pivot is exactly zero: D_0 = A_00 = 0.
// The factor itself is garbage (division by zero) — the point is fail == 1.
__global__ void k_factor_zero_pivot(float* A, int* fail) {
    __shared__ float s_scratch[glass::block::ldlt_scratch_bytes<float>(NZ) / sizeof(float)];
    glass::block::ldlt<float, NZ, /*CHECK=*/true>(A, s_scratch, /*pivot=*/false, nullptr, fail, nullptr);
}

int main() {
    // Symmetric INDEFINITE A (column-major == row-major here, it's symmetric).
    // Pivots come out {+, -, +, -} => inertia {2, 2, 0}: not SPD, potrf would NaN.
    float hA[N*N] = { 2, 1, 0, 0,
                      1,-3, 1, 0,
                      0, 1, 4, 1,
                      0, 0, 1,-2 };
    float x_true[N] = { 1.f, -2.f, 3.f, 0.5f };
    float hb[N];
    for (int i = 0; i < N; i++) {            // b = A * x_true (col-major)
        float s = 0; for (int j = 0; j < N; j++) s += hA[i + j*N] * x_true[j];
        hb[i] = s;
    }

    float *dA, *db; int *dfail, *dinertia;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&db, sizeof(hb));
    cudaMalloc(&dfail, sizeof(int)); cudaMalloc(&dinertia, 3*sizeof(int));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(hb), cudaMemcpyHostToDevice);

    k_factor_solve<<<1, 64>>>(dA, db, dfail, dinertia);
    cudaDeviceSynchronize();

    float hx[N]; int fail, inertia[3];
    cudaMemcpy(hx, db, sizeof(hx), cudaMemcpyDeviceToHost);
    cudaMemcpy(&fail, dfail, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(inertia, dinertia, sizeof(inertia), cudaMemcpyDeviceToHost);

    float md = 0; for (int i = 0; i < N; i++) md = fmaxf(md, fabsf(hx[i] - x_true[i]));
    printf("  ldlt + ldlt_solve   x = %.3f %.3f %.3f %.3f  max_err=%.2e  %s\n",
           hx[0], hx[1], hx[2], hx[3], md, md < 1e-4f ? "ok" : "FAIL");
    printf("  CHECK (good A)      fail=%d  inertia={%d,%d,%d} (expect 0, {2,2,0})\n",
           fail, inertia[0], inertia[1], inertia[2]);
    bool ok = (md < 1e-4f) && (fail == 0)
              && inertia[0] == 2 && inertia[1] == 2 && inertia[2] == 0;

    // Zero leading pivot: D_0 = 0 breaks the non-pivoted recurrence. Without
    // CHECK this silently NaNs; with CHECK, s_fail flags it (a caller could
    // then retry with ldlt(pivot=true) or escalate regularization).
    float hZ[NZ*NZ] = { 0, 1,
                        1, 2 };
    float *dZ; cudaMalloc(&dZ, sizeof(hZ));
    cudaMemcpy(dZ, hZ, sizeof(hZ), cudaMemcpyHostToDevice);
    k_factor_zero_pivot<<<1, 64>>>(dZ, dfail);
    cudaDeviceSynchronize();
    cudaMemcpy(&fail, dfail, sizeof(int), cudaMemcpyDeviceToHost);
    printf("  CHECK (zero pivot)  fail=%d (expect 1)\n", fail);
    ok = ok && (fail == 1);

    cudaFree(dA); cudaFree(db); cudaFree(dZ); cudaFree(dfail); cudaFree(dinertia);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
