// 12_inv.cu — matrix inversion: the augmented [A | I] convention + the robust
// partial-pivoting variant.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 12_inv.cu -o inv && ./inv
//
// glass::inv is Gauss-Jordan on an AUGMENTED buffer: you hand it a column-major
// N x 2N matrix laid out [A | I] (left half A, right half identity) and on
// return the RIGHT half (columns N..2N-1) holds A⁻¹. Scratch is sized by
// glass::inv_scratch_bytes<T>(N).
//
// The plain inv divides by each leading pivot AS-IS — a zero (or tiny) leading
// pivot produces Inf/NaN even when A is perfectly invertible. glass::inv_pivoted
// (scratch: inv_pivoted_scratch_bytes<T>(N)) row-pivots across the full
// augmented width, so the permutation is absorbed and the right half is still
// A⁻¹ directly. This example shows plain inv going non-finite on a zero leading
// pivot and inv_pivoted recovering the exact inverse.

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static constexpr int N  = 3;   // well-behaved matrix
static constexpr int NP = 2;   // zero-leading-pivot matrix

__global__ void k_inv(float* Aaug) {
    __shared__ float s_scratch[glass::block::inv_scratch_bytes<float>(N) / sizeof(float)];
    glass::block::inv<float, N>(Aaug, s_scratch);
}
__global__ void k_inv_plain_np(float* Aaug) {                 // mishandles pivot 0
    __shared__ float s_scratch[glass::block::inv_scratch_bytes<float>(NP) / sizeof(float)];
    glass::block::inv<float, NP>(Aaug, s_scratch);
}
__global__ void k_inv_pivoted_np(float* Aaug) {               // robust
    __shared__ float s_scratch[glass::block::inv_pivoted_scratch_bytes<float>(NP) / sizeof(float)];
    glass::block::inv_pivoted<float, NP>(Aaug, s_scratch);
}

// Build the column-major dim x 2*dim augmented [A | I] buffer from A.
static void make_augmented(int dim, const float* A, float* aug) {
    for (int c = 0; c < dim; c++)
        for (int r = 0; r < dim; r++) {
            aug[r + c*dim]         = A[r + c*dim];                        // left: A
            aug[r + (dim + c)*dim] = (r == c) ? 1.0f : 0.0f;              // right: I
        }
}

// max |(A * Ainv - I)[i,j]| (all column-major dim x dim).
static float inverse_residual(int dim, const float* A, const float* Ainv) {
    float md = 0;
    for (int c = 0; c < dim; c++)
        for (int r = 0; r < dim; r++) {
            float s = 0; for (int k = 0; k < dim; k++) s += A[r + k*dim] * Ainv[k + c*dim];
            md = fmaxf(md, fabsf(s - (r == c ? 1.0f : 0.0f)));
        }
    return md;
}

int main() {
    // ── glass::inv on a well-behaved 3x3 (column-major) ──
    float A[N*N] = { 4, 1, 2,      // col 0
                     1, 3, 0,      // col 1
                     2, 0, 5 };    // col 2
    float aug[N*2*N]; make_augmented(N, A, aug);

    float* dAug; cudaMalloc(&dAug, sizeof(aug));
    cudaMemcpy(dAug, aug, sizeof(aug), cudaMemcpyHostToDevice);
    k_inv<<<1, 64>>>(dAug);
    cudaDeviceSynchronize();
    cudaMemcpy(aug, dAug, sizeof(aug), cudaMemcpyDeviceToHost);

    const float* Ainv = &aug[N*N];                    // right half holds A⁻¹
    float res = inverse_residual(N, A, Ainv);
    printf("  inv (3x3)              ||A·A⁻¹ - I||_max = %.2e  %s\n",
           res, res < 1e-5f ? "ok" : "FAIL");
    bool ok = (res < 1e-5f);

    // ── zero LEADING pivot: A invertible (det = -2) but A[0,0] = 0 ──
    float Z[NP*NP] = { 0, 2,       // col 0
                       1, 3 };     // col 1
    float augZ[NP*2*NP];

    // plain inv: divides by A[0,0] = 0 -> Inf/NaN contaminate the result.
    make_augmented(NP, Z, augZ);
    float* dAugZ; cudaMalloc(&dAugZ, sizeof(augZ));
    cudaMemcpy(dAugZ, augZ, sizeof(augZ), cudaMemcpyHostToDevice);
    k_inv_plain_np<<<1, 64>>>(dAugZ);
    cudaDeviceSynchronize();
    cudaMemcpy(augZ, dAugZ, sizeof(augZ), cudaMemcpyDeviceToHost);
    bool nonfinite = false;
    for (int i = 0; i < NP*NP; i++) nonfinite |= !isfinite(augZ[NP*NP + i]);
    printf("  inv (zero pivot)       non-finite result: %s (expected — plain inv mishandles it)\n",
           nonfinite ? "yes" : "no");
    ok = ok && nonfinite;

    // inv_pivoted: row-swaps the largest |pivot| up first -> exact inverse.
    make_augmented(NP, Z, augZ);
    cudaMemcpy(dAugZ, augZ, sizeof(augZ), cudaMemcpyHostToDevice);
    k_inv_pivoted_np<<<1, 64>>>(dAugZ);
    cudaDeviceSynchronize();
    cudaMemcpy(augZ, dAugZ, sizeof(augZ), cudaMemcpyDeviceToHost);
    res = inverse_residual(NP, Z, &augZ[NP*NP]);
    printf("  inv_pivoted (same A)   ||A·A⁻¹ - I||_max = %.2e  %s\n",
           res, res < 1e-5f ? "ok" : "FAIL");
    ok = ok && (res < 1e-5f);

    cudaFree(dAug); cudaFree(dAugZ);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
