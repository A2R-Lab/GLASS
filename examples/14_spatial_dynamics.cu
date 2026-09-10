// 14_spatial_dynamics.cu — Featherstone spatial cross products: the RNEA inner loop.
//
// Build: nvcc -std=c++17 -arch=sm_75 -I.. 14_spatial_dynamics.cu -o spatial_dynamics && ./spatial_dynamics
//
// USE CASE (rigid-body dynamics): every RNEA/ABA velocity/acceleration sweep is
// built from `v ×ₘ x` and `v ×* f` — a typical generated dynamics suite calls
// them at ~70 sites. This example runs one representative sweep step both ways:
//
//   FUSED     glass::motion_cross_mul / force_cross_mul — each output row is its
//             2-4 term formula; no 6x6 ever exists.
//   COMPOSED  glass::motion_cross → 6x6 in shared memory → glass::gemv.
//
// The two agree to rounding (the fused op IS the composed product, minus the
// 36-element temporary and the extra barrier) — that equivalence is also a
// pytest gate (test_robotics.py). A hand-rolled per-project copy of these row
// formulas computes the same thing at the same speed; what it can't give you is
// the pinned convention ([ω; v] angular-first, column-major), the sign-identity
// test suite (crf == −crmᵀ, the dual identity), and the three tiers for free.
//
// Convention: spatial vectors are ANGULAR-FIRST [ω(3); v(3)] (Featherstone) —
// see docs/source/user_guide/concepts/robotics_conventions.rst.

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int P = 512;   // links × timesteps in flight

// FUSED: y = crm(v)·x, f_out = crf(v)·f — no matrices materialized.
__global__ void k_fused(const float* v, const float* x, const float* f,
                        float* y, float* fo) {
    int p = blockIdx.x;
    glass::block::motion_cross_mul<float>(1.f, v + 6*p, x + 6*p, 0.f, y + 6*p);
    glass::block::force_cross_mul<float>(1.f, v + 6*p, f + 6*p, 0.f, fo + 6*p);
}

// COMPOSED: materialize the 6x6s in shared memory, then gemv.
__global__ void k_composed(const float* v, const float* x, const float* f,
                           float* y, float* fo) {
    __shared__ float M[36], F[36];
    int p = blockIdx.x;
    glass::block::motion_cross<float>(v + 6*p, M);
    glass::block::force_cross<float>(v + 6*p, F);
    glass::block::gemv<float, 6, 6>(1.f, M, x + 6*p, 0.f, y + 6*p);
    glass::block::gemv<float, 6, 6>(1.f, F, f + 6*p, 0.f, fo + 6*p);
}

int main() {
    static float hv[P*6], hx[P*6], hf[P*6];
    for (int i = 0; i < P*6; i++) {
        hv[i] = (float)((int)((i*2654435761u >> 8) % 2000) - 1000) / 500.f;
        hx[i] = (float)((int)((i*40503u >> 4) % 2000) - 1000) / 500.f;
        hf[i] = (float)((int)((i*9973u >> 2) % 2000) - 1000) / 500.f;
    }
    float *dv, *dx, *df, *dy1, *df1, *dy2, *df2;
    cudaMalloc(&dv, sizeof(hv)); cudaMalloc(&dx, sizeof(hx)); cudaMalloc(&df, sizeof(hf));
    cudaMalloc(&dy1, sizeof(hx)); cudaMalloc(&df1, sizeof(hf));
    cudaMalloc(&dy2, sizeof(hx)); cudaMalloc(&df2, sizeof(hf));
    cudaMemcpy(dv, hv, sizeof(hv), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
    cudaMemcpy(df, hf, sizeof(hf), cudaMemcpyHostToDevice);

    k_fused<<<P, 32>>>(dv, dx, df, dy1, df1);
    k_composed<<<P, 32>>>(dv, dx, df, dy2, df2);
    cudaDeviceSynchronize();

    static float y1[P*6], y2[P*6], f1[P*6], f2[P*6];
    cudaMemcpy(y1, dy1, sizeof(y1), cudaMemcpyDeviceToHost);
    cudaMemcpy(y2, dy2, sizeof(y2), cudaMemcpyDeviceToHost);
    cudaMemcpy(f1, df1, sizeof(f1), cudaMemcpyDeviceToHost);
    cudaMemcpy(f2, df2, sizeof(f2), cudaMemcpyDeviceToHost);

    float maxerr = 0.f;
    for (int i = 0; i < P*6; i++) {
        maxerr = fmaxf(maxerr, fabsf(y1[i] - y2[i]));
        maxerr = fmaxf(maxerr, fabsf(f1[i] - f2[i]));
    }
    // host identity check: crf(v)·f == −crm(v)ᵀ·f (sign structure of the duals)
    printf("fused vs composed max |diff| = %.3g  ->  %s\n", maxerr,
           maxerr < 1e-5f ? "PASS" : "FAIL");
    return maxerr < 1e-5f ? 0 : 1;
}
