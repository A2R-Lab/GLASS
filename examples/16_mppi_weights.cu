// 16_mppi_weights.cu — the path-integral (MPPI) weight update: softmax + argmin.
//
// Build: nvcc -std=c++17 -arch=sm_75 -I.. 16_mppi_weights.cu -o mppi_weights && ./mppi_weights
//
// USE CASE (sampling-based control): an MPPI/CEM controller rolls out N
// perturbed control sequences, scores each with a cost J_i, and blends them
// with the exponentially-weighted average
//     w_i = exp(-λ(J_i - min J)) / Σ_j exp(-λ(J_j - min J)).
// That IS `glass::softmax(n, -λ, J, w, scratch)` — the baseline subtraction is
// the max shift, so the weights are overflow-safe and shift-invariant — plus
// `glass::argmin` for the best-rollout index (warm starts, elite selection).
// One BLOCK owns one controller instance here (the block tier); a batched
// multi-robot controller would drop the same calls one tier down.
//
// Every sampling planner hand-rolls this pair (baseline subtraction, the
// normalizer, the argmin tie rule). The GLASS ops pin the numerics: the
// reductions run fixed-order trees, so the weights are BIT-IDENTICAL at any
// block size — a property this example checks directly (64 vs 256 threads).

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int NROLL = 480;      // rollouts per controller
constexpr float LAMBDA = 3.0f;  // temperature

__global__ void k_weights(const float* J, float* w, unsigned int* best) {
    extern __shared__ float scr[];
    glass::block::softmax<float>(NROLL, -LAMBDA, J, w, scr);
    glass::block::argmin<float>(NROLL, J, best, scr);
}

int main() {
    static float hJ[NROLL];
    for (int i = 0; i < NROLL; i++)
        hJ[i] = 5.f + 3.f*sinf(0.1f*i) + (float)((i*2654435761u >> 7) % 1000)/500.f;
    float *dJ, *dw64, *dw256;
    unsigned int* dbest;
    cudaMalloc(&dJ, sizeof(hJ));
    cudaMalloc(&dw64, sizeof(hJ));
    cudaMalloc(&dw256, sizeof(hJ));
    cudaMalloc(&dbest, sizeof(unsigned int));
    cudaMemcpy(dJ, hJ, sizeof(hJ), cudaMemcpyHostToDevice);

    size_t smem = glass::block::softmax_scratch_bytes<float>(NROLL);
    size_t arg_smem = glass::block::argreduce_scratch_bytes<float>(256);
    if (arg_smem > smem) smem = arg_smem;   // the kernel reuses one buffer for both ops
    k_weights<<<1, 64,  smem>>>(dJ, dw64,  dbest);
    k_weights<<<1, 256, smem>>>(dJ, dw256, dbest);
    cudaDeviceSynchronize();

    static float w64[NROLL], w256[NROLL];
    unsigned int best;
    cudaMemcpy(w64, dw64, sizeof(w64), cudaMemcpyDeviceToHost);
    cudaMemcpy(w256, dw256, sizeof(w256), cudaMemcpyDeviceToHost);
    cudaMemcpy(&best, dbest, sizeof(best), cudaMemcpyDeviceToHost);

    // host reference (double): the exact MPPI weights
    double m = hJ[0];
    for (int i = 1; i < NROLL; i++) m = fmin(m, (double)hJ[i]);
    double Z = 0; int href = 0;
    static double wref[NROLL];
    for (int i = 0; i < NROLL; i++) {
        wref[i] = exp(-LAMBDA*((double)hJ[i] - m));
        Z += wref[i];
        if (hJ[i] < hJ[href]) href = i;
    }
    float sum = 0, maxerr = 0;
    bool bitinv = true;
    for (int i = 0; i < NROLL; i++) {
        sum += w64[i];
        maxerr = fmaxf(maxerr, fabsf(w64[i] - (float)(wref[i]/Z)));
        bitinv = bitinv && (w64[i] == w256[i]);
    }
    printf("Σw = %.6f   max |w - ref| = %.3g   best = %u (ref %d)   "
           "bit-identical 64 vs 256 threads: %s\n",
           sum, maxerr, best, href, bitinv ? "yes" : "NO");
    bool pass = fabsf(sum - 1.f) < 1e-5f && maxerr < 1e-6f
             && best == (unsigned int)href && bitinv;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
