// 19_floating_base_retract.cu — batched SE(3) manifold integration, one state per THREAD.
//
// Build: nvcc -std=c++17 -arch=sm_75 -I.. 19_floating_base_retract.cu -o floating_base_retract && ./floating_base_retract
//
// USE CASE (floating-base dynamics / sampling control): a floating-base
// integrator step is a MANIFOLD update — position+quaternion pose ⊞ body twist
// — not a vector add. Naive `q += ω·dt` drifts off the unit sphere and off
// SO(3); the correct step is `glass::se3_retract` (exp on the group, matching
// Pinocchio's `integrate`). A batched rollout engine holds thousands of
// independent states, so the natural mapping is the THREAD tier: one state per
// thread, 32 states per warp, no barriers.
//
// This example integrates P=4096 rigid-body states through K steps of a
// constant body twist and checks the two invariants a hand-rolled integrator
// classically gets wrong:
//   1. the quaternion stays unit to rounding after K compounding steps
//      (the retract renormalizes; additive updates diverge), and
//   2. integrating a constant twist for K steps equals ONE retract of K·(twist)
//      for the rotation part (one-parameter-subgroup property of exp — a
//      composed-vs-fused gate a Euler-style additive update fails badly).

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int P = 4096;
constexpr int K = 200;
constexpr float DT = 0.01f;

__global__ void k_integrate(const float* pose0, const float* rho, const float* phi,
                            float* pose_out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float pose[7], nxt[7], r[3], w[3];
    for (int i = 0; i < 7; i++) pose[i] = pose0[7*p + i];
    for (int i = 0; i < 3; i++) { r[i] = rho[3*p + i]*DT; w[i] = phi[3*p + i]*DT; }
    for (int k = 0; k < K; k++) {
        glass::thread::se3_retract<float>(pose, r, w, nxt);
        for (int i = 0; i < 7; i++) pose[i] = nxt[i];
    }
    for (int i = 0; i < 7; i++) pose_out[7*p + i] = pose[i];
}

__global__ void k_one_shot(const float* pose0, const float* phi, float* q_out) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    float w[3] = {phi[3*p]*DT*K, phi[3*p + 1]*DT*K, phi[3*p + 2]*DT*K};
    float qn[4];
    glass::thread::quat_retract<float>(pose0 + 7*p + 3, w, qn);
    for (int i = 0; i < 4; i++) q_out[4*p + i] = qn[i];
}

int main() {
    static float h0[P*7], hr[P*3], hw[P*3];
    for (int p = 0; p < P; p++) {
        // random-ish unit quaternion + position
        float q[4], n = 0;
        for (int i = 0; i < 4; i++) { q[i] = (float)((p*31 + i*97) % 200 - 100)/100.f + 0.01f; n += q[i]*q[i]; }
        n = sqrtf(n);
        h0[7*p + 0] = 0.1f*p; h0[7*p + 1] = -0.05f*p; h0[7*p + 2] = 0.5f;
        for (int i = 0; i < 4; i++) h0[7*p + 3 + i] = q[i]/n;
        for (int i = 0; i < 3; i++) {
            hr[3*p + i] = (float)((p + i*7) % 100 - 50)/50.f;
            hw[3*p + i] = (float)((p*3 + i*11) % 100 - 50)/100.f;   // |ω| modest
        }
    }
    float *d0, *dr, *dw, *dout, *dq1;
    cudaMalloc(&d0, sizeof(h0)); cudaMalloc(&dr, sizeof(hr)); cudaMalloc(&dw, sizeof(hw));
    cudaMalloc(&dout, sizeof(h0)); cudaMalloc(&dq1, P*4*sizeof(float));
    cudaMemcpy(d0, h0, sizeof(h0), cudaMemcpyHostToDevice);
    cudaMemcpy(dr, hr, sizeof(hr), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, hw, sizeof(hw), cudaMemcpyHostToDevice);

    k_integrate<<<(P + 255)/256, 256>>>(d0, dr, dw, dout);
    k_one_shot<<<(P + 255)/256, 256>>>(d0, dw, dq1);
    cudaDeviceSynchronize();

    static float out[P*7], q1[P*4];
    cudaMemcpy(out, dout, sizeof(out), cudaMemcpyDeviceToHost);
    cudaMemcpy(q1, dq1, sizeof(q1), cudaMemcpyDeviceToHost);

    float max_norm_err = 0.f, max_sub_err = 0.f;
    for (int p = 0; p < P; p++) {
        float n = 0;
        for (int i = 0; i < 4; i++) n += out[7*p + 3 + i]*out[7*p + 3 + i];
        max_norm_err = fmaxf(max_norm_err, fabsf(sqrtf(n) - 1.f));
        // one-parameter subgroup: K small steps == one K·(ω dt) retract (sign-free)
        float dot = 0;
        for (int i = 0; i < 4; i++) dot += out[7*p + 3 + i]*q1[4*p + i];
        max_sub_err = fmaxf(max_sub_err, fabsf(fabsf(dot) - 1.f));
    }
    printf("unit-norm drift after %d steps: %.3g   subgroup gap: %.3g  ->  %s\n",
           K, max_norm_err, max_sub_err,
           (max_norm_err < 1e-5f && max_sub_err < 1e-4f) ? "PASS" : "FAIL");
    return (max_norm_err < 1e-5f && max_sub_err < 1e-4f) ? 0 : 1;
}
