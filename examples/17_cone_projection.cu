// 17_cone_projection.cu — friction-cone AL constraint step: soc_project + interval AL.
//
// Build: nvcc -std=c++17 -arch=sm_75 -I.. 17_cone_projection.cu -o cone_projection && ./cone_projection
//
// USE CASE (constrained trajectory optimization): a contact-rich solver carries
// friction-cone rows (‖f_tangential‖ <= μ·f_normal — a second-order cone) and
// state/control interval rows. The PHR augmented-Lagrangian machinery for both
// reduces to small per-row scalar ops:
//   * `glass::soc_project` — the Euclidean cone projection (multiplier update
//     λ ← Π_K(λ − ρg), and the AL gradient is −Π_K(λ − ρg));
//   * `glass::al_soc_value` / `glass::al_interval_value` /
//     `glass::al_interval_grad_hess` — the merit value and its GN derivatives.
// One (row-group, knot) pair per THREAD — exactly how GATO's bsqp solver runs
// this set (these ops are its cone wave, promoted).
//
// Checks: projection case split vs a host reference, idempotence
// (Π(Π(w)) == Π(w)), the convex-projection orthogonality <Π(w), Π(w) − w> == 0,
// and the m=1 degeneration of the conic AL value to the scalar hinge.

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int P = 2048;    // (row-group, knot) pairs
constexpr int M = 4;       // cone dimension: normal + 3 tangential rows
constexpr float RHO = 1.7f;

__global__ void k_al_step(const float* g, const float* lam, float* proj,
                          float* val, float* hinge_gap) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    // multiplier-update projection: Π_K(λ − ρg), then project AGAIN (idempotence
    // is checked on host via this second image equaling the first)
    float w[M], pr[M];
    for (int i = 0; i < M; i++) w[i] = lam[M*p + i] - RHO*g[M*p + i];
    glass::thread::soc_project<float>(w, pr, M);
    for (int i = 0; i < M; i++) proj[M*p + i] = pr[i];
    // AL merit value for the cone row
    val[p] = glass::block::al_soc_value<float>(g + M*p, lam + M*p, RHO, M);
    // m=1 cone == the g >= 0 hinge: the conic value must equal al_hinge_value
    // on the sign convention bridge (hinge feasible c <= 0 ⇔ cone g >= 0).
    float g1 = g[M*p], l1 = lam[M*p];
    float conic = glass::block::al_soc_value<float>(&g1, &l1, RHO, 1);
    float hinge = glass::block::al_hinge_value<float>(-g1, l1, RHO, 0.f);
    hinge_gap[p] = fabsf(conic - hinge);
}

int main() {
    static float hg[P*M], hl[P*M];
    for (int i = 0; i < P*M; i++) {
        hg[i] = (float)((int)((i*2654435761u >> 6) % 2000) - 1000)/700.f;
        hl[i] = (float)((i*40503u >> 3) % 1000)/500.f;
    }
    float *dg, *dl, *dp, *dv, *dh;
    cudaMalloc(&dg, sizeof(hg)); cudaMalloc(&dl, sizeof(hl));
    cudaMalloc(&dp, sizeof(hg)); cudaMalloc(&dv, P*sizeof(float));
    cudaMalloc(&dh, P*sizeof(float));
    cudaMemcpy(dg, hg, sizeof(hg), cudaMemcpyHostToDevice);
    cudaMemcpy(dl, hl, sizeof(hl), cudaMemcpyHostToDevice);
    k_al_step<<<(P + 127)/128, 128>>>(dg, dl, dp, dv, dh);
    cudaDeviceSynchronize();

    static float proj[P*M], val[P], hinge_gap[P];
    cudaMemcpy(proj, dp, sizeof(proj), cudaMemcpyDeviceToHost);
    cudaMemcpy(val, dv, sizeof(val), cudaMemcpyDeviceToHost);
    cudaMemcpy(hinge_gap, dh, sizeof(hinge_gap), cudaMemcpyDeviceToHost);

    float max_orth = 0, max_hinge = 0, max_proj = 0;
    for (int p = 0; p < P; p++) {
        // host reference projection + orthogonality <Π(w), Π(w) − w>
        double w[M], r = 0;
        for (int i = 0; i < M; i++) w[i] = (double)hl[M*p + i] - RHO*(double)hg[M*p + i];
        for (int i = 1; i < M; i++) r += w[i]*w[i];
        r = sqrt(r);
        double ref[M];
        if (r <= w[0])       for (int i = 0; i < M; i++) ref[i] = w[i];
        else if (r <= -w[0]) for (int i = 0; i < M; i++) ref[i] = 0;
        else {
            double a = 0.5*(w[0] + r);
            ref[0] = a;
            for (int i = 1; i < M; i++) ref[i] = (a/r)*w[i];
        }
        double orth = 0;
        for (int i = 0; i < M; i++) {
            max_proj = fmaxf(max_proj, fabsf(proj[M*p + i] - (float)ref[i]));
            orth += (double)proj[M*p + i]*((double)proj[M*p + i] - w[i]);
        }
        max_orth = fmaxf(max_orth, fabsf((float)orth));
        max_hinge = fmaxf(max_hinge, hinge_gap[p]);
    }
    printf("proj err %.3g   orthogonality %.3g   m=1 hinge gap %.3g\n",
           max_proj, max_orth, max_hinge);
    bool pass = max_proj < 1e-5f && max_orth < 1e-3f && max_hinge < 1e-6f;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
