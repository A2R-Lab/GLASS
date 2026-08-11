// 11_riccati_gain.cu — LQR feedback gain K = (R + BᵀPB)⁻¹ (BᵀPA) in one call.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 11_riccati_gain.cu -o riccati && ./riccati
//
// The control-update solve at the heart of an LQR / iLQR backward pass.
// glass::riccati_gain composes three library primitives in one block:
//   S = R + BᵀPB   (congruence_sym, NU×NU control Hessian)
//   G = BᵀPA       (bilinear,      NU×NX coupling)
//   S·K = G        (checked multi-RHS Cholesky posv; K overwrites G)
// Inputs P, A, B, R are unchanged; Kgain holds K (NU×NX, column-major).
// Scratch is dynamic shared memory sized by the host-callable
// glass::riccati_scratch_bytes<T, NX, NU>() (BYTES, pass as the launch smem).
// s_fail reports a non-PD S (an iLQR caller would escalate rho and retry via
// the REGULARIZE template flag; see 10_ldlt_solve.cu for the CHECK idea).

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static constexpr int NX = 4, NU = 2;

__global__ void k_riccati(const float* P, const float* A, const float* B,
                          const float* R, float* Kgain, int* fail) {
    extern __shared__ float s_scratch[];
    glass::block::riccati_gain<float, NX, NU>(P, A, B, R, Kgain, s_scratch, 0.f, fail);
}

// CPU reference with plain loops (everything column-major, X[row + col*rows]).
static void ref_gain(const float* P, const float* A, const float* B,
                     const float* R, float* K) {
    float PB[NX*NU], PA[NX*NX], S[NU*NU], G[NU*NX];
    for (int c = 0; c < NU; c++) for (int r = 0; r < NX; r++) {          // PB = P·B
        float s = 0; for (int k = 0; k < NX; k++) s += P[r + k*NX] * B[k + c*NX];
        PB[r + c*NX] = s;
    }
    for (int c = 0; c < NX; c++) for (int r = 0; r < NX; r++) {          // PA = P·A
        float s = 0; for (int k = 0; k < NX; k++) s += P[r + k*NX] * A[k + c*NX];
        PA[r + c*NX] = s;
    }
    for (int c = 0; c < NU; c++) for (int r = 0; r < NU; r++) {          // S = R + Bᵀ·PB
        float s = 0; for (int k = 0; k < NX; k++) s += B[k + r*NX] * PB[k + c*NX];
        S[r + c*NU] = R[r + c*NU] + s;
    }
    for (int c = 0; c < NX; c++) for (int r = 0; r < NU; r++) {          // G = Bᵀ·PA
        float s = 0; for (int k = 0; k < NX; k++) s += B[k + r*NX] * PA[k + c*NX];
        G[r + c*NU] = s;
    }
    // Solve S·K = G (S is SPD NU×NU, NX right-hand sides): Gaussian elimination.
    for (int p = 0; p < NU; p++) {
        for (int r = p + 1; r < NU; r++) {
            float m = S[r + p*NU] / S[p + p*NU];
            for (int c = p; c < NU; c++) S[r + c*NU] -= m * S[p + c*NU];
            for (int c = 0; c < NX; c++) G[r + c*NU] -= m * G[p + c*NU];
        }
    }
    for (int c = 0; c < NX; c++)
        for (int r = NU - 1; r >= 0; r--) {
            float s = G[r + c*NU];
            for (int k = r + 1; k < NU; k++) s -= S[r + k*NU] * K[k + c*NU];
            K[r + c*NU] = s / S[r + r*NU];
        }
}

int main() {
    // P: symmetric PD cost-to-go; A: state Jacobian; B: control Jacobian; R: SPD.
    float P[NX*NX], A[NX*NX], B[NX*NU];
    for (int i = 0; i < NX; i++) for (int j = 0; j < NX; j++) {
        P[i + j*NX] = 0.1f*(i + j) + (i == j ? 2.0f + i : 0.0f);         // symmetric PD
        A[i + j*NX] = 0.1f*i - 0.05f*j + (i == j ? 1.0f : 0.0f);
    }
    for (int i = 0; i < NX*NU; i++) B[i] = 0.2f*i - 0.3f;
    float R[NU*NU] = { 1.0f, 0.2f,
                       0.2f, 0.8f };

    float ref_K[NU*NX]; ref_gain(P, A, B, R, ref_K);

    float *dP, *dA, *dB, *dR, *dK; int *dfail;
    cudaMalloc(&dP, sizeof(P)); cudaMalloc(&dA, sizeof(A)); cudaMalloc(&dB, sizeof(B));
    cudaMalloc(&dR, sizeof(R)); cudaMalloc(&dK, sizeof(float)*NU*NX); cudaMalloc(&dfail, sizeof(int));
    cudaMemcpy(dP, P, sizeof(P), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(dR, R, sizeof(R), cudaMemcpyHostToDevice);

    const size_t smem = glass::block::riccati_scratch_bytes<float, NX, NU>();   // BYTES
    k_riccati<<<1, 128, smem>>>(dP, dA, dB, dR, dK, dfail);
    cudaDeviceSynchronize();

    float K[NU*NX]; int fail;
    cudaMemcpy(K, dK, sizeof(K), cudaMemcpyDeviceToHost);
    cudaMemcpy(&fail, dfail, sizeof(int), cudaMemcpyDeviceToHost);

    float md = 0; for (int i = 0; i < NU*NX; i++) md = fmaxf(md, fabsf(K[i] - ref_K[i]));
    printf("  riccati_gain (NX=%d, NU=%d, smem=%zu B)  fail=%d  max_err vs CPU = %.2e\n",
           NX, NU, smem, fail, md);
    for (int r = 0; r < NU; r++)
        printf("    K[%d,:] = %8.4f %8.4f %8.4f %8.4f\n",
               r, K[r + 0*NU], K[r + 1*NU], K[r + 2*NU], K[r + 3*NU]);

    bool ok = (md < 1e-4f) && (fail == 0);
    cudaFree(dP); cudaFree(dA); cudaFree(dB); cudaFree(dR); cudaFree(dK); cudaFree(dfail);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
