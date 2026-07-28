// 23_best_fit_rotation.cu — batched point alignment + rotation cleanup (est kit).
//
// Build: nvcc -std=c++17 -arch=sm_75 -I.. 23_best_fit_rotation.cu -o best_fit_rotation && ./best_fit_rotation
//
// USE CASE (estimation / registration): the inner op of ICP and point-cloud
// alignment is Wahba's problem — given correspondences (a_i, b_i), find the
// rotation minimizing Σ‖b_i − R·a_i‖². The GPU-batched recipe, one problem
// per THREAD (thousands of independent alignments in one launch):
//   1. accumulate the 3x3 cross covariance M = Σ b_i·a_iᵀ (nine FMAs per
//      correspondence — glass::ger shaped, done inline here),
//   2. glass::thread::closest_rotation(M, R) — the SVD-based Kabsch solution
//      with the det fix (never returns a reflection).
// The SAME op is the rotation-matrix re-orthonormalizer: feed a drifted
// product of many incremental rotations and it returns the nearest proper
// rotation — the classic cleanup after long integrations.
//
// Checks (self-verifying): recovered rotations match the known ground truth
// to f32 tolerance for every problem; re-orthonormalized matrices are
// orthonormal with det +1 and stay near the drifted input.

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int P  = 4096;   // independent alignment problems
constexpr int NC = 8;      // correspondences per problem

__global__ void k_align(const float* a, const float* b, float* R) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    // M = Σ b_i·a_iᵀ (column-major: M[c*3 + r] += b[r]·a[c])
    float M[9];
    for (int i = 0; i < 9; i++) M[i] = 0.f;
    for (int i = 0; i < NC; i++) {
        const float* ai = a + (size_t)p*NC*3 + 3*i;
        const float* bi = b + (size_t)p*NC*3 + 3*i;
        for (int c = 0; c < 3; c++)
            for (int r = 0; r < 3; r++)
                M[c*3 + r] += bi[r] * ai[c];
    }
    glass::thread::closest_rotation<float>(M, R + (size_t)p*9);
}

__global__ void k_cleanup(const float* A, float* R) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    glass::thread::closest_rotation<float>(A + (size_t)p*9, R + (size_t)p*9);
}

// host-side rotation from an axis/angle (double, column-major)
static void rot_from_aa(const double* ax, double th, double* R) {
    double n = sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2]);
    double u[3] = {ax[0]/n, ax[1]/n, ax[2]/n};
    double c = cos(th), s = sin(th), oc = 1.0 - c;
    R[0] = c + u[0]*u[0]*oc;      R[3] = u[0]*u[1]*oc - u[2]*s;  R[6] = u[0]*u[2]*oc + u[1]*s;
    R[1] = u[1]*u[0]*oc + u[2]*s; R[4] = c + u[1]*u[1]*oc;       R[7] = u[1]*u[2]*oc - u[0]*s;
    R[2] = u[2]*u[0]*oc - u[1]*s; R[5] = u[2]*u[1]*oc + u[0]*s;  R[8] = c + u[2]*u[2]*oc;
}

int main() {
    static double Rtrue[P][9];
    float *ha = (float*)malloc((size_t)P*NC*3*sizeof(float));
    float *hb = (float*)malloc((size_t)P*NC*3*sizeof(float));
    float *hA = (float*)malloc((size_t)P*9*sizeof(float));
    unsigned s = 12345u;
    auto frand = [&]() { s = s*1664525u + 1013904223u; return ((s >> 8) / 8388608.0) * 2.0 - 1.0; };
    for (int p = 0; p < P; p++) {
        double ax[3] = {frand(), frand(), frand()};
        rot_from_aa(ax, 0.1 + 2.8 * ((frand() + 1.0) / 2.0), Rtrue[p]);
        for (int i = 0; i < NC; i++) {
            double ai[3] = {frand(), frand(), frand()};
            for (int k = 0; k < 3; k++) ha[(size_t)p*NC*3 + 3*i + k] = (float)ai[k];
            for (int r = 0; r < 3; r++)   // b = R_true·a (exact correspondences)
                hb[(size_t)p*NC*3 + 3*i + r] =
                    (float)(Rtrue[p][r]*ai[0] + Rtrue[p][3+r]*ai[1] + Rtrue[p][6+r]*ai[2]);
        }
        for (int k = 0; k < 9; k++)       // drifted rotation for the cleanup leg
            hA[(size_t)p*9 + k] = (float)(Rtrue[p][k] + 5e-3 * frand());
    }
    float *da, *db, *dA, *dR1, *dR2;
    cudaMalloc(&da, (size_t)P*NC*3*sizeof(float)); cudaMalloc(&db, (size_t)P*NC*3*sizeof(float));
    cudaMalloc(&dA, (size_t)P*9*sizeof(float));
    cudaMalloc(&dR1, (size_t)P*9*sizeof(float)); cudaMalloc(&dR2, (size_t)P*9*sizeof(float));
    cudaMemcpy(da, ha, (size_t)P*NC*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, (size_t)P*NC*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dA, hA, (size_t)P*9*sizeof(float), cudaMemcpyHostToDevice);

    k_align<<<(P + 127)/128, 128>>>(da, db, dR1);
    k_cleanup<<<(P + 127)/128, 128>>>(dA, dR2);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("FAIL: kernel error\n"); return 1; }

    float *hR1 = (float*)malloc((size_t)P*9*sizeof(float));
    float *hR2 = (float*)malloc((size_t)P*9*sizeof(float));
    cudaMemcpy(hR1, dR1, (size_t)P*9*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hR2, dR2, (size_t)P*9*sizeof(float), cudaMemcpyDeviceToHost);

    double worst_fit = 0.0, worst_orth = 0.0, worst_det = 0.0, worst_drift = 0.0;
    for (int p = 0; p < P; p++) {
        for (int k = 0; k < 9; k++) {
            double d = fabs((double)hR1[(size_t)p*9 + k] - Rtrue[p][k]);
            if (d > worst_fit) worst_fit = d;
            double dd = fabs((double)hR2[(size_t)p*9 + k] - Rtrue[p][k]);
            if (dd > worst_drift) worst_drift = dd;
        }
        const float* R = hR2 + (size_t)p*9;   // orthonormality + det of the cleanup
        double det = (double)R[0]*((double)R[4]*R[8] - (double)R[5]*R[7])
                   - (double)R[3]*((double)R[1]*R[8] - (double)R[2]*R[7])
                   + (double)R[6]*((double)R[1]*R[5] - (double)R[2]*R[4]);
        if (fabs(det - 1.0) > worst_det) worst_det = fabs(det - 1.0);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double dot = 0.0;
                for (int k = 0; k < 3; k++) dot += (double)R[i*3 + k]*R[j*3 + k];
                double e = fabs(dot - (i == j ? 1.0 : 0.0));
                if (e > worst_orth) worst_orth = e;
            }
    }
    printf("align: worst |R - R_true|     = %.3e  (exact correspondences)\n", worst_fit);
    printf("clean: worst orthonormality   = %.3e, worst |det-1| = %.3e\n", worst_orth, worst_det);
    printf("clean: worst |R - R_true|     = %.3e  (5e-3 drift input)\n", worst_drift);
    bool ok = worst_fit < 2e-4 && worst_orth < 1e-5 && worst_det < 1e-5 && worst_drift < 2e-2;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
