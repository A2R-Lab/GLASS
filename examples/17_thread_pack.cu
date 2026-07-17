// 17_thread_pack.cu — the glass::thread:: tier: 32 low-DOF SPD solves packed per warp.
//
// Build (from this examples/ dir, pure SIMT — no MathDx needed):
//   nvcc -std=c++17 -arch=sm_75 -I.. 17_thread_pack.cu -o thread_pack && ./thread_pack
//
// The low-DOF corner: at N=6 a warp-per-problem factor leaves ~26 of 32 lanes idle
// on the serial pivot steps. glass::thread:: flips the mapping — ONE problem per
// THREAD, sequential (no barriers, no shuffles, no threadIdx read inside the op),
// so a warp carries 32 independent problems at once. The price: compile-time sizes
// only, and the operands must stay register-resident (measured ceiling N<=7 —
// past it the thread-local arrays spill and the tier's premise is gone).
//
// The kernel is the caller's: it stages each problem global -> thread-local
// registers, runs the op on its own arrays, and writes back. Launch shape comes
// from glass::suggested_threads_per_block<>() (a seed heuristic, not a measured
// table entry — see glass-defaults.cuh).

#include "glass.cuh"
#include "glass-defaults.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

constexpr int N = 6;        // per-problem size: a 6-DOF arm's normal equations
constexpr int P = 4096;     // independent problems

__global__ void k_thread_posv(const float* A, const float* rhs, float* x, int np) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= np) return;                       // ragged tail: fine — no barriers inside
    float a[N * N], b[N];                      // register-resident at N<=7
    for (int i = 0; i < N * N; i++) a[i] = A[(size_t)p * N * N + i];
    for (int i = 0; i < N; i++)     b[i] = rhs[(size_t)p * N + i];
    glass::thread::posv<float, N>(a, b);       // factor + fwd/back solve, one thread
    for (int i = 0; i < N; i++)     x[(size_t)p * N + i] = b[i];
}

int main() {
    // Host: build P well-conditioned SPD systems (A = M Mᵀ + N·I, column-major).
    static float hA[P * N * N], hb[P * N], hx[P * N];
    for (int p = 0; p < P; p++) {
        float M[N * N];
        for (int i = 0; i < N * N; i++) M[i] = (float)(((p * 131 + i * 7919) % 200) - 100) / 100.f;
        for (int c = 0; c < N; c++)
            for (int r = 0; r < N; r++) {
                float s = 0.f;
                for (int k = 0; k < N; k++) s += M[r + k * N] * M[c + k * N];
                hA[p * N * N + r + c * N] = s + (r == c ? (float)N : 0.f);
            }
        for (int i = 0; i < N; i++) hb[p * N + i] = (float)((p + i) % 5) - 2.f;
    }

    float *dA, *db, *dx;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&db, sizeof(hb)); cudaMalloc(&dx, sizeof(hb));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(hb), cudaMemcpyHostToDevice);

    // One problem per thread; TPB from the defaults heuristic (N=6 -> 64).
    constexpr uint32_t TPB = glass::suggested_threads_per_block<glass::op::posv, N, float>();
    k_thread_posv<<<(P + TPB - 1) / TPB, TPB>>>(dA, db, dx, P);
    cudaMemcpy(hx, dx, sizeof(hx), cudaMemcpyDeviceToHost);

    // Verify every problem: ||A x - b||_inf against the untouched host copies.
    float worst = 0.f;
    for (int p = 0; p < P; p++) {
        for (int r = 0; r < N; r++) {
            float s = 0.f;
            for (int c = 0; c < N; c++) s += hA[p * N * N + r + c * N] * hx[p * N + c];
            worst = fmaxf(worst, fabsf(s - hb[p * N + r]));
        }
    }
    printf("thread-packed posv: %d problems of N=%d at TPB=%u, worst |Ax-b| = %.2e -> %s\n",
           P, N, TPB, worst, worst < 1e-3f ? "PASS" : "FAIL");
    cudaFree(dA); cudaFree(db); cudaFree(dx);
    return worst < 1e-3f ? 0 : 1;
}
