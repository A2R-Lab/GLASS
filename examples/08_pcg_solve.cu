// 08_pcg_solve.cu — block-tridiagonal PCG solve (glass::pcg), pure SIMT.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 08_pcg_solve.cu -o pcg && ./pcg
//
// Solves an SPD block-tridiagonal system S x = b in ONE CUDA block with
// preconditioned conjugate gradient, using a block-Jacobi preconditioner Pinv
// (inverse of each diagonal block). The matrix is stored as KP row-major
// [L | D | R] strips and the vectors use the padded (KP+2)*SS layout. See
// docs/source/user_guide/concepts/block_tridiagonal.rst.

#include "glass.cuh"
#include <cstdio>
#include <cuda_runtime.h>

constexpr int SS  = 2;                 // state_size (block dimension)
constexpr int KP  = 3;                 // knot_points (number of block-rows)
constexpr int BRL = 3 * SS;            // columns per strip: [L|D|R]
constexpr int N   = SS * KP;           // unpadded system size
constexpr int VEC = (KP + 2) * SS;     // padded vector length

__global__ void pcg_kernel(float *x, float *S, float *Pinv, float *b,
                           unsigned max_iters, float rel_tol, float abs_tol,
                           unsigned *iters) {
    extern __shared__ float s_mem[];
    glass::pcg<float, SS, KP>(x, S, Pinv, b, s_mem,
                                     max_iters, rel_tol, abs_tol, iters);
}

// Write a 2x2 block into strip k at column offset `coloff` (0=L, SS=D, 2SS=R).
static void put2x2(float *M, int k, int coloff,
                   float a, float b, float c, float d) {
    float *strip = M + k * BRL * SS;   // SS rows x BRL cols, row-major
    strip[0 * BRL + coloff + 0] = a; strip[0 * BRL + coloff + 1] = b;
    strip[1 * BRL + coloff + 0] = c; strip[1 * BRL + coloff + 1] = d;
}

int main() {
    // Diagonally-dominant SPD blocks: D = [[4,1],[1,4]], off-diagonals 0.1*I.
    float S[KP * BRL * SS]    = {0};
    float Pinv[KP * BRL * SS] = {0};
    for (int k = 0; k < KP; ++k) {
        put2x2(S, k, SS, 4, 1, 1, 4);                       // D
        if (k > 0)      put2x2(S, k, 0,      0.1f, 0, 0, 0.1f);   // L
        if (k < KP - 1) put2x2(S, k, 2 * SS, 0.1f, 0, 0, 0.1f);  // R
        // Block-Jacobi preconditioner: inv(D) = (1/15)[[4,-1],[-1,4]].
        put2x2(Pinv, k, SS, 4.f / 15, -1.f / 15, -1.f / 15, 4.f / 15);
    }

    // Padded RHS (b = 1 on the real entries, zeros in the pads) and zero guess.
    float b[VEC] = {0}, x[VEC] = {0};
    for (int i = 0; i < N; ++i) b[SS + i] = 1.0f;

    float *dS, *dPinv, *db, *dx; unsigned *diters;
    cudaMalloc(&dS, sizeof(S));      cudaMalloc(&dPinv, sizeof(Pinv));
    cudaMalloc(&db, sizeof(b));      cudaMalloc(&dx, sizeof(x));
    cudaMalloc(&diters, sizeof(unsigned));
    cudaMemcpy(dS, S, sizeof(S), cudaMemcpyHostToDevice);
    cudaMemcpy(dPinv, Pinv, sizeof(Pinv), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);

    const int threads = 32;            // must be a multiple of 32 (warp-dot)
    size_t smem = glass::pcg_scratch_bytes<float, SS, KP>(threads);
    pcg_kernel<<<1, threads, smem>>>(dx, dS, dPinv, db, 100, 1e-6f, 1e-12f, diters);
    cudaDeviceSynchronize();

    unsigned iters;
    cudaMemcpy(x, dx, sizeof(x), cudaMemcpyDeviceToHost);
    cudaMemcpy(&iters, diters, sizeof(unsigned), cudaMemcpyDeviceToHost);

    printf("PCG converged in %u iters; x =", iters);
    for (int i = 0; i < N; ++i) printf(" %.4f", x[SS + i]);   // strip the pads
    printf("\n");

    cudaFree(dS); cudaFree(dPinv); cudaFree(db); cudaFree(dx); cudaFree(diters);
    return 0;
}
