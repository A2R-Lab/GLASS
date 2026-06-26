// 12_nrm2.cu — Euclidean norm (BLAS nrm2), the standardized name for the old
// `l2norm`.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 12_nrm2.cu -o nrm2 && ./nrm2
//
//   nrm2(x) = sqrt(Σ xᵢ²)
//   NumPy:  np.linalg.norm(x)
//   Eigen:  x.norm()
//
// Block forms are glass::nrm2_fast / glass::nrm2_lowmem (they write the
// result to an output slot); glass::warp::nrm2 returns the value in-register.

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static constexpr int Ncompile = 8;

__global__ void k_block(float* x, float* scratch) {
    // nrm2_fast<T, N> — compile-time length, warp-reduced, DESTRUCTIVE:
    // the result lands in x[0] (the block forms overwrite their input).
    glass::nrm2_fast<float, Ncompile>(x, scratch);
}
__global__ void k_warp(uint32_t n, const float* x, float* out) {
    float r = glass::warp::nrm2<float>(n, x);   // value-returning, non-destructive
    if ((threadIdx.x & 31) == 0) *out = r;
}

int main() {
    float x[Ncompile]; for (int i = 0; i < Ncompile; i++) x[i] = 0.5f*i - 1.3f;
    double s = 0; for (int i = 0; i < Ncompile; i++) s += (double)x[i]*x[i];
    float expected = (float)sqrt(s);

    float *dx, *dout, *dscr; cudaMalloc(&dx, sizeof(x)); cudaMalloc(&dout, sizeof(float)); cudaMalloc(&dscr, sizeof(float)*8);

    float block_r, warp_r;
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
    k_block<<<1, 64>>>(dx, dscr); cudaDeviceSynchronize();
    cudaMemcpy(&block_r, dx, sizeof(float), cudaMemcpyDeviceToHost);   // result in x[0]
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);              // restore (block was destructive)
    k_warp<<<1, 32>>>(Ncompile, dx, dout); cudaDeviceSynchronize();
    cudaMemcpy(&warp_r, dout, sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = fabsf(block_r - expected) < 1e-5 && fabsf(warp_r - expected) < 1e-5;
    printf("  nrm2 block=%.6f  warp=%.6f  expected=%.6f (np.linalg.norm / Eigen x.norm())\n",
           block_r, warp_r, expected);
    cudaFree(dx); cudaFree(dout); cudaFree(dscr);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
