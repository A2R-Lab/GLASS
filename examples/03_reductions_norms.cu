// 03_reductions_norms.cu — block reductions and norms: the in-place halving
// `reduce`, the warp-shuffle `reduce_fast` (with its scratch-sizing helper),
// and the nrm2 family across block + warp tiers. (Merged from the former
// 03_reduce / 12_nrm2 examples, 2026-08-11.)
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 03_reductions_norms.cu -o reductions && ./reductions
//
//   reduce(x):   x[0] = Σ xᵢ           (in place, destructive)
//   nrm2(x)  =   sqrt(Σ xᵢ²)           NumPy: np.linalg.norm(x); Eigen: x.norm()
//
// The `_fast` suffix names the warp-shuffle REDUCTION STRATEGY (one scratch
// slot per warp — size with `reduce_fast_scratch_bytes<T>(blockDim)`);
// `glass::warp::nrm2` is the warp-tier form and returns its value in-register.

#include "glass.cuh"
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static constexpr int N = 8;

__global__ void k_reduce(float *x, int n) {
    glass::block::reduce(static_cast<uint32_t>(n), x);           // x[0] = sum(x)
}
__global__ void k_reduce_fast(float *x, int n) {
    extern __shared__ float scratch[];                            // one float per warp
    glass::block::reduce_fast(static_cast<uint32_t>(n), x, scratch);
}
__global__ void k_nrm2_block(float* x, float* scratch) {
    // nrm2_fast<T, N> — compile-time length, warp-reduced, DESTRUCTIVE (x[0] gets the result).
    glass::block::nrm2_fast<float, N>(x, scratch);
}
__global__ void k_nrm2_warp(uint32_t n, const float* x, float* out) {
    float r = glass::warp::nrm2<float>(n, x);   // value-returning, non-destructive
    if ((threadIdx.x & 31) == 0) *out = r;
}

int main() {
    float hx[N];
    for (int i = 0; i < N; ++i) hx[i] = static_cast<float>(i + 1);   // sum = 36
    float xn[N]; for (int i = 0; i < N; i++) xn[i] = 0.5f*i - 1.3f;
    double s = 0; for (int i = 0; i < N; i++) s += (double)xn[i]*xn[i];
    const float nrm_expected = (float)sqrt(s);

    float *dx, *dout, *dscr;
    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dout, sizeof(float));
    cudaMalloc(&dscr, 8 * sizeof(float));
    int bad = 0;
    float out;

    cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
    k_reduce<<<1, 256>>>(dx, N); cudaDeviceSynchronize();
    cudaMemcpy(&out, dx, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  reduce       sum(1..8) = %.0f (expect 36)\n", out);
    bad += (out != 36.f);

    cudaMemcpy(dx, hx, sizeof(hx), cudaMemcpyHostToDevice);
    const int threads = 256;
    size_t smem = glass::reduce_fast_scratch_bytes<float>(threads);
    k_reduce_fast<<<1, threads, smem>>>(dx, N); cudaDeviceSynchronize();
    cudaMemcpy(&out, dx, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  reduce_fast  sum(1..8) = %.0f (expect 36)\n", out);
    bad += (out != 36.f);

    float block_r, warp_r;
    cudaMemcpy(dx, xn, sizeof(xn), cudaMemcpyHostToDevice);
    k_nrm2_block<<<1, 64>>>(dx, dscr); cudaDeviceSynchronize();
    cudaMemcpy(&block_r, dx, sizeof(float), cudaMemcpyDeviceToHost);   // result in x[0]
    cudaMemcpy(dx, xn, sizeof(xn), cudaMemcpyHostToDevice);            // restore (destructive)
    k_nrm2_warp<<<1, 32>>>(N, dx, dout); cudaDeviceSynchronize();
    cudaMemcpy(&warp_r, dout, sizeof(float), cudaMemcpyDeviceToHost);
    printf("  nrm2 block=%.6f  warp=%.6f  expected=%.6f\n", block_r, warp_r, nrm_expected);
    bad += (fabsf(block_r - nrm_expected) >= 1e-5) + (fabsf(warp_r - nrm_expected) >= 1e-5);

    cudaFree(dx); cudaFree(dout); cudaFree(dscr);
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad;
}
