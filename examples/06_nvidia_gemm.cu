// 06_nvidia_gemm.cu — cuBLASDx-backed GEMM via glass::nvidia::  (REQUIRES MathDx).
//
// This is the ONLY example that needs NVIDIA MathDx (cuBLASDx). The pure-SIMT
// examples 01-05 build with plain nvcc; this one does not.
//
// Build (from this examples/ dir), with MATHDX_ROOT pointing at your MathDx
// install (see ../bench/INSTALL.md):
//
//   nvcc -std=c++17 -arch=sm_86 -I.. \
//        -DGLASS_BENCH_CUBLASDX -DSMS=860 \
//        --expt-relaxed-constexpr -Xptxas -O1 \
//        -I$MATHDX_ROOT/include \
//        -I$MATHDX_ROOT/external/cutlass/include \
//        06_nvidia_gemm.cu -o nvidia_gemm && ./nvidia_gemm
//
// Notes:
//   * -DGLASS_BENCH_CUBLASDX force-includes <cublasdx.hpp> from glass-nvidia.cuh.
//   * -DSMS=XXX must match your -arch (860 for sm_86, 1200 for sm_120, ...);
//     it selects the cuBLASDx-tuned config and the pre-instantiated GEMM table.
//   * 16x16x16 is a pre-instantiated cuBLASDx shape (see glass-nvidia.cuh); the
//     default form launches with EXACTLY gemm_threads<>() threads and
//     gemm_scratch_bytes<>() bytes of shared memory — a mismatch deadlocks.

#include "glass-nvidia.cuh"
#include <cstdio>
#include <cuda_runtime.h>

constexpr int M = 16, N = 16, K = 16;

// Host-queryable, constexpr: the thread count + shared bytes cuBLASDx wants.
constexpr auto SMEM    = glass::nvidia::gemm_scratch_bytes<float, M, N, K>();
constexpr auto THREADS = glass::nvidia::gemm_threads<float, M, N, K>();

__global__ void nvidia_gemm(float *A, float *B, float *C) {
    extern __shared__ __align__(16) char smem_buf[];
    glass::nvidia::gemm<float, M, N, K>(1.f, A, B, 0.f, C, smem_buf);
}

int main() {
    float hA[M * N], hB[N * K], hC[M * K] = {0};
    // Column-major: A = identity, B = ramp -> C = B.
    for (int i = 0; i < M * N; ++i) hA[i] = 0.f;
    for (int i = 0; i < M; ++i) hA[i + i * M] = 1.f;      // A = I
    for (int i = 0; i < N * K; ++i) hB[i] = static_cast<float>(i);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    // Launch with the EXACT thread count + smem cuBLASDx picked.
    nvidia_gemm<<<1, THREADS, SMEM>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
    printf("glass::nvidia::gemm  C = I*B  (threads=%u smem=%zu)\n",
           (unsigned)THREADS, (size_t)SMEM);
    printf("C[0]=%.0f C[17]=%.0f C[255]=%.0f (expect 0 17 255)\n",
           hC[0], hC[17], hC[255]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
