// 05_gemm_dispatch.cu — glass::gemm_dispatch + dynamic shared memory (tiled path).
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 05_gemm_dispatch.cu -o dispatch && ./dispatch
//
// glass::gemm_dispatch auto-selects the shared-memory-tiled GEMM when scratch
// pointers are supplied (and m*n <= blockDim), else the plain path. The host
// helper glass_gemm_dispatch_smem() computes the bytes to launch with; it
// returns 0 when tiling is not warranted (then pass nullptr scratch).

#include "glass.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// have_smem != 0 => the launch reserved dynamic shared memory for the tiles.
__global__ void dispatch_kernel(float *A, float *B, float *C,
                                int m, int n, int k, int have_smem) {
    extern __shared__ float scratch[];
    // TILE defaults to 8, so the B-tile starts m*8 floats past s_A.
    float *s_A = have_smem ? scratch          : nullptr;
    float *s_B = have_smem ? scratch + m * 8  : nullptr;
    glass::gemm_dispatch(static_cast<uint32_t>(m), static_cast<uint32_t>(n),
                         static_cast<uint32_t>(k), 1.f, A, B, 0.f, C, s_A, s_B);
}

int main() {
    const int m = 4, n = 4, k = 4;
    const int threads = 256;
    float hA[m * n], hB[n * k], hC[m * k] = {0};
    // Column-major: A = identity, B = ramp -> C = B.
    for (int i = 0; i < m * n; ++i) hA[i] = 0.f;
    for (int i = 0; i < m; ++i) hA[i + i * m] = 1.f;      // A = I
    for (int i = 0; i < n * k; ++i) hB[i] = static_cast<float>(i);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    // Host: how many shared bytes does the dispatched path need?
    size_t smem = glass_gemm_dispatch_smem<float>(m, n, threads);
    printf("dispatch smem = %zu bytes (%s)\n", smem,
           smem ? "tiled path" : "plain path");

    dispatch_kernel<<<1, threads, smem>>>(dA, dB, dC, m, n, k, smem > 0 ? 1 : 0);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
    printf("C = I*B -> C[0]=%.0f C[5]=%.0f C[15]=%.0f (expect 0 5 15)\n",
           hC[0], hC[5], hC[15]);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
