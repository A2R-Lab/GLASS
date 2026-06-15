// 04_cgrps.cu — cooperative-groups variant of GEMM (glass::cgrps::), pure SIMT.
//
// Build (from this examples/ dir):
//   nvcc -std=c++17 -arch=sm_75 -I.. 04_cgrps.cu -o cgrps && ./cgrps
//
// Same GEMM as 02, but driven through the cooperative-groups namespace, which
// uses g.thread_rank()/g.size() so the op can run on a whole block OR a
// sub-block tile (e.g. a single warp).

#include "glass-cgrps.cuh"
#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>
namespace cgrps = cooperative_groups;

__global__ void cgrps_gemm(float *A, float *B, float *C) {
    // Whole block (default group): C = A*B, 2x2 col-major.
    glass::cgrps::gemm<float, 2, 2, 2>(1.f, A, B, 0.f, C);

    // Warp-level tiling is also supported by passing an explicit group, e.g.:
    //   auto warp = cgrps::tiled_partition<32>(cgrps::this_thread_block());
    //   glass::cgrps::gemm<float, 2, 2, 2>(1.f, A, B, 0.f, C, warp);
}

int main() {
    const int m = 2, k = 2;
    float hA[] = {1, 2, 3, 4};   // col-major A = [[1,3],[2,4]]
    float hB[] = {1, 0, 0, 1};   // identity
    float hC[m * k] = {0};

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(hA));
    cudaMalloc(&dB, sizeof(hB));
    cudaMalloc(&dC, sizeof(hC));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    cgrps_gemm<<<1, 64>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeof(hC), cudaMemcpyDeviceToHost);
    printf("cgrps C = A*I -> [%.0f %.0f; %.0f %.0f]\n",
           hC[0], hC[2], hC[1], hC[3]);   // expect [1 3; 2 4]

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
