#include <glass/glass.cuh>

__global__ void install_canary(float* a, float* b, float* c) {
    glass::gemm<float, 4, 4, 4>(1.0f, a, b, 0.0f, c);
}
