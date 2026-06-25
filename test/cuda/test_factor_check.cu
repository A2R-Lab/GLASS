// test_factor_check.cu — exercise the CHECK flag on glass::cholDecomp_InPlace
// (block / warp / cgrps) and glass::ldlt (block, + inertia). Prints a scalar
// line then the factored matrix.
//
// Usage:
//   cholchk <block|warp|cgrps> <THREADS> <N> <A.bin>   -> "<fail>\n<L (N*N)>"
//   ldltchk <THREADS> <N> <A.bin>                      -> "<fail> <npos> <nneg> <nzero>\n<LD (N*N)>"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "helpers.cuh"
#include "../../glass-cgrps.cuh"

__global__ void k_cholchk_block(int n, float* A, int* fail) {
    extern __shared__ float s[];
    for (int i = threadIdx.x; i < n*n; i += blockDim.x) s[i] = A[i];
    __syncthreads();
    glass::cholDecomp_InPlace<float, true>((uint32_t)n, s, fail);
    __syncthreads();
    for (int i = threadIdx.x; i < n*n; i += blockDim.x) A[i] = s[i];
}

__global__ void k_cholchk_cgrps(int n, float* A, int* fail) {
    extern __shared__ float s[];
    auto g = cooperative_groups::this_thread_block();
    for (int i = threadIdx.x; i < n*n; i += blockDim.x) s[i] = A[i];
    __syncthreads();
    glass::cgrps::cholDecomp_InPlace<float, true>((uint32_t)n, s, g, fail);
    __syncthreads();
    for (int i = threadIdx.x; i < n*n; i += blockDim.x) A[i] = s[i];
}

template <uint32_t N>
__global__ void k_cholchk_warp(float* A, int* fail) {
    extern __shared__ float s[];
    for (uint32_t i = threadIdx.x; i < N*N; i += blockDim.x) s[i] = A[i];
    __syncthreads();
    glass::warp::cholDecomp_InPlace<float, N, true>(s, fail);   // launched with 32 threads
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < N*N; i += blockDim.x) A[i] = s[i];
}

__global__ void k_ldltchk(int n, float* A, int* fail, int* inertia) {
    extern __shared__ float s[];
    float* sA = s; float* st = s + n*n;
    for (int i = threadIdx.x; i < n*n; i += blockDim.x) sA[i] = A[i];
    __syncthreads();
    glass::ldlt<float, true>((uint32_t)n, sA, st, false, nullptr, fail, inertia);
    __syncthreads();
    for (int i = threadIdx.x; i < n*n; i += blockDim.x) A[i] = sA[i];
}

#define WARP_NS(_) _(1) _(2) _(3) _(4) _(5) _(6) _(7) _(8) _(12) _(14)

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: <op> ...\n"); return 1; }
    const char* op = argv[1];

    if (strcmp(op, "ldltchk") == 0) {
        int th = atoi(argv[2]); int n = atoi(argv[3]);
        float* dA = read_device_vec(argv[4], n*n);
        int *dFail, *dIn; cudaMalloc(&dFail, sizeof(int)); cudaMalloc(&dIn, 3*sizeof(int));
        int smem = (n*n + n + 1) * sizeof(float);
        k_ldltchk<<<1, th, smem>>>(n, dA, dFail, dIn);
        cudaDeviceSynchronize();
        int fail, in[3];
        cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(in, dIn, 3*sizeof(int), cudaMemcpyDeviceToHost);
        printf("%d %d %d %d\n", fail, in[0], in[1], in[2]);
        print_device_vec(dA, n*n);
        return 0;
    }

    // cholchk
    int surf = (strcmp(argv[2], "block")==0) ? 0 : (strcmp(argv[2], "warp")==0) ? 1 : 2;
    int th = atoi(argv[3]); int n = atoi(argv[4]);
    float* dA = read_device_vec(argv[5], n*n);
    int* dFail; cudaMalloc(&dFail, sizeof(int));
    int smem = n*n*sizeof(float);
    if (surf == 0)      k_cholchk_block<<<1, th, smem>>>(n, dA, dFail);
    else if (surf == 2) k_cholchk_cgrps<<<1, th, smem>>>(n, dA, dFail);
    else {
        bool ok = false;
        #define WCASE(NN) if (!ok && n==NN) { k_cholchk_warp<NN><<<1, 32, smem>>>(dA, dFail); ok=true; }
        WARP_NS(WCASE)
        #undef WCASE
        if (!ok) { fprintf(stderr, "warp N=%d not instantiated\n", n); return 1; }
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "kernel err: %s\n", cudaGetErrorString(e)); return 1; }
    int fail; cudaMemcpy(&fail, dFail, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", fail);
    print_device_vec(dA, n*n);
    return 0;
}
