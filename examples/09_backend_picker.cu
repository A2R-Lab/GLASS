// 09_backend_picker.cu — choosing a backend + launch config with glass-defaults.cuh.
//
// Build (from this examples/ dir, pure SIMT — no MathDx needed):
//   nvcc -std=c++17 -arch=sm_75 -I.. 09_backend_picker.cu -o picker && ./picker
//   (to make the `nvidia` tier eligible, include glass-nvidia.cuh first + link MathDx.)
//
// glass-defaults.cuh exposes the measured warp/block/nvidia ladder (bench/MEGA_SWEEP_RESULTS.md)
// as constexpr helpers. The pick is host-/codegen-side because warp, block, and nvidia need
// DIFFERENT <<<grid,block>>> launches — so you query at compile time and branch the launch.
// With no MathDx linked (as here), the `nvidia` tier collapses to its warp/block runner-up.

#include "glass.cuh"
#include "glass-defaults.cuh"
#include <cstdio>
#include <cuda_runtime.h>

using glass::op;
using glass::backend;

static const char* name(backend b) {
    return b == backend::warp ? "warp" : b == backend::block ? "block" : "nvidia";
}

// ── one SPD solve A x = b, dispatched to the picked backend ──────────────────
template <int N> __global__ void k_block_posv(float* A, float* b) { glass::posv<float, N>(A, b); }
template <int N> __global__ void k_warp_posv (float* A, float* b) {
    int w = blockIdx.x * blockDim.y + threadIdx.y;            // one warp per problem
    glass::warp::posv<float, N>(A + (size_t)w*N*N, b + w*N);
}

template <int N>
static void solve_dispatch(float* dA, float* db) {
    // Compile-time pick from the measured table (T=float, build's SM).
    constexpr backend be = glass::suggested_backend<op::posv, N, float>();
    printf("  posv N=%d -> backend=%s", N, name(be));
    if constexpr (be == backend::warp) {
        constexpr int WPB = glass::suggested_warps_per_block<op::posv>();
        printf(" (WPB=%d)\n", WPB);
        k_warp_posv<N><<<1, dim3(32, 1)>>>(dA, db);           // 1 problem here -> 1 warp
    } else { // block (or nvidia collapsed to block); a real nvidia tier would launch cuSOLVERDx
        constexpr int TB = glass::suggested_block_threads<op::posv, N, float>();
        printf(" (TB=%d)\n", TB);
        k_block_posv<N><<<1, TB>>>(dA, db);
    }
    cudaDeviceSynchronize();
}

int main() {
    // 1) Show what the picker chooses across ops/sizes (all compile-time constants).
    printf("backend picks (T=float, this build's SM; no MathDx -> nvidia collapses):\n");
    printf("  dot  N=64  : %s\n", name(glass::suggested_backend<op::dot,  64, float>()));
    printf("  gemv N=16  : %s\n", name(glass::suggested_backend<op::gemv, 16, float>()));
    printf("  gemv N=64  : %s\n", name(glass::suggested_backend<op::gemv, 64, float>()));
    printf("  gemm N=8   : %s\n", name(glass::suggested_backend<op::gemm,  8, float>()));
    printf("  gemm N=32  : %s\n", name(glass::suggested_backend<op::gemm, 32, float>()));
    printf("  chol N=8   : %s\n", name(glass::suggested_backend<op::chol,  8, float>()));
    printf("  chol N=64  : %s\n", name(glass::suggested_backend<op::chol, 64, float>()));

    // 2) Use the pick to dispatch a real solve. SPD A = M·Mᵀ + N·I (column-major), N=16.
    const int N = 16;
    float hA[N*N], hb[N];
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) {
        float m=0; for (int k=0;k<N;k++) m += (((i+2*k)%5)*0.1f) * (((j+2*k)%5)*0.1f);
        hA[i+j*N] = m + (i==j ? (float)N : 0.0f);
    }
    for (int i=0;i<N;i++) hb[i] = 1.0f + 0.1f*i;
    float *dA,*db; cudaMalloc(&dA,N*N*4); cudaMalloc(&db,N*4);
    cudaMemcpy(dA,hA,N*N*4,cudaMemcpyHostToDevice); cudaMemcpy(db,hb,N*4,cudaMemcpyHostToDevice);

    printf("\ndispatch a real solve:\n");
    solve_dispatch<N>(dA, db);

    float hx[N]; cudaMemcpy(hx,db,N*4,cudaMemcpyDeviceToHost);
    float res=0; for (int i=0;i<N;i++){ float Ax=0; for (int j=0;j<N;j++) Ax+=hA[i+j*N]*hx[j]; float r=Ax-hb[i]; res = r<0?(res<-r?res:-r):(res<r?r:res); }
    printf("  residual ||A x - b||_inf = %.2e  -> %s\n", res, res < 1e-3f ? "OK" : "FAIL");

    cudaFree(dA); cudaFree(db);
    return 0;
}
