// 08_backend_picker.cu — choosing an execution plan with glass-defaults.cuh.
//
// Build (from this examples/ dir, pure SIMT — no MathDx needed):
//   nvcc -std=c++17 -arch=sm_75 -I.. 08_backend_picker.cu -o picker && ./picker
//
// recommend() exposes the measured native/NVIDIA and thread/warp/block ladder
// as one constexpr plan. The choice is host-/codegen-side because scopes need
// different launches. This example requests native-only implementations.

#include "glass.cuh"
#include "glass-defaults.cuh"
#include <cstdio>
#include <cuda_runtime.h>

using glass::op;
static const char* name(glass::execution_plan p) {
    const char* family = p.implementation == glass::family::nvidia ? "nvidia" : "native";
    const char* scope = p.execution_scope == glass::scope::thread ? "thread"
                      : p.execution_scope == glass::scope::warp ? "warp" : "block";
    static char text[32];
    std::snprintf(text, sizeof(text), "%s/%s", family, scope);
    return text;
}

// ── one SPD solve A x = b, dispatched to the picked backend ──────────────────
template <int N> __global__ void k_block_posv(float* A, float* b) { glass::block::posv<float, N>(A, b); }
template <int N> __global__ void k_warp_posv (float* A, float* b) {
    int w = blockIdx.x * blockDim.y + threadIdx.y;            // one warp per problem
    glass::warp::posv<float, N>(A + (size_t)w*N*N, b + w*N);
}
template <int N> __global__ void k_thread_posv(float* A, float* b) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;            // one problem per THREAD
    float a[N*N], x[N];                                       // register-resident at N<=7
    for (int i = 0; i < N*N; i++) a[i] = A[(size_t)p*N*N + i];
    for (int i = 0; i < N;   i++) x[i] = b[(size_t)p*N + i];
    glass::thread::posv<float, N>(a, x);
    for (int i = 0; i < N;   i++) b[(size_t)p*N + i] = x[i];
}

template <int N>
static void solve_dispatch(float* dA, float* db) {
    // Compile-time pick from the measured table (T=float, build's SM).
    constexpr auto plan = glass::recommend<op::posv, float, N>();
    printf("  posv N=%d -> plan=%s", N, name(plan));
    if constexpr (plan.execution_scope == glass::scope::thread) {
        printf(" (TPB=%u)\n", plan.block_threads);
        k_thread_posv<N><<<1, 1>>>(dA, db);                   // 1 problem here -> 1 thread
    } else if constexpr (plan.execution_scope == glass::scope::warp) {
        printf(" (WPB=%u)\n", plan.problems_per_block);
        k_warp_posv<N><<<1, dim3(32, 1)>>>(dA, db);           // 1 problem here -> 1 warp
    } else {
        printf(" (TB=%u)\n", plan.block_threads);
        k_block_posv<N><<<1, plan.block_threads>>>(dA, db);
    }
    cudaDeviceSynchronize();
}

int main() {
    // 1) Show what the picker chooses across ops/sizes (all compile-time constants).
    printf("execution plans (T=float, this build's SM, native-only):\n");
    printf("  dot   N=8  : %s\n", name(glass::recommend<op::dot, float, 8>()));
    printf("  dot   N=64 : %s\n", name(glass::recommend<op::dot, float, 64>()));
    printf("  posv  N=8  : %s\n", name(glass::recommend<op::posv, float, 8>()));
    printf("  gemv  N=16 : %s\n", name(glass::recommend<op::gemv, float, 16>()));
    printf("  gemv  N=64 : %s\n", name(glass::recommend<op::gemv, float, 64>()));
    printf("  gemm  N=8  : %s\n", name(glass::recommend<op::gemm, float, 8>()));
    printf("  gemm  N=32 : %s\n", name(glass::recommend<op::gemm, float, 32>()));
    printf("  potrf N=8  : %s\n", name(glass::recommend<op::potrf, float, 8>()));
    printf("  potrf N=64 : %s\n", name(glass::recommend<op::potrf, float, 64>()));

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
