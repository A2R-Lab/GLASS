// smoke_simt.cu — backwards-reach probe: one TU touching all three SIMT tiers
// (block L3 factor/solve, warp L1/L2, thread L1) plus the no-vendor dispatch
// collapse. Compile-only, per old -arch target: if this TU builds, the GLASS
// SIMT surface reaches that architecture.
#include "glass.cuh"
#include "glass-defaults.cuh"

template <typename T, uint32_t N>
__global__ void smoke(T* A, T* B, T* C, T* x, T* y) {
    glass::block::gemm<T>(N, N, N, (T)1, A, B, (T)0, C);
    glass::block::potrf<T>(N, A);
    glass::block::trsv<T>(N, A, x);
    T d = glass::warp::dot<T, N>(x, y);
    glass::warp::reduce<T, N>(y);
    T t = glass::thread::dot<T, 4>(x, y);
    if (threadIdx.x == 0) x[0] = d + t;
}

// dispatch tables must resolve on an arch no sweep has measured (collapse path)
static_assert(glass::defaults::ideal(glass::op::gemm, 16, false, 620u) ==
              glass::defaults::ideal_generic(glass::op::gemm, 16, false),
              "unmeasured SM falls to generic");

template __global__ void smoke<float, 16>(float*, float*, float*, float*, float*);
template __global__ void smoke<double, 16>(double*, double*, double*, double*, double*);
