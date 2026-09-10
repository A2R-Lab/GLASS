// Static resource canary — compile-only, never launched.
//
// One `extern "C" __global__` wrapper per representative (op, tier, dtype)
// so `nvcc -Xptxas -v` reports per-kernel registers / stack / spill / smem
// under stable unmangled names. resource_canary.py compiles this TU GPU-less,
// parses the ptxas report, and diffs it against the committed baseline —
// catching silent register-pressure or spill regressions before any timing
// run. Data flows g -> shared/local -> op -> g so nothing is dead-code
// eliminated. Sizes: N=8 for block/warp tiers, N=6 for the thread tier
// (inside its measured register-promotion ceiling of N<=7).
#include "glass.cuh"

namespace {
template <typename T, int N>
__device__ __forceinline__ void ld(T *dst, const T *src) {
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
}
template <typename T, int N>
__device__ __forceinline__ void st(T *dst, const T *src) {
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x) dst[i] = src[i];
}
}  // namespace

#define BLOCK_CANARY(T, TAG)                                                   \
extern "C" __global__ void canary_block_potrf_##TAG(T *g) {                    \
    __shared__ T A[64]; ld<T, 64>(A, g);                                       \
    glass::block::potrf<T, 8>(A); st<T, 64>(g, A);                             \
}                                                                              \
extern "C" __global__ void canary_block_posv_##TAG(T *g) {                     \
    __shared__ T A[64], b[8]; ld<T, 64>(A, g); ld<T, 8>(b, g + 64);            \
    glass::block::posv<T, 8>(A, b); st<T, 8>(g + 64, b);                       \
}                                                                              \
extern "C" __global__ void canary_block_gemm_##TAG(T *g) {                     \
    __shared__ T A[64], B[64], C[64]; ld<T, 64>(A, g); ld<T, 64>(B, g + 64);   \
    glass::block::gemm<T, 8, 8, 8>(T(1), A, B, T(0), C); st<T, 64>(g + 128, C);\
}                                                                              \
extern "C" __global__ void canary_block_trsv_##TAG(T *g) {                     \
    __shared__ T A[64], x[8]; ld<T, 64>(A, g); ld<T, 8>(x, g + 64);            \
    glass::block::trsv<T, 8>(A, x); st<T, 8>(g + 64, x);                       \
}                                                                              \
extern "C" __global__ void canary_block_gemv_##TAG(T *g) {                     \
    __shared__ T A[64], x[8], y[8]; ld<T, 64>(A, g); ld<T, 8>(x, g + 64);      \
    glass::block::gemv<T, 8, 8>(T(1), A, x, T(0), y); st<T, 8>(g + 72, y);     \
}                                                                              \
extern "C" __global__ void canary_warp_potrf_##TAG(T *g) {                     \
    __shared__ T A[64]; ld<T, 64>(A, g);                                       \
    glass::warp::potrf<T, 8>(A); st<T, 64>(g, A);                              \
}                                                                              \
extern "C" __global__ void canary_warp_gemv_##TAG(T *g) {                      \
    __shared__ T A[64], x[8], y[8]; ld<T, 64>(A, g); ld<T, 8>(x, g + 64);      \
    glass::warp::gemv<T, 8, 8>(T(1), A, x, T(0), y); st<T, 8>(g + 72, y);      \
}                                                                              \
extern "C" __global__ void canary_thread_posv_##TAG(T *g) {                    \
    T A[36], b[6];                                                             \
    for (int i = 0; i < 36; i++) A[i] = g[i];                                  \
    for (int i = 0; i < 6; i++)  b[i] = g[36 + i];                             \
    glass::thread::posv<T, 6>(A, b);                                           \
    T acc = T(0);                                                              \
    for (int i = 0; i < 6; i++) acc += b[i];                                   \
    g[36 + threadIdx.x] = acc;                                                 \
}                                                                              \
extern "C" __global__ void canary_thread_gemm_##TAG(T *g) {                    \
    T A[36], B[36], C[36];                                                     \
    for (int i = 0; i < 36; i++) { A[i] = g[i]; B[i] = g[36 + i]; }            \
    glass::thread::gemm<T, 6, 6, 6>(T(1), A, B, T(0), C);                      \
    T acc = T(0);                                                              \
    for (int i = 0; i < 36; i++) acc += C[i];                                  \
    g[72 + threadIdx.x] = acc;                                                 \
}

BLOCK_CANARY(float, f32)
BLOCK_CANARY(double, f64)
