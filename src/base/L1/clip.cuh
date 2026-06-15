#pragma once
#include <cstdint>

/**
 * @brief Element-wise clamp in place: `x = clamp(x, l, u)`.
 *
 * Each element is bounded below by the corresponding `l[i]` and above by
 * `u[i]`. NumPy equivalent: `np.clip(x, l, u)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n` (overwritten with the clamped values).
 * @param l  Per-element lower bounds, length `n`.
 * @param u  Per-element upper bounds, length `n`.
 */
template <typename T>
__device__ void clip(uint32_t n, T *x, T *l, T *u)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n; i += size)
        x[i] = max(l[i], min(x[i], u[i]));
}

/**
 * @brief Element-wise clamp in place: `x = clamp(x, l, u)`, compile-time size.
 *
 * Compile-time-`N` overload of clip. NumPy equivalent: `np.clip(x, l, u)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N` (overwritten with the clamped values).
 * @param l  Per-element lower bounds, length `N`.
 * @param u  Per-element upper bounds, length `N`.
 */
template <typename T, uint32_t N>
__device__ void clip(T *x, T *l, T *u)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N; i += size)
        x[i] = max(l[i], min(x[i], u[i]));
}
