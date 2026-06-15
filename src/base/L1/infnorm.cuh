#pragma once
#include <cstdint>

/**
 * @brief Infinity norm: `x[0] = ‖x‖∞ = max|x[i]|` (in-place, destructive).
 *
 * Computes the maximum absolute value via a block-wide halving reduction; the
 * result lands in `x[0]` (the input is overwritten). NumPy equivalent:
 * `np.max(np.abs(x))`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Number of elements.
 * @param x  In/out vector of length `n`; the result lands in `x[0]`.
 */
template <typename T>
__device__ void infnorm(uint32_t n, T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = n;
    bool odd;
    while (left > 3) {
        odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = ind; i < left; i += stride)
            x[i] = max(abs(x[i]), abs(x[i + left]));
        if (ind == 0 && odd) x[0] = max(abs(x[0]), abs(x[2*left]));
        __syncthreads();
    }
    if (ind == 0) { for (uint32_t i = 1; i < left; i++) x[0] = max(abs(x[0]), abs(x[i])); }
}

/**
 * @brief Infinity norm: `x[0] = ‖x‖∞ = max|x[i]|`, compile-time size.
 *
 * Compile-time-`N` overload (in-place, destructive). NumPy equivalent:
 * `np.max(np.abs(x))`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param x  In/out vector of length `N`; the result lands in `x[0]`.
 */
template <typename T, uint32_t N>
__device__ void infnorm(T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    uint32_t left = N;
    bool odd;
    while (left > 3) {
        odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = ind; i < left; i += stride)
            x[i] = max(abs(x[i]), abs(x[i + left]));
        if (ind == 0 && odd) x[0] = max(abs(x[0]), abs(x[2*left]));
        __syncthreads();
    }
    if (ind == 0) { for (uint32_t i = 1; i < left; i++) x[0] = max(abs(x[0]), abs(x[i])); }
}
