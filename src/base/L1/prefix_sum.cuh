#pragma once
#include <cstdint>

/**
 * @brief Exclusive prefix sum (scan): `s_output[i] = Σ_{j<i} s_input[j]`.
 *
 * Block-wide Hillis-Steele scan; `s_output[0]` is 0 and each subsequent entry
 * is the running total of all strictly-earlier inputs. NumPy equivalent:
 * `s_output = np.concatenate([[0], np.cumsum(s_input)[:-1]])`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param s_input   Input vector of length `n` (in shared memory).
 * @param s_output  Output scan buffer of length `n` (in shared memory).
 * @param n         Number of elements (must not exceed `blockDim.x`).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void prefix_sum_exclusive(T *s_input, T *s_output, int n)
{
    int tid = threadIdx.x;
    s_output[tid] = (tid < n && tid > 0) ? s_input[tid-1] : static_cast<T>(0);
    __syncthreads();
    T tmp;
    for (int d = 1; d < n; d *= 2) {
        __syncthreads();
        if (tid < n && tid >= d) tmp = s_output[tid] + s_output[tid-d];
        __syncthreads();
        if (tid < n && tid >= d) s_output[tid] = tmp;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Inclusive prefix sum (scan): `s_output[i] = Σ_{j<=i} s_input[j]`.
 *
 * Block-wide Hillis-Steele scan; each entry is the running total of all inputs
 * up to and including that index. NumPy equivalent: `s_output = np.cumsum(s_input)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param s_input   Input vector of length `n` (in shared memory).
 * @param s_output  Output scan buffer of length `n` (in shared memory).
 * @param n         Number of elements (must not exceed `blockDim.x`).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void prefix_sum_inclusive(T *s_input, T *s_output, int n)
{
    int tid = threadIdx.x;
    if (tid < n) s_output[tid] = s_input[tid];
    __syncthreads();
    T tmp;
    for (int d = 1; d < n; d *= 2) {
        __syncthreads();
        if (tid < n && tid >= d) tmp = s_output[tid-d] + s_output[tid];
        __syncthreads();
        if (tid < n && tid >= d) s_output[tid] = tmp;
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
