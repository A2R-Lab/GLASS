#pragma once
#include <cstdint>

// Compile-time-size GEMV with an explicit column-major leading dimension.
// A[i][j] = A_ptr[i + j*ROW_STRIDE].
// When ROW_STRIDE == M this is identical to glass::gemv<T,M,N>.
// Useful for spatial 6x6 matrices embedded in larger arrays (e.g., GRiD).
//
// Uses threadIdx-based parallelism (same as gemv_impl_ct): threads are
// distributed over rows, and the inner column loop is fully unrolled by the
// compiler since N and ROW_STRIDE are compile-time constants.

/**
 * @brief Column-major GEMV with an explicit leading dimension: `y = alpha * A * x + beta * y`.
 *
 * Compile-time-`M`,`N` matrix-vector product where `A[i][j] = A[i + j*ROW_STRIDE]`,
 * letting an `M×N` matrix be addressed inside a larger array (e.g. a spatial 6×6
 * embedded in a wider buffer, as in GRiD). When `ROW_STRIDE == M` this is
 * identical to `glass::gemv<T,M,N>`. NumPy equivalent: `y = alpha*A@x + beta*y`.
 *
 * @tparam T           Scalar type (e.g. `float`, `double`).
 * @tparam M           Number of rows of `A` (compile-time constant).
 * @tparam N           Number of columns of `A` (compile-time constant).
 * @tparam ROW_STRIDE  Column-major leading dimension of `A` (default `M`).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix, addressed at `A[row + col*ROW_STRIDE]`.
 * @param x      Input vector of length `N`.
 * @param beta   Scalar multiplier on the prior `y`.
 * @param y      In/out vector of length `M`.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M>
__device__ void gemv_strided(T alpha, const T* A, const T* x, T beta, T* y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t row = rank; row < M; row += size) {
        T res = static_cast<T>(0);
        for (uint32_t col = 0; col < N; col++)
            res += A[row + col * ROW_STRIDE] * x[col];
        y[row] = alpha * res + beta * y[row];
    }
}

/**
 * @brief Column-major GEMV with an explicit leading dimension: `y = alpha * A * x`, no-beta overload.
 *
 * No-`beta` variant of the strided GEMV that overwrites `y`, with
 * `A[i][j] = A[i + j*ROW_STRIDE]`. NumPy equivalent: `y = alpha*A@x`.
 *
 * @tparam T           Scalar type (e.g. `float`, `double`).
 * @tparam M           Number of rows of `A` (compile-time constant).
 * @tparam N           Number of columns of `A` (compile-time constant).
 * @tparam ROW_STRIDE  Column-major leading dimension of `A` (default `M`).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix, addressed at `A[row + col*ROW_STRIDE]`.
 * @param x      Input vector of length `N`.
 * @param y      Output vector of length `M`.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M>
__device__ void gemv_strided(T alpha, const T* A, const T* x, T* y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t row = rank; row < M; row += size) {
        T res = static_cast<T>(0);
        for (uint32_t col = 0; col < N; col++)
            res += A[row + col * ROW_STRIDE] * x[col];
        y[row] = alpha * res;
    }
}
