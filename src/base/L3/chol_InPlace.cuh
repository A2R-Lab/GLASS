#pragma once
#include <cstdint>
#include <math.h>

/**
 * @brief In-place Cholesky factorization of an SPD matrix (LAPACK potrf, lower).
 *
 * Factors `A = L * L^T` and overwrites `A` with the lower-triangular factor `L`
 * (only the lower triangle is written; the upper triangle keeps its input
 * values). Single-block, column-major storage, in-place. `A` must be symmetric
 * positive-definite. NumPy equivalent: `L = np.linalg.cholesky(A)`.
 *
 * @tparam T  Scalar type.
 * @param n    Matrix dimension (A is n x n).
 * @param s_A  In/out n x n matrix (column-major); on return its lower triangle holds L.
 */
template <typename T>
__device__ void cholDecomp_InPlace(uint32_t n, T *s_A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t row = 0; row < n; row++) {
        if (rank == 0) {
            T sum = static_cast<T>(0);
            T val = s_A[n*row + row];
            for (int32_t rl = 0; rl < (int32_t)row; rl++) sum += s_A[rl*n + row]*s_A[rl*n + row];
            s_A[row*n + row] = sqrtf(val - sum);
        }
        __syncthreads();
        for (uint32_t col = rank + row + 1; col < n; col += size) {
            T sum = static_cast<T>(0);
            for (uint32_t kk = 0; kk < row; kk++) sum += s_A[kk*n + col]*s_A[kk*n + row];
            s_A[row*n + col] = (static_cast<T>(1)/s_A[row*n + row])*(s_A[row*n + col] - sum);
        }
        __syncthreads();
    }
}

/**
 * @brief Compile-time-size in-place Cholesky factorization (LAPACK potrf, lower).
 *
 * Same as the runtime overload but with the dimension as a template parameter,
 * letting the compiler bake `N` in. Factors the SPD matrix `A = L * L^T` in
 * place, writing only the lower triangle. NumPy equivalent:
 * `L = np.linalg.cholesky(A)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N).
 * @param s_A  In/out N x N matrix (column-major); on return its lower triangle holds L.
 */
template <typename T, uint32_t N>
__device__ void cholDecomp_InPlace(T *s_A)
{
    cholDecomp_InPlace<T>(N, s_A);
}
