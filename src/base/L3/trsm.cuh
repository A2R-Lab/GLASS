#pragma once
#include <cstdint>

/**
 * @brief Lower-triangular solve `L x = b` in place via forward substitution (TRSM/TRSV).
 *
 * Solves for `x` given lower-triangular `L` (column-major) and right-hand side
 * `b`, overwriting `b` with the solution. Single-block. SciPy equivalent:
 * `x = scipy.linalg.solve_triangular(L, b, lower=True)`.
 *
 * @tparam T  Scalar type.
 * @param n  Dimension (L is n x n, b has length n).
 * @param L  Lower-triangular matrix (column-major).
 * @param b  In/out right-hand side; on return holds the solution x.
 */
// Solve lower-triangular Lx=b in-place (column-major L, result overwrites b)
template <typename T>
__device__ void trsm(uint32_t n, T *L, T *b)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t col = 0; col < n; col++) {
        if (rank == 0) b[col] /= L[col*n + col];
        __syncthreads();
        T factor = b[col];
        for (uint32_t row = rank + col + 1; row < n; row += size)
            b[row] -= L[col*n + row] * factor;
        __syncthreads();
    }
}

/**
 * @brief Compile-time-size lower-triangular solve `L x = b` in place (TRSM/TRSV).
 *
 * Same as the runtime `trsm` but with the dimension as a template parameter.
 * SciPy equivalent: `x = scipy.linalg.solve_triangular(L, b, lower=True)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Dimension (L is N x N, b has length N).
 * @param L  Lower-triangular matrix (column-major).
 * @param b  In/out right-hand side; on return holds the solution x.
 */
template <typename T, uint32_t N>
__device__ void trsm(T *L, T *b)
{
    trsm<T>(N, L, b);
}
