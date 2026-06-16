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

namespace warp {
    /**
     * @brief Single-warp lower-triangular solve `L x = b` (forward substitution), compile-time size.
     *
     * One 32-lane warp solves `L x = b` in place (column-major lower-triangular `L`),
     * overwriting `b`. For warp-per-problem solvers; pairs with `warp::trsm_transpose`
     * to solve an SPD system from its Cholesky factor. No `__syncthreads`. SciPy:
     * `x = scipy.linalg.solve_triangular(L, b, lower=True)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (L is N x N, b has length N).
     * @param L  Lower-triangular matrix (column-major).
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void trsm(T *L, T *b)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t col = 0; col < N; col++) {
            if (lane == 0) b[col] /= L[col*N + col];
            __syncwarp();
            T factor = b[col];
            for (uint32_t row = lane + col + 1; row < N; row += 32)
                b[row] -= L[col*N + row] * factor;
            __syncwarp();
        }
    }

    /**
     * @brief Single-warp transpose-triangular solve `Lᵀ x = b` (back substitution), compile-time size.
     *
     * One 32-lane warp solves `Lᵀ x = b` in place given a lower-triangular `L`
     * (column-major), overwriting `b`. Together with `warp::trsm` this solves an SPD
     * system `A x = b` from `A = L Lᵀ`: factor with `warp::cholDecomp_InPlace`, then
     * `warp::trsm` (forward) then `warp::trsm_transpose` (back). No `__syncthreads`.
     * SciPy: `x = scipy.linalg.solve_triangular(L.T, b, lower=False)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (L is N x N, b has length N).
     * @param L  Lower-triangular matrix (column-major); `Lᵀ` is used implicitly.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void trsm_transpose(T *L, T *b)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (int32_t col = (int32_t)N - 1; col >= 0; col--) {
            if (lane == 0) b[col] /= L[col*N + col];
            __syncwarp();
            T factor = b[col];
            // eliminate x[col] from rows i < col:  b[i] -= (Lᵀ)_{i,col} x[col] = L_{col,i} x[col]
            for (uint32_t i = lane; i < (uint32_t)col; i += 32)
                b[i] -= L[i*N + col] * factor;
            __syncwarp();
        }
    }
}
