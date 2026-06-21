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
 * @brief Fused in-place Cholesky factorization of K independent SPD matrices (lower).
 *
 * Factors `K` SPD matrices simultaneously in one block by interleaving their
 * column sweeps over a single shared `MAX_DIM = max(dims)` row loop: matrix `m`
 * participates while `row < dims[m]` and sits idle thereafter. Each matrix keeps
 * the same column-major in-place `A = L*L^T` convention as the single-matrix
 * `cholDecomp_InPlace` — on return the lower triangle of `mats[m]` holds its
 * factor `L` (the upper triangle keeps its input values). Same two-barriers-per
 * step structure as the single-matrix path; no shared scratch required.
 *
 * The K diagonals of a given step are distributed across threads
 * (`for (m = rank; m < K; m += size)`, more parallel than rank-0 alone); then the
 * trailing sub-diagonal column entries of each active matrix are updated in
 * parallel. NumPy equivalent (per matrix `m`):
 * `cholesky(m) = np.linalg.cholesky(mats[m])` (lower).
 *
 * @tparam T  Scalar type.
 * @param K        Number of matrices.
 * @param dims     Per-matrix dimensions (`dims[m]` for matrix `m`).
 * @param MAX_DIM  `max(dims[0..K-1])` — the shared row-loop length (precondition).
 * @param mats     Array of K in/out column-major SPD buffers (`dims[m] x dims[m]`);
 *                 on return each lower triangle holds its Cholesky factor `L`.
 */
template <typename T>
__device__ void cholDecomp_InPlace(uint32_t K, const uint32_t *dims, uint32_t MAX_DIM, T **mats)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t row = 0; row < MAX_DIM; row++) {
        // Phase 1: diagonal of each active matrix — distribute the K diagonals across threads.
        for (uint32_t m = rank; m < K; m += size) {
            uint32_t n = dims[m];
            if (row < n) {
                T *s_A = mats[m];
                T sum = static_cast<T>(0);
                T val = s_A[n*row + row];
                for (int32_t rl = 0; rl < (int32_t)row; rl++) sum += s_A[rl*n + row]*s_A[rl*n + row];
                s_A[row*n + row] = sqrtf(val - sum);
            }
        }
        __syncthreads();
        // Phase 2: trailing sub-diagonal column entries of each active matrix.
        for (uint32_t m = 0; m < K; m++) {
            uint32_t n = dims[m];
            if (row < n) {
                T *s_A = mats[m];
                for (uint32_t col = rank + row + 1; col < n; col += size) {
                    T sum = static_cast<T>(0);
                    for (uint32_t kk = 0; kk < row; kk++) sum += s_A[kk*n + col]*s_A[kk*n + row];
                    s_A[row*n + col] = (static_cast<T>(1)/s_A[row*n + row])*(s_A[row*n + col] - sum);
                }
            }
        }
        __syncthreads();
    }
}

/**
 * @brief Fused in-place Cholesky factorization of TWO SPD matrices (lower).
 *
 * Thin wrapper over the K-way `cholDecomp_InPlace` (K=2). Same column-major
 * in-place `A = L*L^T` convention and output as the single-matrix path.
 * NumPy: `La, Lb = cholesky(A), cholesky(B)` (lower).
 *
 * @tparam T  Scalar type.
 * @param dimA,dimB  Matrix dimensions.
 * @param MAX_DIM    `max(dimA, dimB)` — the shared row-loop length.
 * @param A,B        In/out column-major SPD buffers (dim x dim); lower triangles hold L.
 */
template <typename T>
__device__ void cholDecomp_InPlace(uint32_t dimA, uint32_t dimB, uint32_t MAX_DIM, T *A, T *B)
{
    uint32_t dims[2] = {dimA, dimB};
    T *mats[2] = {A, B};
    cholDecomp_InPlace<T>(2, dims, MAX_DIM, mats);
}

/**
 * @brief Fused in-place Cholesky factorization of THREE SPD matrices (lower).
 *
 * Thin wrapper over the K-way `cholDecomp_InPlace` (K=3). Same column-major
 * in-place `A = L*L^T` convention and output as the single-matrix path.
 * NumPy: invert-free factor each independently (lower).
 *
 * @tparam T  Scalar type.
 * @param dimA,dimB,dimC  Matrix dimensions.
 * @param MAX_DIM         `max(dimA, dimB, dimC)` — the shared row-loop length.
 * @param A,B,C           In/out column-major SPD buffers (dim x dim); lower triangles hold L.
 */
template <typename T>
__device__ void cholDecomp_InPlace(uint32_t dimA, uint32_t dimB, uint32_t dimC, uint32_t MAX_DIM, T *A, T *B, T *C)
{
    uint32_t dims[3] = {dimA, dimB, dimC};
    T *mats[3] = {A, B, C};
    cholDecomp_InPlace<T>(3, dims, MAX_DIM, mats);
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

namespace warp {
    /**
     * @brief Single-warp in-place Cholesky factorization (LAPACK potrf, lower), compile-time size.
     *
     * One 32-lane warp factors the SPD matrix `A = L * L^T` in place, writing only
     * the lower triangle (column-major). For warp-per-problem solvers on small
     * systems (e.g. N≈7 normal equations). Lane 0 computes each diagonal; the
     * remaining sub-diagonal entries of the column are filled by the warp's lanes
     * (stride 32), synchronized with `__syncwarp`. No shared scratch, no
     * `__syncthreads`. `A` must be SPD. NumPy equivalent:
     * `L = np.linalg.cholesky(A)`.
     *
     * @tparam T  Scalar type (use `double` for stability on ill-conditioned A).
     * @tparam N  Matrix dimension (A is N x N).
     * @param s_A  In/out N x N matrix (column-major); on return its lower triangle holds L.
     */
    template <typename T, uint32_t N>
    __device__ void cholDecomp_InPlace(T *s_A)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t k = 0; k < N; k++) {
            T diag = static_cast<T>(0);
            if (lane == 0) {
                T sum = static_cast<T>(0);
                T val = s_A[k*N + k];
                for (uint32_t r = 0; r < k; r++) sum += s_A[r*N + k]*s_A[r*N + k];
                diag = sqrt(val - sum);
                s_A[k*N + k] = diag;
            }
            // Broadcast the pivot from lane 0's REGISTER via __shfl_sync rather than having
            // every lane re-read s_A[k*N+k] from shared. The shared re-read is the same
            // write-then-read-same-location pattern that nvcc can cache stale when the buffer
            // is reached through a caller `__restrict__` pointer (observed: in-place warp solve
            // gave wrong results for ~10% of inputs under -restrict; shfl broadcast is immune
            // and matches glass::warp::reduce's own shfl-based design).
            diag = __shfl_sync(0xffffffffu, diag, 0);
            for (uint32_t row = lane + k + 1; row < N; row += 32) {
                T sum = static_cast<T>(0);
                for (uint32_t kk = 0; kk < k; kk++) sum += s_A[kk*N + row]*s_A[kk*N + k];
                s_A[k*N + row] = (s_A[k*N + row] - sum) / diag;
            }
            __syncwarp();
        }
    }
}
