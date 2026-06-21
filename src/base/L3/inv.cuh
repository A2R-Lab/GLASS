#pragma once
#include <cstdint>

/**
 * @brief In-place matrix inverse via Gauss-Jordan on an augmented `[A | I]` layout.
 *
 * Reduces a column-major augmented `dimA x (2*dimA)` matrix `[A | I]` so that on
 * return columns `dimA..2*dimA-1` hold `A^-1`. Single-block; the pivot loop is
 * serial (pivot-to-pivot dependency) while each pivot's cell updates are
 * parallelized across the block. NumPy equivalent: `Ainv = np.linalg.inv(A)`.
 *
 * @tparam T  Scalar type.
 * @param dimA    Matrix dimension (A is dimA x dimA).
 * @param A       In/out augmented `[A | I]` buffer (column-major, dimA x 2*dimA);
 *                on return its right half holds `A^-1`.
 * @param s_temp  Shared scratch of `(2*dimA + 1) * sizeof(T)` bytes.
 */
// Gauss-Jordan inversion of an augmented dimA×(2*dimA) matrix in-place.
// Expected layout: column-major [A | I]; on return columns dimA..2*dimA-1 hold A^-1.
// s_temp: (2*dimA+1)*sizeof(T) bytes.
template <typename T>
__device__ void invertMatrix(uint32_t dimA, T *A, T *s_temp)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivOff = pivRC * dimA;
        T pvInv = static_cast<T>(1) / A[pivRC + pivOff];
        for (unsigned ind = rank; ind < 2*dimA+1; ind++) {
            unsigned AInd = (ind < dimA) ? (ind + pivOff) : (pivRC + pivOff + (ind-dimA)*dimA);
            s_temp[ind] = A[AInd];
        }
        __syncthreads();
        for (unsigned ind = rank; ind < dimA*(dimA+1); ind += size) {
            unsigned row = ind % dimA, col = ind / dimA, coff = ind - row;
            if (row == pivRC) A[row + pivOff + coff] *= pvInv;
            else A[row + pivOff + coff] -= s_temp[row]*pvInv*s_temp[dimA+col];
        }
        __syncthreads();
    }
}

/**
 * @brief Compile-time-size in-place matrix inverse (augmented `[A | I]` Gauss-Jordan).
 *
 * Same as the runtime `invertMatrix` but with the dimension as a template
 * parameter. NumPy equivalent: `Ainv = np.linalg.inv(A)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N).
 * @param A       In/out augmented `[A | I]` buffer (column-major, N x 2*N);
 *                on return its right half holds `A^-1`.
 * @param s_temp  Shared scratch of `(2*N + 1) * sizeof(T)` bytes.
 */
template <typename T, uint32_t N>
__device__ void invertMatrix(T *A, T *s_temp)
{
    invertMatrix<T>(N, A, s_temp);
}

/**
 * @brief Fused in-place inverse of K independent matrices (augmented `[V | I]`).
 *
 * Inverts `K` matrices simultaneously in one block by interleaving their
 * Gauss-Jordan sweeps over a single shared `MAX_DIM = max(dims)` pivot loop:
 * matrix `m` participates while `pivRC < dims[m]` and sits idle thereafter.
 * Every matrix keeps the same augmented `[V | I]` convention as the
 * single-matrix `invertMatrix` — buffer `mats[m]` is column-major
 * `dims[m] x (2*dims[m])` and on return its right half holds `inv(mats[m])`.
 * Fewer barriers than K separate calls (one save→update barrier pair per pivot
 * step, shared by all K matrices). Used by GATO's Schur kernel
 * (Q_k, Q_kp1, R_k → K=3).
 *
 * Scratch layout: matrix `m` owns the contiguous span
 * `[Σ_{j<m}(2*dims[j]+1), Σ_{j<=m}(2*dims[j]+1))` of `s_temp` (each matrix needs
 * `2*dims[m]+1` slots: `dims[m]` for its pivot column, `dims[m]+1` for its pivot
 * row plus the augmented column). The per-matrix base offset is the prefix sum
 * `Σ_{j<m}(2*dims[j]+1)`, recomputed locally per thread by scanning `dims[]`
 * (no shared write, so race-free). Total scratch = `Σ_m (2*dims[m]+1)` elements.
 *
 * NumPy equivalent (per matrix `m`): `inv(m) = np.linalg.inv(mats[m])`.
 *
 * @tparam T  Scalar type.
 * @param K        Number of matrices.
 * @param dims     Per-matrix dimensions (`dims[m]` for matrix `m`).
 * @param MAX_DIM  `max(dims[0..K-1])` — the shared pivot-loop length (precondition).
 * @param mats     Array of K in/out augmented `[V | I]` buffers (column-major,
 *                 `dims[m] x 2*dims[m]`); on return each right half holds its inverse.
 * @param s_temp   Shared scratch of `(Σ_m (2*dims[m]+1)) * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix(uint32_t K, const uint32_t *dims, uint32_t MAX_DIM, T **mats, T *s_temp)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++) {
        // Phase 1: save each active matrix's pivot row + column into its scratch span.
        // Strided over the union of work; per-matrix scratch base = prefix sum of (2*dim+1).
        uint32_t sOff = 0;
        for (unsigned m = 0; m < K; m++) {
            uint32_t dim = dims[m];
            if (pivRC < dim) {
                T *M = mats[m];
                T *s_mem = &s_temp[sOff];
                unsigned pivOff = pivRC * dim;
                // pivot column (dim entries) + pivot row & augmented column (dim+1 entries)
                for (unsigned ind = rank; ind < dim; ind += size)
                    s_mem[ind] = M[ind + pivOff];
                for (unsigned ind = rank; ind < dim + 1; ind += size)
                    s_mem[ind + dim] = M[ind*dim + pivRC + pivOff];
            }
            sOff += 2*dim + 1;
        }
        __syncthreads();
        // Phase 2: Gauss-Jordan cell update for each active matrix.
        sOff = 0;
        for (unsigned m = 0; m < K; m++) {
            uint32_t dim = dims[m];
            if (pivRC < dim) {
                T *M = mats[m];
                T *s_mem = &s_temp[sOff];
                unsigned pivOff = pivRC * dim;
                for (unsigned ind = rank; ind < dim*(dim + 1); ind += size) {
                    unsigned row = ind % dim, col = ind / dim;
                    if (row == pivRC) M[pivOff + ind] /= s_mem[pivRC];
                    else M[pivOff + ind] -= s_mem[row] / s_mem[pivRC] * s_mem[dim + col];
                }
            }
            sOff += 2*dim + 1;
        }
        __syncthreads();
    }
}

/**
 * @brief Fused in-place inverse of TWO independent matrices (augmented `[V | I]`).
 *
 * Thin wrapper over the K-way `invertMatrix` (K=2). Inverts `A` (dimA x dimA) and
 * `B` (dimB x dimB) simultaneously in one block; same augmented `[V | I]`
 * convention and output as the single-matrix `invertMatrix`.
 * NumPy: `Ainv, Binv = inv(A), inv(B)`.
 *
 * @tparam T  Scalar type.
 * @param dimA,dimB  Matrix dimensions.
 * @param MAX_DIM    `max(dimA, dimB)` — the shared pivot-loop length.
 * @param A,B        In/out augmented `[V | I]` buffers (column-major, dim x 2*dim).
 * @param s_temp     Shared scratch of `(2*dimA + 2*dimB + 2) * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix(uint32_t dimA, uint32_t dimB, uint32_t MAX_DIM, T *A, T *B, T *s_temp)
{
    uint32_t dims[2] = {dimA, dimB};
    T *mats[2] = {A, B};
    invertMatrix<T>(2, dims, MAX_DIM, mats, s_temp);
}

/**
 * @brief Fused in-place inverse of THREE independent matrices (augmented `[V | I]`).
 *
 * Thin wrapper over the K-way `invertMatrix` (K=3). Inverts `A`,`B`,`C`
 * simultaneously in one block; same augmented `[V | I]` convention and output as
 * the single-matrix `invertMatrix`. Used by GATO's Schur kernel (Q_k, Q_kp1, R_k).
 * NumPy: invert each independently.
 *
 * @tparam T  Scalar type.
 * @param dimA,dimB,dimC  Matrix dimensions.
 * @param MAX_DIM         `max(dimA, dimB, dimC)` — the shared pivot-loop length.
 * @param A,B,C           In/out augmented `[V | I]` buffers (column-major, dim x 2*dim).
 * @param s_temp          Shared scratch of `(2*dimA + 2*dimB + 2*dimC + 3) * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix(uint32_t dimA, uint32_t dimB, uint32_t dimC, uint32_t MAX_DIM, T *A, T *B, T *C, T *s_temp)
{
    uint32_t dims[3] = {dimA, dimB, dimC};
    T *mats[3] = {A, B, C};
    invertMatrix<T>(3, dims, MAX_DIM, mats, s_temp);
}


/**
 * @brief Dense (no augmented buffer) in-place matrix inverse via dual-update Gauss-Jordan.
 *
 * Inverts a column-major `dimA x dimA` matrix without materializing the `[A | I]`
 * augmented layout: tracks A and a separate inverse buffer through the same
 * `[A | I] -> [I | A^{-1}]` reduction. On return BOTH `A` and `Ainv` hold
 * `A^{-1}` (callers that want the original A preserved should copy it first).
 * Single-block: serial pivot loop, block-parallel per-pivot updates. NumPy
 * equivalent: `Ainv = np.linalg.inv(A)`.
 *
 * @tparam T  Scalar type.
 * @param dimA    Matrix dimension (A is dimA x dimA).
 * @param A       In/out column-major dimA x dimA matrix; on return holds `A^{-1}`.
 * @param Ainv    Workspace column-major dimA x dimA; on return also holds `A^{-1}`.
 * @param s_temp  Shared scratch of `3 * dimA * sizeof(T)` bytes.
 */
// Block-cooperative Gauss-Jordan inversion of a dimA×dimA matrix.
//   A:    in/out, column-major dimA×dimA.  On return: A := A^{-1}.
//   Ainv: workspace, column-major dimA×dimA (overwritten).  On return: A^{-1}.
//          (Some callers want A unchanged + a separate inverse — they get both.)
//   s_temp: shared scratch of size (3*dimA)*sizeof(T).
// Pivot loop is serial (pivot-to-pivot data dependency); within each pivot, the
// dimA save and the dimA*dimA cell update are parallelized across the block.
//
// Algorithm: same dual-update (A, Ainv) Gauss-Jordan as a classic [A | I] →
// [I | A^{-1}] reduction but without materializing the augmented layout. A and
// Ainv are tracked in separate buffers so callers that need A^{-1} alongside
// the original A get it without extra copies.
template <typename T>
__device__ void invertMatrix_dense(uint32_t dimA, T *A, T *Ainv, T *s_temp)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    // Seed Ainv = I (parallel over dimA*dimA cells).
    for (uint32_t ind = rank; ind < dimA*dimA; ind += size) {
        uint32_t row = ind % dimA, col = ind / dimA;
        Ainv[ind] = (row == col) ? static_cast<T>(1) : static_cast<T>(0);
    }
    __syncthreads();
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivColOffset = pivRC * dimA;
        // Block-cooperative save: pivot row of A, pivot row of Ainv, pivot column of A.
        // 3*dimA entries → block-strided thread distribution.
        for (uint32_t ind = rank; ind < dimA; ind += size) {
            s_temp[ind]            = A[pivRC + dimA * ind];          // pivot row of A
            s_temp[ind + dimA]     = Ainv[pivRC + dimA * ind];        // pivot row of Ainv
            s_temp[ind + dimA * 2] = A[ind + pivColOffset];           // pivot column of A
        }
        __syncthreads();
        T pvInv = static_cast<T>(1) / s_temp[pivRC];
        // Block-cooperative Gauss-Jordan update across dimA*dimA cells.
        // Each thread owns a unique (row, col) so writes are race-free.
        for (uint32_t ind = rank; ind < dimA * dimA; ind += size) {
            uint32_t row = ind % dimA, col = ind / dimA;
            if (row == pivRC) {
                A[row + dimA * col]    = s_temp[col] * pvInv;
                Ainv[row + dimA * col] = s_temp[col + dimA] * pvInv;
            } else {
                T multiplier = s_temp[row + dimA * 2] * pvInv;
                A[row + dimA * col]    -= multiplier * s_temp[col];
                Ainv[row + dimA * col] -= multiplier * s_temp[col + dimA];
            }
        }
        __syncthreads();
    }
}

/**
 * @brief Compile-time-size dense in-place matrix inverse (dual-update Gauss-Jordan).
 *
 * Same as the runtime `invertMatrix_dense` but with the dimension as a template
 * parameter; on return both `A` and `Ainv` hold `A^{-1}`. NumPy equivalent:
 * `Ainv = np.linalg.inv(A)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N).
 * @param A       In/out column-major N x N matrix; on return holds `A^{-1}`.
 * @param Ainv    Workspace column-major N x N; on return also holds `A^{-1}`.
 * @param s_temp  Shared scratch of `3 * N * sizeof(T)` bytes.
 */
template <typename T, uint32_t N>
__device__ void invertMatrix_dense(T *A, T *Ainv, T *s_temp)
{
    invertMatrix_dense<T>(N, A, Ainv, s_temp);
}
