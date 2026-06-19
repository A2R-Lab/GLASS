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
 * @brief Fused in-place inverse of TWO independent matrices (augmented `[A | I]`).
 *
 * Inverts `A` (dimA x dimA) and `B` (dimB x dimB) simultaneously in one block by
 * interleaving their Gauss-Jordan sweeps over a shared `MAX_DIM = max(dimA, dimB)`
 * pivot loop (a matrix sits idle once its dimension is exhausted). Same augmented
 * `[V | I]` convention as the single-matrix `invertMatrix`: each buffer is
 * column-major `dim x (2*dim)` and on return its right half holds the inverse.
 * Fewer barriers than two separate calls. NumPy: `Ainv, Binv = inv(A), inv(B)`.
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
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T *s_memA = s_temp;
    T *s_memB = &s_memA[2*dimA + 1];
    for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++) {
        bool AActive = pivRC < dimA;
        bool BActive = pivRC < dimB;
        unsigned pivOffA = pivRC * dimA;
        unsigned pivOffB = pivRC * dimB;
        for (unsigned ind = rank; ind < MAX_DIM; ind += size) {
            if (AActive && ind < dimA) s_memA[ind] = A[ind + pivOffA];
            if (BActive && ind < dimB) s_memB[ind] = B[ind + pivOffB];
        }
        for (unsigned ind = rank; ind < MAX_DIM + 1; ind += size) {
            if (AActive && ind < dimA + 1) s_memA[ind + dimA] = A[ind*dimA + pivRC + pivOffA];
            if (BActive && ind < dimB + 1) s_memB[ind + dimB] = B[ind*dimB + pivRC + pivOffB];
        }
        __syncthreads();
        for (unsigned ind = rank; ind < MAX_DIM*(MAX_DIM + 1); ind += size) {
            if (AActive && ind < dimA*(dimA + 1)) {
                unsigned row = ind % dimA, col = ind / dimA;
                if (row == pivRC) A[pivOffA + ind] /= s_memA[pivRC];
                else A[pivOffA + ind] -= s_memA[row] / s_memA[pivRC] * s_memA[dimA + col];
            }
            if (BActive && ind < dimB*(dimB + 1)) {
                unsigned row = ind % dimB, col = ind / dimB;
                if (row == pivRC) B[pivOffB + ind] /= s_memB[pivRC];
                else B[pivOffB + ind] -= s_memB[row] / s_memB[pivRC] * s_memB[dimB + col];
            }
        }
        __syncthreads();
    }
}

/**
 * @brief Fused in-place inverse of THREE independent matrices (augmented `[A | I]`).
 *
 * Inverts `A`,`B`,`C` simultaneously in one block over a shared
 * `MAX_DIM = max(dimA, dimB, dimC)` pivot loop (each matrix idles once exhausted).
 * Same augmented `[V | I]` convention as the single-matrix `invertMatrix`. Used by
 * GATO's Schur kernel (Q_k, Q_kp1, R_k). NumPy: invert each independently.
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
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T *s_memA = s_temp;
    T *s_memB = &s_memA[2*dimA + 1];
    T *s_memC = &s_memB[2*dimB + 1];
    for (unsigned pivRC = 0; pivRC < MAX_DIM; pivRC++) {
        bool AActive = pivRC < dimA;
        bool BActive = pivRC < dimB;
        bool CActive = pivRC < dimC;
        unsigned pivOffA = pivRC * dimA;
        unsigned pivOffB = pivRC * dimB;
        unsigned pivOffC = pivRC * dimC;
        for (unsigned ind = rank; ind < MAX_DIM; ind += size) {
            if (AActive && ind < dimA) s_memA[ind] = A[ind + pivOffA];
            if (BActive && ind < dimB) s_memB[ind] = B[ind + pivOffB];
            if (CActive && ind < dimC) s_memC[ind] = C[ind + pivOffC];
        }
        for (unsigned ind = rank; ind < MAX_DIM + 1; ind += size) {
            if (AActive && ind < dimA + 1) s_memA[ind + dimA] = A[ind*dimA + pivRC + pivOffA];
            if (BActive && ind < dimB + 1) s_memB[ind + dimB] = B[ind*dimB + pivRC + pivOffB];
            if (CActive && ind < dimC + 1) s_memC[ind + dimC] = C[ind*dimC + pivRC + pivOffC];
        }
        __syncthreads();
        for (unsigned ind = rank; ind < MAX_DIM*(MAX_DIM + 1); ind += size) {
            if (AActive && ind < dimA*(dimA + 1)) {
                unsigned row = ind % dimA, col = ind / dimA;
                if (row == pivRC) A[pivOffA + ind] /= s_memA[pivRC];
                else A[pivOffA + ind] -= s_memA[row] / s_memA[pivRC] * s_memA[dimA + col];
            }
            if (BActive && ind < dimB*(dimB + 1)) {
                unsigned row = ind % dimB, col = ind / dimB;
                if (row == pivRC) B[pivOffB + ind] /= s_memB[pivRC];
                else B[pivOffB + ind] -= s_memB[row] / s_memB[pivRC] * s_memB[dimB + col];
            }
            if (CActive && ind < dimC*(dimC + 1)) {
                unsigned row = ind % dimC, col = ind / dimC;
                if (row == pivRC) C[pivOffC + ind] /= s_memC[pivRC];
                else C[pivOffC + ind] -= s_memC[row] / s_memC[pivRC] * s_memC[dimC + col];
            }
        }
        __syncthreads();
    }
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
