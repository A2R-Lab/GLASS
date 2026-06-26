#pragma once
#include "../barrier.cuh"
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
 * @param s_scratch  Shared scratch of `(2*dimA + 1) * sizeof(T)` bytes.
 */
// Gauss-Jordan inversion of an augmented dimA×(2*dimA) matrix in-place.
// Expected layout: column-major [A | I]; on return columns dimA..2*dimA-1 hold A^-1.
// s_scratch: (2*dimA+1)*sizeof(T) bytes.
// Shared body: Gauss-Jordan inversion; barrier policy supplies rank/size + the
// two per-pivot syncs, shared by the glass:: and cgrps:: surfaces.
template <typename Bar, typename T>
__device__ void invertMatrix_impl(Bar bar, uint32_t dimA, T *A, T *s_scratch)
{
    uint32_t rank = bar.rank(), size = bar.size();
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivOff = pivRC * dimA;
        T pvInv = static_cast<T>(1) / A[pivRC + pivOff];
        for (unsigned ind = rank; ind < 2*dimA+1; ind += size) {
            unsigned AInd = (ind < dimA) ? (ind + pivOff) : (pivRC + pivOff + (ind-dimA)*dimA);
            s_scratch[ind] = A[AInd];
        }
        bar.sync();
        for (unsigned ind = rank; ind < dimA*(dimA+1); ind += size) {
            unsigned row = ind % dimA, col = ind / dimA, coff = ind - row;
            if (row == pivRC) A[row + pivOff + coff] *= pvInv;
            else A[row + pivOff + coff] -= s_scratch[row]*pvInv*s_scratch[dimA+col];
        }
        bar.sync();
    }
}

template <typename T>
__device__ void invertMatrix(uint32_t dimA, T *A, T *s_scratch)
{
    invertMatrix_impl<BlockBarrier, T>(BlockBarrier{}, dimA, A, s_scratch);
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
 * @param s_scratch  Shared scratch of `(2*N + 1) * sizeof(T)` bytes.
 */
template <typename T, uint32_t N>
__device__ void invertMatrix(T *A, T *s_scratch)
{
    invertMatrix<T>(N, A, s_scratch);
}

/**
 * @brief Scratch size in bytes for `invertMatrix` (augmented `[A | I]`).
 *
 * The unpivoted Gauss-Jordan path saves the active pivot column + row window plus
 * one slot: `2*dimA + 1` elements of `T`. Allocate
 * `invertMatrix_scratch_bytes<T>(dimA)` for the `s_scratch` argument.
 *
 * @tparam T  Scalar type.
 * @param dimA  Matrix dimension (A is dimA x dimA).
 * @return Bytes to allocate for `invertMatrix`'s `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t invertMatrix_scratch_bytes(uint32_t dimA)
{
    return static_cast<std::size_t>(2 * dimA + 1) * sizeof(T);
}

/**
 * @brief Scratch size in bytes for `invertMatrix_pivoted`.
 *
 * Row pivoting permutes the already-built inverse columns, so (unlike the
 * unpivoted path) the elimination cannot use the reduced active-column window —
 * it must save and update the **full** `2*dimA`-wide pivot row. Layout:
 * `dimA` slots for the pivot column, `2*dimA` for the full pivot row, and **one**
 * trailing slot to broadcast the chosen pivot-row index from the argmax.
 *
 * Total = `3*dimA + 1` elements of `T`.
 *
 * @tparam T  Scalar type.
 * @param dimA  Matrix dimension (A is dimA x dimA).
 * @return Bytes to allocate for `invertMatrix_pivoted`'s `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t invertMatrix_pivoted_scratch_bytes(uint32_t dimA)
{
    return static_cast<std::size_t>(3 * dimA + 1) * sizeof(T);
}

/**
 * @brief In-place ROBUST (partial-pivoting) matrix inverse, augmented `[A | I]`.
 *
 * Partial-pivoting (row-pivoted) Gauss-Jordan sibling of `invertMatrix`. Same
 * augmented column-major `dimA x (2*dimA)` `[A | I]` input/output contract: on
 * return columns `dimA..2*dimA-1` hold `A^-1`. NumPy equivalent:
 * `Ainv = np.linalg.inv(A)`.
 *
 * @par Why pivoted
 * The plain `invertMatrix` divides by `A[pivRC + pivRC*dimA]` as-is, so it loses
 * accuracy (or fails outright) when a leading pivot is small/zero even though `A`
 * is invertible — e.g. a tiny `A[0,0]` beneath a large later row, or a row
 * permutation that parks a zero on the diagonal. At each step `k` this variant
 * instead selects the row `r >= k` with the largest `|A[r + k*dimA]|`, swaps rows
 * `k` and `r` across the FULL augmented width `[A | I]` (all `2*dimA` columns),
 * then does the identical divide/eliminate. Because both halves are permuted
 * together, the row permutation is absorbed into the result and the returned
 * right half is `A^-1` directly — **no separate `piv` output is needed**.
 *
 * @par Parallelism / barriers
 * Single-block, strided (`ind += size`), thread-count invariant. Per pivot step:
 *   1. save the pivot column + pivot row + augmented column into `s_scratch`;
 *   2. block-wide argmax over the pivot-column tail `[k, dimA)` (deterministic
 *      lower-index tie-break, matching `glass::iamax`) chooses the pivot row `r`,
 *      broadcast to all threads through a shared slot + `__syncthreads()` (never
 *      a racy re-read);
 *   3. if `r != k`, swap rows `k` and `r` across all `2*dimA` columns (parallel);
 *   4. re-save the (now swapped) pivot row/column, then divide/eliminate exactly
 *      as the unpivoted path.
 * Each phase is separated by a barrier so the argmax result and the swapped data
 * are block-visible before they are consumed.
 *
 * @par Scratch
 * `s_scratch` must hold `invertMatrix_pivoted_scratch_bytes<T>(dimA)` = `3*dimA + 1`
 * elements of `T`: `dimA` for the pivot column, `2*dimA` for the full pivot row,
 * and one trailing slot to broadcast the chosen pivot-row index.
 *
 * @tparam T  Scalar type.
 * @param dimA    Matrix dimension (A is dimA x dimA).
 * @param A       In/out augmented `[A | I]` buffer (column-major, dimA x 2*dimA);
 *                on return its right half holds `A^-1`.
 * @param s_scratch  Shared scratch of `(3*dimA + 1) * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix_pivoted(uint32_t dimA, T *A, T *s_scratch)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    const unsigned W = 2*dimA;          // augmented width
    // s_scratch layout (3*dimA + 1 slots):
    //   [0 .. dimA)       pivot column
    //   [dimA .. 3*dimA)  full pivot row (all 2*dimA augmented columns)
    //   [3*dimA]          broadcast slot for the chosen pivot-row index
    T *s_pcol = s_scratch;
    T *s_prow = &s_scratch[dimA];
    T *s_piv  = &s_scratch[3*dimA];
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivOff = pivRC * dimA;

        // ── Partial pivot: argmax over |A[r + pivOff]| for r in [pivRC, dimA). ──
        // Thread 0 scans the (short) pivot-column tail with a deterministic
        // lower-index tie-break (matching glass::iamax) and broadcasts the chosen
        // row through s_piv; the broadcast + __syncthreads makes it race-free.
        if (rank == 0) {
            T key = static_cast<T>(0);
            uint32_t idx = UINT32_MAX;
            for (unsigned r = pivRC; r < dimA; r++) {
                T v = A[r + pivOff];
                v = (v < static_cast<T>(0)) ? -v : v;
                if (v > key || (v == key && r < idx)) { key = v; idx = r; }
            }
            // All-zero column (singular) → keep the diagonal row (idx==MAX → pivRC).
            s_piv[0] = static_cast<T>((idx == UINT32_MAX) ? pivRC : idx);
        }
        __syncthreads();
        unsigned pivRow = static_cast<unsigned>(s_piv[0]);

        // ── Swap rows pivRC and pivRow across the full augmented width [A|I]. ──
        if (pivRow != pivRC) {
            for (unsigned col = rank; col < W; col += size) {
                unsigned coff = col * dimA;
                T tmp = A[pivRC + coff];
                A[pivRC + coff] = A[pivRow + coff];
                A[pivRow + coff] = tmp;
            }
            __syncthreads();
        }

        // ── Save the pivot column and the FULL pivot row, then divide/eliminate
        //    over all 2*dimA columns (row pivoting forbids the reduced window). ──
        for (unsigned r = rank; r < dimA; r += size) s_pcol[r] = A[r + pivOff];
        for (unsigned c = rank; c < W;    c += size) s_prow[c] = A[pivRC + c*dimA];
        __syncthreads();
        T pvInv = static_cast<T>(1) / s_prow[pivRC];
        for (unsigned ind = rank; ind < dimA*W; ind += size) {
            unsigned row = ind % dimA, col = ind / dimA;
            if (row == pivRC) A[row + col*dimA]  = s_prow[col] * pvInv;
            else              A[row + col*dimA] -= s_pcol[row] * pvInv * s_prow[col];
        }
        __syncthreads();
    }
}

/**
 * @brief Compile-time-size ROBUST (partial-pivoting) matrix inverse (`[A | I]`).
 *
 * Same as the runtime `invertMatrix_pivoted` but with the dimension as a template
 * parameter; partial-pivoting Gauss-Jordan, tolerant of small leading pivots that
 * the plain `invertMatrix` mishandles. NumPy equivalent: `Ainv = np.linalg.inv(A)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Matrix dimension (A is N x N).
 * @param A       In/out augmented `[A | I]` buffer (column-major, N x 2*N);
 *                on return its right half holds `A^-1`.
 * @param s_scratch  Shared scratch of `(3*N + 1) * sizeof(T)` bytes
 *                (= `invertMatrix_pivoted_scratch_bytes<T>(N)`).
 */
template <typename T, uint32_t N>
__device__ void invertMatrix_pivoted(T *A, T *s_scratch)
{
    invertMatrix_pivoted<T>(N, A, s_scratch);
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
 * `[Σ_{j<m}(2*dims[j]+1), Σ_{j<=m}(2*dims[j]+1))` of `s_scratch` (each matrix needs
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
 * @param s_scratch   Shared scratch of `(Σ_m (2*dims[m]+1)) * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix(uint32_t K, const uint32_t *dims, uint32_t MAX_DIM, T **mats, T *s_scratch)
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
                T *s_mem = &s_scratch[sOff];
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
                T *s_mem = &s_scratch[sOff];
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
 * @param s_scratch     Shared scratch of `(2*dimA + 2*dimB + 2) * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix(uint32_t dimA, uint32_t dimB, uint32_t MAX_DIM, T *A, T *B, T *s_scratch)
{
    uint32_t dims[2] = {dimA, dimB};
    T *mats[2] = {A, B};
    invertMatrix<T>(2, dims, MAX_DIM, mats, s_scratch);
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
 * @param s_scratch          Shared scratch of `(2*dimA + 2*dimB + 2*dimC + 3) * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix(uint32_t dimA, uint32_t dimB, uint32_t dimC, uint32_t MAX_DIM, T *A, T *B, T *C, T *s_scratch)
{
    uint32_t dims[3] = {dimA, dimB, dimC};
    T *mats[3] = {A, B, C};
    invertMatrix<T>(3, dims, MAX_DIM, mats, s_scratch);
}


/**
 * @brief Scratch size in bytes for `invertMatrix_dense`.
 *
 * The dual-buffer dense path saves one `3*dimA`-element pivot working set. Allocate
 * `invertMatrix_dense_scratch_bytes<T>(dimA)` for the `s_scratch` argument.
 *
 * @tparam T  Scalar type.
 * @param dimA  Matrix dimension (A is dimA x dimA).
 * @return Bytes to allocate for `invertMatrix_dense`'s `s_scratch`.
 */
template <typename T>
__host__ __device__ constexpr std::size_t invertMatrix_dense_scratch_bytes(uint32_t dimA)
{
    return static_cast<std::size_t>(3 * dimA) * sizeof(T);
}

// Block-cooperative Gauss-Jordan inversion of a dimA×dimA matrix.
//   A:    in/out, column-major dimA×dimA.  On return: A := A^{-1}.
//   Ainv: workspace, column-major dimA×dimA (overwritten).  On return: A^{-1}.
//          (Some callers want A unchanged + a separate inverse — they get both.)
//   s_scratch: shared scratch of size (3*dimA)*sizeof(T).
// Pivot loop is serial (pivot-to-pivot data dependency); within each pivot, the
// dimA save and the dimA*dimA cell update are parallelized across the block.
//
// Algorithm: same dual-update (A, Ainv) Gauss-Jordan as a classic [A | I] →
// [I | A^{-1}] reduction but without materializing the augmented layout. A and
// Ainv are tracked in separate buffers so callers that need A^{-1} alongside
// the original A get it without extra copies.

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
 * @param s_scratch  Shared scratch of `3 * dimA * sizeof(T)` bytes.
 */
template <typename T>
__device__ void invertMatrix_dense(uint32_t dimA, T *A, T *Ainv, T *s_scratch)
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
            s_scratch[ind]            = A[pivRC + dimA * ind];          // pivot row of A
            s_scratch[ind + dimA]     = Ainv[pivRC + dimA * ind];        // pivot row of Ainv
            s_scratch[ind + dimA * 2] = A[ind + pivColOffset];           // pivot column of A
        }
        __syncthreads();
        T pvInv = static_cast<T>(1) / s_scratch[pivRC];
        // Block-cooperative Gauss-Jordan update across dimA*dimA cells.
        // Each thread owns a unique (row, col) so writes are race-free.
        for (uint32_t ind = rank; ind < dimA * dimA; ind += size) {
            uint32_t row = ind % dimA, col = ind / dimA;
            if (row == pivRC) {
                A[row + dimA * col]    = s_scratch[col] * pvInv;
                Ainv[row + dimA * col] = s_scratch[col + dimA] * pvInv;
            } else {
                T multiplier = s_scratch[row + dimA * 2] * pvInv;
                A[row + dimA * col]    -= multiplier * s_scratch[col];
                Ainv[row + dimA * col] -= multiplier * s_scratch[col + dimA];
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
 * @param s_scratch  Shared scratch of `3 * N * sizeof(T)` bytes.
 */
template <typename T, uint32_t N>
__device__ void invertMatrix_dense(T *A, T *Ainv, T *s_scratch)
{
    invertMatrix_dense<T>(N, A, Ainv, s_scratch);
}
