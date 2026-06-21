#pragma once
#include <cstdint>
#include "chol_InPlace.cuh"  // warp::cholDecomp_InPlace, composed by warp::posv

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
            T factor = static_cast<T>(0);
            if (lane == 0) { factor = b[col] / L[col*N + col]; b[col] = factor; }
            // broadcast from lane 0's register (not a shared re-read of b[col]) — see the note
            // in warp::cholDecomp_InPlace; immune to the nvcc __restrict__ stale-cache miscompile.
            factor = __shfl_sync(0xffffffffu, factor, 0);
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
            T factor = static_cast<T>(0);
            if (lane == 0) { factor = b[col] / L[col*N + col]; b[col] = factor; }
            // broadcast from lane 0's register (not a shared re-read of b[col]); immune to the
            // nvcc __restrict__ stale-cache miscompile — see warp::cholDecomp_InPlace.
            factor = __shfl_sync(0xffffffffu, factor, 0);
            // eliminate x[col] from rows i < col:  b[i] -= (Lᵀ)_{i,col} x[col] = L_{col,i} x[col]
            for (uint32_t i = lane; i < (uint32_t)col; i += 32)
                b[i] -= L[i*N + col] * factor;
            __syncwarp();
        }
    }

    // ── NEW upper-stored single-warp triangular solves (own warp impls) ──────────
    // These mirror warp::trsm / warp::trsm_transpose but for an UPPER-stored
    // triangular matrix U (column-major; only the upper triangle is read). They are
    // self-contained warp implementations — they do NOT depend on the block trsv:
    // warp and block can't share an impl (`__shfl`/`__syncwarp` vs `__syncthreads`).
    // Every pivot is broadcast from lane 0's REGISTER via `__shfl_sync` (never a
    // shared re-read of b[col]), immune to the nvcc __restrict__ stale-cache
    // miscompile (see warp::cholDecomp_InPlace). The `UNIT` flag skips the diagonal
    // divide (implicit unit diagonal).

    /**
     * @brief Single-warp upper-triangular solve `U x = b` (back substitution), compile-time size.
     *
     * One 32-lane warp solves `U x = b` in place given an upper-triangular `U`
     * (column-major, only the upper triangle read), overwriting `b`. Back
     * substitution from the last unknown. The pivot is broadcast from lane 0's
     * register; no shared scratch, no `__syncthreads`. With `UNIT=true` the diagonal
     * is treated as all ones (not read). SciPy:
     * `x = scipy.linalg.solve_triangular(U, b, lower=False)`.
     *
     * @tparam T     Scalar type.
     * @tparam N     Dimension (U is N x N, b has length N).
     * @tparam UNIT  When true, `U` has an implicit unit diagonal (diagonal not read).
     * @param U  Upper-triangular matrix (column-major).
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N, bool UNIT = false>
    __device__ void trsm_upper(T *U, T *b)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (int32_t col = (int32_t)N - 1; col >= 0; col--) {
            T factor = static_cast<T>(0);
            if (lane == 0) {
                factor = UNIT ? b[col] : (b[col] / U[col*N + col]);
                b[col] = factor;
            }
            factor = __shfl_sync(0xffffffffu, factor, 0);
            // eliminate x[col] from rows i < col:  b[i] -= U_{i,col} x[col]
            for (uint32_t i = lane; i < (uint32_t)col; i += 32)
                b[i] -= U[col*N + i] * factor;
            __syncwarp();
        }
    }

    /**
     * @brief Single-warp transpose upper-triangular solve `Uᵀ x = b` (forward substitution), compile-time size.
     *
     * One 32-lane warp solves `Uᵀ x = b` in place given an upper-triangular `U`
     * (column-major; `Uᵀ` is lower-triangular, used implicitly), overwriting `b`.
     * Forward substitution from the first unknown. The pivot is broadcast from
     * lane 0's register; no shared scratch, no `__syncthreads`. With `UNIT=true` the
     * diagonal is treated as all ones (not read). SciPy:
     * `x = scipy.linalg.solve_triangular(U.T, b, lower=True)`.
     *
     * @tparam T     Scalar type.
     * @tparam N     Dimension (U is N x N, b has length N).
     * @tparam UNIT  When true, `U` has an implicit unit diagonal (diagonal not read).
     * @param U  Upper-triangular matrix (column-major); `Uᵀ` is used implicitly.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N, bool UNIT = false>
    __device__ void trsm_upper_transpose(T *U, T *b)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t col = 0; col < N; col++) {
            T factor = static_cast<T>(0);
            if (lane == 0) {
                factor = UNIT ? b[col] : (b[col] / U[col*N + col]);
                b[col] = factor;
            }
            factor = __shfl_sync(0xffffffffu, factor, 0);
            // eliminate x[col] from rows i > col:  b[i] -= (Uᵀ)_{i,col} x[col] = U_{col,i} x[col]
            for (uint32_t i = lane + col + 1; i < N; i += 32)
                b[i] -= U[i*N + col] * factor;
            __syncwarp();
        }
    }

    // ── unit-diagonal lower variants (the existing warp::trsm/trsm_transpose are
    //    non-unit only); needed so trsv can offer the full {LOWER,UNIT,TRANS} matrix.
    /**
     * @brief Single-warp unit-lower solve `L x = b` (forward substitution), compile-time size.
     *
     * Like `warp::trsm` but with an implicit unit diagonal (the diagonal of `L` is
     * not read). One 32-lane warp, pivot broadcast from a register, no shared
     * scratch, no `__syncthreads`. SciPy:
     * `x = scipy.linalg.solve_triangular(L, b, lower=True, unit_diagonal=True)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (L is N x N, b has length N).
     * @param L  Lower-triangular matrix (column-major), unit diagonal assumed.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void trsm_unit(T *L, T *b)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t col = 0; col < N; col++) {
            // unit diag: x[col] = b[col] (no divide). Read on lane 0's register then
            // broadcast (§1g), never a shared re-read on the consuming lanes.
            T factor = static_cast<T>(0);
            if (lane == 0) factor = b[col];
            factor = __shfl_sync(0xffffffffu, factor, 0);
            for (uint32_t row = lane + col + 1; row < N; row += 32)
                b[row] -= L[col*N + row] * factor;
            __syncwarp();
        }
    }

    /**
     * @brief Single-warp unit-lower transpose solve `Lᵀ x = b` (back substitution), compile-time size.
     *
     * Like `warp::trsm_transpose` but with an implicit unit diagonal. One 32-lane
     * warp, pivot broadcast from a register, no shared scratch, no `__syncthreads`.
     * SciPy: `x = scipy.linalg.solve_triangular(L.T, b, lower=False, unit_diagonal=True)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (L is N x N, b has length N).
     * @param L  Lower-triangular matrix (column-major), unit diagonal assumed; `Lᵀ` used implicitly.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void trsm_transpose_unit(T *L, T *b)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (int32_t col = (int32_t)N - 1; col >= 0; col--) {
            // unit diag: x[col] = b[col] (no divide). Broadcast from lane 0's
            // register (§1g), never a shared re-read on the consuming lanes.
            T factor = static_cast<T>(0);
            if (lane == 0) factor = b[col];
            factor = __shfl_sync(0xffffffffu, factor, 0);
            for (uint32_t i = lane; i < (uint32_t)col; i += 32)
                b[i] -= L[i*N + col] * factor;
            __syncwarp();
        }
    }

    /**
     * @brief Single-warp triangular solve `op(A) x = b` (TRSV), flagged dispatch, compile-time size.
     *
     * Thin compile-time-flagged wrapper selecting the right single-warp triangular
     * solve for any `{LOWER, UNIT, TRANS}` combination, dispatching to the lower
     * `warp::trsm`/`warp::trsm_transpose` (and their unit-diagonal twins) or the
     * upper `warp::trsm_upper`/`warp::trsm_upper_transpose`. Solves in place
     * (`b` overwritten with `x`); `A` is column-major and only the named triangle is
     * read. One 32-lane warp, no shared scratch, no `__syncthreads`; this is its OWN
     * warp implementation and does not share an impl with the block trsv. SciPy:
     * `x = scipy.linalg.solve_triangular(A, b, lower=LOWER, trans=TRANS, unit_diagonal=UNIT)`.
     *
     * @tparam T      Scalar type.
     * @tparam N      Dimension (A is N x N, b has length N).
     * @tparam LOWER  When true, `A` is lower-triangular (default true); else upper.
     * @tparam UNIT   When true, `A` has an implicit unit diagonal (default false).
     * @tparam TRANS  When true, solve `Aᵀ x = b` instead of `A x = b` (default false).
     * @param A  Triangular matrix (column-major); only the `LOWER`/upper triangle read.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N, bool LOWER = true, bool UNIT = false, bool TRANS = false>
    __device__ void trsv(T *A, T *b)
    {
        if (LOWER) {
            if (!TRANS) { if (UNIT) trsm_unit<T, N>(A, b);           else trsm<T, N>(A, b); }
            else        { if (UNIT) trsm_transpose_unit<T, N>(A, b); else trsm_transpose<T, N>(A, b); }
        } else {
            if (!TRANS) trsm_upper<T, N, UNIT>(A, b);
            else        trsm_upper_transpose<T, N, UNIT>(A, b);
        }
    }

    /**
     * @brief Single-warp SPD solve `A x = b` via Cholesky (LAPACK posv), compile-time size.
     *
     * One 32-lane warp solves the symmetric-positive-definite system `A x = b` in
     * place: it factors `A = L Lᵀ` with `warp::cholDecomp_InPlace` (lower triangle
     * overwrites `A`), then a forward solve `L y = b` (`trsv<…,LOWER,!TRANS>`) and a
     * back solve `Lᵀ x = y` (`trsv<…,LOWER,TRANS>`). On return `b` holds `x` and the
     * lower triangle of `A` holds `L`. This is the composed warp-per-problem solve —
     * the proof that the warp L1/L2/L3 glue closes the gap. No shared scratch, no
     * `__syncthreads`; every pivot broadcast from a register (§1g). `A` must be SPD
     * (use `double` for ill-conditioned systems). NumPy equivalent:
     * `x = np.linalg.solve(A, b)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (A is N x N, b has length N).
     * @param A  In/out SPD matrix (column-major); on return its lower triangle holds L.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void posv(T *A, T *b)
    {
        cholDecomp_InPlace<T, N>(A);
        trsv<T, N, /*LOWER=*/true, /*UNIT=*/false, /*TRANS=*/false>(A, b);  // forward: L y = b
        trsv<T, N, /*LOWER=*/true, /*UNIT=*/false, /*TRANS=*/true>(A, b);   // back:   Lᵀ x = y
    }
}
