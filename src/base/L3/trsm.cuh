#pragma once
#include "../barrier.cuh"
#include "../flags.cuh"   // FillMode / Diag
#include <cstdint>
#include "potrf.cuh"  // warp::potrf, composed by warp::posv

// ─────────────────────────────────────────────────────────────────────────────
// trsm — triangular solve with multiple right-hand sides, op(A) X = B, in place
// (BLAS TRSM, left side, alpha = 1). B is n×nrhs column-major and is
// overwritten with X. Storage and flag semantics match trsv (see trsv.cuh):
// FILL names the stored triangle, DIAG the implicit-unit choice, TRANSPOSE
// solves op(A) = Aᵀ against that same stored triangle.
//
// Parallelism: each elimination step resolves all `nrhs` pivots in parallel
// (thread-strided over columns), then updates the remaining (row, rhs) cells
// flat-strided over the whole rectangle — so wide B keeps every thread busy
// even for small n. Two barriers per step, shared across all right-hand sides
// (vs 2·n·nrhs for nrhs separate trsv calls).
// ─────────────────────────────────────────────────────────────────────────────

// Shared body: barrier policy supplies rank/size + the per-step syncs, shared
// by glass:: and cgrps::. SizeT/SizeU are deduced: uint32_t from the runtime
// overload, ct_size<N>/ct_size<NRHS> from the compile-time overload
// (constant-folds the trip counts and the flat-index %/ by `rows`).
template <typename Bar, typename T, FillMode FILL, Diag DIAG, bool TRANSPOSE,
          typename SizeT, typename SizeU>
__device__ void trsm_impl(Bar bar, SizeT n, SizeU nrhs, const T *A, T *B)
{
    static_assert(FILL != FillMode::Full, "trsm: FILL must name a triangle (Lower or Upper)");
    constexpr bool LOWER   = (FILL == FillMode::Lower);
    constexpr bool UNIT    = (DIAG == Diag::Unit);
    constexpr bool FORWARD = (LOWER != TRANSPOSE);   // op(A) lower ⇒ forward sweep
    uint32_t rank = bar.rank(), size = bar.size();
    for (uint32_t step = 0; step < n; step++) {
        uint32_t k = FORWARD ? step : (n - 1 - step);
        // resolve the pivot row across all right-hand sides: B[k,c] /= op(A)[k][k]
        if (!UNIT) {
            for (uint32_t c = rank; c < nrhs; c += size)
                B[k + c * n] /= A[k + k * n];
            bar.sync();
        }
        // eliminate x[k] from the remaining unknowns of every column:
        //   op(A)[i][k] = TRANSPOSE ? A[k + i*n] : A[i + k*n]
        uint32_t rows = FORWARD ? (n - 1 - k) : k;       // unknowns still open
        for (uint32_t flat = rank; flat < rows * nrhs; flat += size) {
            uint32_t i = FORWARD ? (k + 1 + flat % rows) : (flat % rows);
            uint32_t c = flat / rows;
            B[i + c * n] -= (TRANSPOSE ? A[k + i * n] : A[i + k * n]) * B[k + c * n];
        }
        bar.sync();
    }
}

/**
 * @brief Triangular solve with multiple right-hand sides `op(A) X = B`, in place (TRSM).
 *
 * Solves the triangular system for every column of `B` (n×nrhs, column-major),
 * overwriting `B` with `X`. `A` is `n×n` column-major; only the triangle named
 * by `FILL` is read. `TRANSPOSE=true` solves `Aᵀ X = B` against that same
 * stored triangle; `DIAG=Diag::Unit` means an implicit unit diagonal. All
 * right-hand sides share each elimination step's two barriers, and the update
 * is flat-strided over the (rows × nrhs) rectangle, so wide `B` keeps every
 * thread busy even at small `n`. Ends on a barrier (composes cleanly).
 * SciPy equivalent:
 * `X = scipy.linalg.solve_triangular(A, B, lower=(FILL==Lower), unit_diagonal=(DIAG==Unit), trans=(1 if TRANSPOSE else 0))`.
 *
 * @tparam T     Scalar type.
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true solve `Aᵀ X = B` (default false).
 * @param n     Dimension (`A` is `n×n`; each column of `B` has length `n`).
 * @param nrhs  Number of right-hand sides (columns of `B`).
 * @param A     Triangular matrix (column-major; read-only).
 * @param B     In/out right-hand sides (`n×nrhs`, column-major); on return holds `X`.
 */
template <typename T, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trsm(uint32_t n, uint32_t nrhs, const T *A, T *B)
{
    trsm_impl<BlockBarrier, T, FILL, DIAG, TRANSPOSE>(BlockBarrier{}, n, nrhs, A, B);
}

/**
 * @brief Triangular solve with multiple right-hand sides `op(A) X = B`, in place (TRSM), compile-time size.
 *
 * Same as the runtime `trsm` but with the dimensions as template parameters.
 * SciPy equivalent:
 * `X = scipy.linalg.solve_triangular(A, B, lower=(FILL==Lower), unit_diagonal=(DIAG==Unit), trans=(1 if TRANSPOSE else 0))`.
 *
 * @tparam T     Scalar type.
 * @tparam N     Dimension (`A` is `N×N`; each column of `B` has length `N`).
 * @tparam NRHS  Number of right-hand sides (columns of `B`).
 * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
 * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
 * @tparam TRANSPOSE  When true solve `Aᵀ X = B` (default false).
 * @param A  Triangular matrix (column-major; read-only).
 * @param B  In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
 */
template <typename T, uint32_t N, uint32_t NRHS,
          FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
__device__ void trsm(const T *A, T *B)
{
    trsm_impl<BlockBarrier, T, FILL, DIAG, TRANSPOSE>(BlockBarrier{}, ct_size<N>{}, ct_size<NRHS>{}, A, B);
}

namespace thread {
    /**
     * @brief Single-thread triangular solve with multiple right-hand sides
     *        `op(A) X = B`, in place (TRSM), compile-time size.
     *
     * ONE thread solves the `N×N` triangular system for all `NRHS` columns of
     * `B` (`N×NRHS`, column-major), overwriting `B` with `X` — for
     * thread-per-problem solvers that pack 32 independent low-DOF problems into
     * a warp. `A` is column-major and read-only; only the triangle named by
     * `FILL` is read; `TRANSPOSE=true` solves `Aᵀ X = B` against that same
     * stored triangle; `DIAG=Diag::Unit` skips the diagonal divide. No shared
     * scratch, no barriers, no `threadIdx` read; operands may be thread-local
     * register arrays. SciPy equivalent:
     * `X = scipy.linalg.solve_triangular(A, B, lower=(FILL==Lower), unit_diagonal=(DIAG==Unit), trans=(1 if TRANSPOSE else 0))`.
     *
     * Delegates to the same `trsm_impl` body the block surface uses, via
     * `ThreadBarrier` (rank=0, size=1, no-op sync) — the same algorithm and
     * operand order as `glass::trsm<T, N, NRHS, …>` on one thread, agreeing to a
     * few ULP (FMA-contraction jitter; bit-identity across the two
     * instantiations is NOT guaranteed — see test/test_thread.py).
     *
     * @tparam T     Scalar type.
     * @tparam N     Dimension (`A` is `N×N`; each column of `B` has length `N`). N<=7 keeps a
     *               `T[N*N]` operand register-resident (measured ceiling, both dtypes — see the
     *               thread-tier constraints in CLAUDE.md); larger N — or a `B` wider than that
     *               element budget — still computes correctly but spills to local memory,
     *               forfeiting the tier's premise.
     * @tparam NRHS  Number of right-hand sides (columns of `B`).
     * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
     * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
     * @tparam TRANSPOSE  When true solve `Aᵀ X = B` (default false).
     * @param A  Triangular matrix (column-major, `N*N`; read-only).
     * @param B  In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
     */
    template <typename T, uint32_t N, uint32_t NRHS,
              FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
    __device__ void trsm(const T *A, T *B)
    {
        trsm_impl<ThreadBarrier, T, FILL, DIAG, TRANSPOSE>(ThreadBarrier{}, ct_size<N>{}, ct_size<NRHS>{}, A, B);
    }
}

namespace warp {
    /**
     * @brief Single-warp triangular solve `op(A) x = b` in place (TRSV), compile-time size.
     *
     * One 32-lane warp solves the triangular system for any `{FILL, DIAG,
     * TRANSPOSE}` combination, overwriting `b` with `x`. `A` is column-major and
     * only the triangle named by `FILL` is read; `TRANSPOSE=true` solves
     * `Aᵀx = b` against that same stored triangle; `DIAG=Diag::Unit` skips the
     * diagonal divide. Every pivot is broadcast from lane 0's REGISTER via
     * `__shfl_sync` (never a shared re-read of `b[k]`) — immune to the nvcc
     * `__restrict__` stale-cache miscompile (see `warp::potrf`). This is its OWN
     * warp implementation (warp and block can't share an impl:
     * `__shfl`/`__syncwarp` vs `__syncthreads`). No shared scratch, no
     * `__syncthreads`. SciPy:
     * `x = scipy.linalg.solve_triangular(A, b, lower=(FILL==Lower), unit_diagonal=(DIAG==Unit), trans=(1 if TRANSPOSE else 0))`.
     *
     * @tparam T     Scalar type.
     * @tparam N     Dimension (`A` is `N×N`, `b` has length `N`).
     * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
     * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
     * @tparam TRANSPOSE  When true solve `Aᵀx = b` (default false).
     * @param A  Triangular matrix (column-major); only the `FILL` triangle read.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N, FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
    __device__ void trsv(const T *A, T *b)
    {
        static_assert(FILL != FillMode::Full, "warp::trsv: FILL must name a triangle (Lower or Upper)");
        constexpr bool LOWER   = (FILL == FillMode::Lower);
        constexpr bool UNIT    = (DIAG == Diag::Unit);
        constexpr bool FORWARD = (LOWER != TRANSPOSE);   // op(A) lower ⇒ forward sweep
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t step = 0; step < N; step++) {
            uint32_t k = FORWARD ? step : (N - 1 - step);
            // resolve pivot on lane 0's register, then broadcast (§1g) — never a
            // shared re-read of b[k] on the consuming lanes.
            T factor = static_cast<T>(0);
            if (lane == 0) {
                factor = UNIT ? b[k] : (b[k] / A[k + k * N]);
                b[k] = factor;
            }
            factor = __shfl_sync(0xffffffffu, factor, 0);
            // eliminate x[k]: op(A)[i][k] = TRANSPOSE ? A[k + i*N] : A[i + k*N]
            if constexpr (FORWARD) {
                for (uint32_t i = lane + k + 1; i < N; i += 32)
                    b[i] -= (TRANSPOSE ? A[k + i * N] : A[i + k * N]) * factor;
            } else {
                for (uint32_t i = lane; i < k; i += 32)
                    b[i] -= (TRANSPOSE ? A[k + i * N] : A[i + k * N]) * factor;
            }
            __syncwarp();
        }
    }

    /**
     * @brief Single-warp triangular solve with multiple right-hand sides `op(A) X = B` (TRSM), compile-time size.
     *
     * Warp-per-problem parity with the block `glass::trsm`: one 32-lane warp
     * solves all `NRHS` columns of `B` (`N×NRHS`, column-major) in place. Each
     * elimination step resolves the pivot row across all columns (lane-strided)
     * and flat-strides the update over the (rows × NRHS) rectangle, sharing the
     * per-step `__syncwarp()` across every right-hand side. No shared scratch,
     * no `__syncthreads`.
     *
     * @tparam T     Scalar type.
     * @tparam N     Dimension (`A` is `N×N`; each column of `B` has length `N`).
     * @tparam NRHS  Number of right-hand sides (columns of `B`).
     * @tparam FILL  Which triangle of `A` holds the data (default `FillMode::Lower`).
     * @tparam DIAG  `Diag::Unit` for an implicit unit diagonal (default `Diag::NonUnit`).
     * @tparam TRANSPOSE  When true solve `Aᵀ X = B` (default false).
     * @param A  Triangular matrix (column-major; read-only).
     * @param B  In/out right-hand sides (`N×NRHS`, column-major); on return holds `X`.
     */
    template <typename T, uint32_t N, uint32_t NRHS,
              FillMode FILL = FillMode::Lower, Diag DIAG = Diag::NonUnit, bool TRANSPOSE = false>
    __device__ void trsm(const T *A, T *B)
    {
        static_assert(FILL != FillMode::Full, "warp::trsm: FILL must name a triangle (Lower or Upper)");
        constexpr bool LOWER   = (FILL == FillMode::Lower);
        constexpr bool UNIT    = (DIAG == Diag::Unit);
        constexpr bool FORWARD = (LOWER != TRANSPOSE);
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t step = 0; step < N; step++) {
            uint32_t k = FORWARD ? step : (N - 1 - step);
            if constexpr (!UNIT) {
                for (uint32_t c = lane; c < NRHS; c += 32)
                    B[k + c * N] /= A[k + k * N];
                __syncwarp();
            }
            uint32_t rows = FORWARD ? (N - 1 - k) : k;
            for (uint32_t flat = lane; flat < rows * NRHS; flat += 32) {
                uint32_t i = FORWARD ? (k + 1 + flat % rows) : (flat % rows);
                uint32_t c = flat / rows;
                B[i + c * N] -= (TRANSPOSE ? A[k + i * N] : A[i + k * N]) * B[k + c * N];
            }
            __syncwarp();
        }
    }

    /**
     * @brief Single-warp SPD solve `A x = b` via Cholesky (LAPACK posv), compile-time size.
     *
     * One 32-lane warp solves the symmetric-positive-definite system `A x = b` in
     * place: it factors `A = L Lᵀ` with `warp::potrf` (lower triangle
     * overwrites `A`), then a forward solve `L y = b` and a back solve `Lᵀ x = y`
     * (both `warp::trsv`). On return `b` holds `x` and the lower triangle of `A`
     * holds `L`. This is the composed warp-per-problem solve — the proof that the
     * warp L1/L2/L3 glue closes the gap. No shared scratch, no `__syncthreads`;
     * every pivot broadcast from a register (§1g). `A` must be SPD (use `double`
     * for ill-conditioned systems). NumPy equivalent: `x = np.linalg.solve(A, b)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (A is N x N, b has length N).
     * @param A  In/out SPD matrix (column-major); on return its lower triangle holds L.
     * @param b  In/out right-hand side; on return holds the solution x.
     */
    template <typename T, uint32_t N>
    __device__ void posv(T *A, T *b)
    {
        potrf<T, N>(A);
        trsv<T, N>(A, b);                                                        // forward: L y = b
        trsv<T, N, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(A, b);    // back:   Lᵀ x = y
    }

    /**
     * @brief Single-warp SPD solve from a precomputed Cholesky factor (LAPACK potrs), compile-time size.
     *
     * Given the lower factor `L` (e.g. from `warp::potrf`), one 32-lane warp
     * solves `L Lᵀ x = b` by forward then back substitution — the same two
     * `warp::trsv` legs `warp::posv` composes, without the re-factor: the
     * reusable-factor / multi-solve path. `L` is read-only; `b` is overwritten
     * with `x`. No shared scratch, no `__syncthreads`; every pivot is broadcast
     * from lane 0's register via `__shfl_sync` (§1g — immune to the
     * `__restrict__` stale-shared-reread miscompile, see `warp::trsv`). SciPy
     * equivalent: `x = scipy.linalg.cho_solve((L, True), b)`.
     *
     * @tparam T  Scalar type.
     * @tparam N  Dimension (`L` is `N×N`, `b` has length `N`).
     * @param L  Lower Cholesky factor (column-major, `N*N`; read-only).
     * @param b  In/out right-hand side; on return holds the solution `x`.
     */
    template <typename T, uint32_t N>
    __device__ void potrs(const T *L, T *b)
    {
        trsv<T, N>(L, b);                                                        // forward: L y = b
        trsv<T, N, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(L, b);    // back:   Lᵀ x = y
    }

    /**
     * @brief Add a diagonal regularization shift to A in place (single-warp helper).
     *
     * Lane-strided over the `n` diagonal entries; `REG_DIAG=false` adds `rho·I`
     * (Marquardt), `REG_DIAG=true` adds `rho·diag(A)` (Levenberg). Trailing
     * `__syncwarp()` so the shifted A is warp-visible before factoring. Internal.
     */
    template <typename T, bool REG_DIAG = false, typename SizeT>
    __device__ void _posv_regularize(SizeT n, T *A, T rho)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t i = lane; i < n; i += 32) {
            if constexpr (REG_DIAG) A[i*n + i] += rho * A[i*n + i];   // rho*diag(A)
            else                    A[i*n + i] += rho;                // rho*I
        }
        __syncwarp();
    }

    /**
     * @brief Single-warp regularized/checked multi-RHS SPD solve `A X = B` (LAPACK posv).
     *
     * Warp-per-problem parity with the block multi-RHS `glass::posv`: one 32-lane
     * warp optionally shifts `A`'s diagonal (`REGULARIZE`: `rho·I`, or `rho·diag(A)`
     * when `REG_DIAG`), factors `A = L Lᵀ` via `warp::potrf<…,CHECK>`
     * (reporting a non-PD pivot through `s_fail`), then forward/back-solves all
     * `NRHS` columns of `B` at once with the multi-RHS `warp::trsm`. On return
     * `A` holds `L` and `B` holds `X`. No shared scratch, no `__syncthreads`.
     *
     * A flagged **single**-RHS solve is just NRHS=1 — the form HJCD's LM step wants:
     * `warp::posv<T, DIM, 1, REGULARIZE=true, CHECK=true, REG_DIAG=true>(A, b, lambda, &s_fail)`
     * folds the `A += lambda*diag(A)` damping and the non-PD net into one call. (The
     * unflagged 2-arg `warp::posv<T,N>(A,b)` stays; flags cannot live on it without
     * colliding with this overload at NRHS in {0,1}.)
     *
     * @tparam T     Scalar type (use `double` for ill-conditioned A).
     * @tparam N     Dimension (A is N x N, each column of B has length N).
     * @tparam NRHS  Number of right-hand sides (columns of B).
     * @tparam REGULARIZE  If true, shift A before factoring (default false, compiles out).
     * @tparam CHECK  If true, report a non-PD pivot via `s_fail` (default false, compiles out).
     * @tparam REG_DIAG    With REGULARIZE: shift by `rho·diag(A)` instead of `rho·I` (default false).
     * @param A  In/out SPD matrix (column-major); on return its lower triangle holds L.
     * @param B  In/out right-hand sides (N x NRHS, column-major); on return holds X.
     * @param rho    Diagonal shift applied when REGULARIZE (ignored otherwise).
     * @param s_fail Optional non-PD flag when CHECK (set to 1 on a non-PD pivot, else 0).
     */
    template <typename T, uint32_t N, uint32_t NRHS,
              bool REGULARIZE = false, bool CHECK = false, bool REG_DIAG = false>
    __device__ void posv(T *A, T *B, T rho = T(0), int *s_fail = nullptr)
    {
        // Qualified: ct_size's home namespace is glass, so an unqualified call
        // would ADL-pull the block-scoped glass::_posv_regularize into the
        // overload set and be ambiguous with this warp-scoped one.
        if constexpr (REGULARIZE) warp::_posv_regularize<T, REG_DIAG>(ct_size<N>{}, A, rho);  // rho*I or rho*diag(A)
        potrf<T, N, CHECK>(A, s_fail);
        trsm<T, N, NRHS>(A, B);                                                        // forward: L Y = B
        trsm<T, N, NRHS, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(A, B);    // back:   Lᵀ X = Y
    }
}
