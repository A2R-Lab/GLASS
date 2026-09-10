#pragma once
#include <cstdint>
#include <cstddef>

/**
 * @file bdsv.cuh
 * @brief Direct block-tridiagonal SPD factor + solve (`glass::bdsv`).
 *
 * Block-Cholesky (block-Thomas) sweep over a block-tridiagonal SPD matrix in
 * the same `[L|D|R]` strip layout as `glass::bdmv` / `glass::pcg`: block-row
 * `br` is a `BlockSize × (3*BlockSize)` row-major tile starting at
 * `s_matrix + br * 3*BlockSize*BlockSize`, and vectors use the padded
 * `(NumBlockRows+2)*BlockSize` layout (one zero pad block each end).
 *
 * This is the DIRECT alternative to the iterative `glass::pcg` for the
 * Riccati/KKT/Schur systems consumers assemble in this layout — exact in one
 * sweep, no convergence tuning; serial over the `NumBlockRows` knots (the
 * dependency chain is inherent) with full in-block parallelism inside each
 * knot's potrf/trsm/syrk/gemv/trsv.
 *
 * Factorization (`bdsv_factor`, in place on the strips):
 *   F_0 = chol(D_0);   for k>0:  E_k = L_k F_{k-1}⁻ᵀ,
 *                                F_k = chol(D_k − E_k E_kᵀ)
 * After it returns, block-row k's `MAIN` slot holds the Cholesky factor `F_k`
 * (lower triangle; its upper triangle keeps stale `D_k` entries) and its
 * `LEFT` slot holds `E_k`. `RIGHT` slots are never read or written (for a
 * symmetric system they mirror `LEFT` of the next row). The strips no longer
 * hold `A` — keep a copy if you still need `bdmv` products with `A`.
 *
 * Solve (`bdsv_solve`, in place on the padded vector): forward sweep
 * `y_k = F_k⁻¹ (b_k − E_k y_{k-1})`, then backward sweep
 * `x_k = F_k⁻ᵀ (y_k − E_{k+1}ᵀ x_{k+1})`.
 *
 * Scratch: `bdsv_scratch_bytes<T, BlockSize>()` bytes (two dense
 * `BlockSize×BlockSize` staging buffers — strip blocks are strided inside the
 * row-major tile, so each knot's blocks are staged contiguous via
 * `load_block`/`store_block`).
 *
 * Thread-count invariant; composes entirely from existing primitives, so it
 * ends on their trailing barriers.
 */

/**
 * @brief Scratch size in bytes for `bdsv` / `bdsv_factor` / `bdsv_solve`.
 *
 * @tparam T          Scalar type.
 * @tparam BlockSize  Block dimension.
 * @return Bytes to allocate for `s_scratch` (two dense BlockSize² blocks).
 */
template <typename T, uint32_t BlockSize>
__host__ __device__ constexpr std::size_t bdsv_scratch_bytes()
{
    return static_cast<std::size_t>(2u * BlockSize * BlockSize) * sizeof(T);
}

/**
 * @brief In-place block-Cholesky factorization of a block-tridiagonal SPD matrix.
 *
 * See the file docs for the factor layout left in the strips. With `CHECK`,
 * a non-positive-definite pivot in any knot's Cholesky sets `*s_fail = 1`
 * (the sweep still runs to completion; treat the factor as invalid).
 *
 * @tparam T             Scalar type (prefer `double` for long/ill-conditioned chains).
 * @tparam NumBlockRows  Number of block-rows (knots).
 * @tparam BlockSize     Block dimension.
 * @tparam CHECK         Report a non-PD pivot via `s_fail` (default false, compiles out).
 * @param s_matrix   In/out `[L|D|R]` strips; on return `MAIN`←`F_k` (lower), `LEFT`←`E_k`.
 * @param s_scratch  Shared scratch of `bdsv_scratch_bytes<T, BlockSize>()` bytes.
 * @param s_fail     Optional non-PD flag when CHECK (set to 1 on failure, else untouched).
 */
template <typename T, uint32_t NumBlockRows, uint32_t BlockSize, bool CHECK = false>
__device__ void bdsv_factor(T *s_matrix, T *s_scratch, int *s_fail = nullptr)
{
    constexpr uint32_t BRL = 3 * BlockSize;
    T *s1 = s_scratch;                            // F_{k-1}, then D̃_k
    T *s2 = s_scratch + BlockSize * BlockSize;    // L_kᵀ → E_kᵀ
    for (uint32_t k = 0; k < NumBlockRows; k++) {
        T *strip = s_matrix + k * BRL * BlockSize;
        if (k > 0) {
            const T *prev = s_matrix + (k - 1) * BRL * BlockSize;
            // s1 ← F_{k-1} (as-is → col-major), s2 ← L_kᵀ (transposed load)
            load_block<T, BlockSize, BRL, /*TRANSPOSE=*/true>(s1, prev, BandSlot::MAIN);
            load_block<T, BlockSize, BRL, /*TRANSPOSE=*/false>(s2, strip, BandSlot::LEFT);
            // E_kᵀ = F_{k-1}⁻¹ L_kᵀ  (forward triangular solve, BlockSize RHS)
            trsm<T, BlockSize, BlockSize>(s1, s2);
            // write E_k back into the LEFT slot (store the transpose of s2)
            store_block<T, BlockSize, BRL, /*TRANSPOSE=*/false>(strip, BandSlot::LEFT, s2);
            // s1 ← D_k, then D̃_k = D_k − E_k E_kᵀ = D_k − (E_kᵀ)ᵀ(E_kᵀ)
            load_block<T, BlockSize, BRL, /*TRANSPOSE=*/true>(s1, strip, BandSlot::MAIN);
            syrk<T, FillMode::Lower, /*TRANSPOSE=*/true>(BlockSize, BlockSize,
                static_cast<T>(-1), s2, static_cast<T>(1), s1);
        } else {
            load_block<T, BlockSize, BRL, /*TRANSPOSE=*/true>(s1, strip, BandSlot::MAIN);
        }
        potrf<T, CHECK>(BlockSize, s1, s_fail);   // s1 ← F_k (lower)
        store_block<T, BlockSize, BRL, /*TRANSPOSE=*/true>(strip, BandSlot::MAIN, s1);
    }
}

/**
 * @brief Solve `A x = b` from a `bdsv_factor`ed block-tridiagonal system.
 *
 * Forward then backward block substitution. `s_vector` uses the padded
 * `(NumBlockRows+2)*BlockSize` layout with `b` in the interior blocks
 * (block-row `br` at offset `(br+1)*BlockSize`); on return the interior holds
 * `x`. Reusable: factor once, call this per right-hand side.
 *
 * @tparam T             Scalar type.
 * @tparam NumBlockRows  Number of block-rows (knots).
 * @tparam BlockSize     Block dimension.
 * @param s_matrix   Factored strips from `bdsv_factor` (read-only).
 * @param s_vector   In/out padded right-hand side; interior overwritten with `x`.
 * @param s_scratch  Shared scratch of `bdsv_scratch_bytes<T, BlockSize>()` bytes.
 */
template <typename T, uint32_t NumBlockRows, uint32_t BlockSize>
__device__ void bdsv_solve(const T *s_matrix, T *s_vector, T *s_scratch)
{
    constexpr uint32_t BRL = 3 * BlockSize;
    T *s1 = s_scratch;
    // forward: y_k = F_k⁻¹ (b_k − E_k y_{k-1})
    for (uint32_t k = 0; k < NumBlockRows; k++) {
        const T *strip = s_matrix + k * BRL * BlockSize;
        T *xk = s_vector + (k + 1) * BlockSize;
        if (k > 0) {
            load_block<T, BlockSize, BRL, /*TRANSPOSE=*/true>(s1, strip, BandSlot::LEFT); // E_k
            gemv<T>(BlockSize, BlockSize, static_cast<T>(-1), s1, xk - BlockSize,
                    static_cast<T>(1), xk);
        }
        load_block<T, BlockSize, BRL, /*TRANSPOSE=*/true>(s1, strip, BandSlot::MAIN);     // F_k
        trsv<T>(BlockSize, s1, xk);
    }
    // backward: x_k = F_k⁻ᵀ (y_k − E_{k+1}ᵀ x_{k+1})
    for (uint32_t step = 0; step < NumBlockRows; step++) {
        uint32_t k = NumBlockRows - 1 - step;
        const T *strip = s_matrix + k * BRL * BlockSize;
        T *xk = s_vector + (k + 1) * BlockSize;
        if (k + 1 < NumBlockRows) {
            const T *next = s_matrix + (k + 1) * BRL * BlockSize;
            load_block<T, BlockSize, BRL, /*TRANSPOSE=*/false>(s1, next, BandSlot::LEFT); // E_{k+1}ᵀ
            gemv<T>(BlockSize, BlockSize, static_cast<T>(-1), s1, xk + BlockSize,
                    static_cast<T>(1), xk);
        }
        load_block<T, BlockSize, BRL, /*TRANSPOSE=*/true>(s1, strip, BandSlot::MAIN);     // F_k
        trsv<T, FillMode::Lower, Diag::NonUnit, /*TRANSPOSE=*/true>(BlockSize, s1, xk);
    }
}

/**
 * @brief Direct block-tridiagonal SPD solve: factor + solve in one call (in place).
 *
 * `bdsv_factor` then `bdsv_solve`. On return the strips hold the factor
 * (`MAIN`←`F_k` lower, `LEFT`←`E_k`) and the padded vector's interior holds
 * `x`. The exact one-sweep alternative to `glass::pcg` on the same layout.
 * NumPy equivalent: `x = np.linalg.solve(A_dense, b)` (A SPD block-tridiag).
 *
 * @tparam T             Scalar type.
 * @tparam NumBlockRows  Number of block-rows (knots).
 * @tparam BlockSize     Block dimension.
 * @tparam CHECK         Report a non-PD pivot via `s_fail` (default false, compiles out).
 * @param s_matrix   In/out `[L|D|R]` strips (see `bdsv_factor`).
 * @param s_vector   In/out padded right-hand side; interior overwritten with `x`.
 * @param s_scratch  Shared scratch of `bdsv_scratch_bytes<T, BlockSize>()` bytes.
 * @param s_fail     Optional non-PD flag when CHECK.
 */
template <typename T, uint32_t NumBlockRows, uint32_t BlockSize, bool CHECK = false>
__device__ void bdsv(T *s_matrix, T *s_vector, T *s_scratch, int *s_fail = nullptr)
{
    bdsv_factor<T, NumBlockRows, BlockSize, CHECK>(s_matrix, s_scratch, s_fail);
    bdsv_solve<T, NumBlockRows, BlockSize>(s_matrix, s_vector, s_scratch);
}
