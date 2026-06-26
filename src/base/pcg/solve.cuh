#pragma once
#include <cstdint>

/**
 * @file solve.cuh
 * @brief Block-wide preconditioned conjugate gradient (`glass::pcg`).
 *
 * Solves a single block-tridiagonal SPD system `S x = b` inside ONE CUDA block,
 * with a block-tridiagonal preconditioner `Pinv` applied as `z = Pinv r`. This
 * is the single-block (one-block-per-problem) PCG — launch one block per
 * independent solve. It uses only `__syncthreads()` (no cooperative groups);
 * for the cooperative grid-wide variant see the backlog `glass::cgrps::grid`.
 *
 * Layouts (see `glass::bdmv`): `S` and `Pinv` are block-tridiagonal
 * `[L|D|R]` row-major strips; all vectors use the padded layout
 * `(knot_points + 2) * state_size` (one `state_size` pad block on each end).
 * The caller seeds `x` with an initial guess (zeros are fine).
 *
 * Convergence is tested on the preconditioned residual `rho = rᵀ z`:
 * `|rho| < abs_tol + rel_tol * |rho_init|`.
 *
 * Shared scratch: pass `s_mem` of `pcg_scratch_bytes<T,state_size,knot_points>(threads)`
 * elements (5 padded vectors + the warp-dot scratch). Five scalars live in
 * static `__shared__`. Requires `blockDim.x` be a multiple of 32 (the warp dot).
 */

/**
 * @brief Shared-memory element count needed by `glass::pcg`.
 *
 * Host- or device-callable. Multiply by `sizeof(T)` for the dynamic-shared-mem
 * bytes to pass at launch. Covers the 5 padded work vectors plus the warp-dot
 * scratch (`ceil(threads/32)`); the 5 scalars are static `__shared__`.
 *
 * @tparam T            Scalar type.
 * @tparam state_size   Block dimension.
 * @tparam knot_points  Number of block-rows.
 * @param threads  Launch thread count (`blockDim.x`).
 * @return Bytes of dynamic shared memory required.
 */
template <typename T, uint32_t state_size, uint32_t knot_points>
__host__ __device__ inline constexpr std::size_t pcg_scratch_bytes(uint32_t threads)
{
    return (static_cast<std::size_t>(5u * ((knot_points + 2u) * state_size)) + ((threads + 31u) / 32u)) * sizeof(T);
}

/**
 * @brief Solve `S x = b` by preconditioned conjugate gradient, one block (`glass::pcg`).
 *
 * @tparam T            Scalar type (e.g. `float`, `double`).
 * @tparam state_size   Block dimension (= `BlockSize` of the banded layout).
 * @tparam knot_points  Number of block-rows.
 * @param x          In/out padded solution, length `(knot_points+2)*state_size`
 *                   (seed with an initial guess; result written back here).
 * @param S          Block-tridiagonal SPD system, `[L|D|R]` row-major strips.
 * @param Pinv       Block-tridiagonal preconditioner, `[L|D|R]` row-major strips.
 * @param b          Padded right-hand side.
 * @param s_mem      Shared scratch of `pcg_scratch_bytes<T,...>(blockDim.x)` elements.
 * @param max_iters  Maximum CG iterations.
 * @param rel_tol    Relative tolerance on the preconditioned residual.
 * @param abs_tol    Absolute tolerance on the preconditioned residual.
 * @param iters      Output: iteration count written by thread 0 (may be null-safe
 *                   only if the caller guarantees a valid pointer; pass a valid
 *                   device address).
 */
template <typename T, uint32_t state_size, uint32_t knot_points>
__device__ void pcg(T *x, T *S, T *Pinv, T *b, T *s_mem,
                    uint32_t max_iters, T rel_tol, T abs_tol, uint32_t *iters)
{
    constexpr uint32_t VEC = (knot_points + 2) * state_size;

    T *s_Ap  = s_mem;
    T *s_x   = s_Ap + VEC;
    T *s_r   = s_x  + VEC;
    T *s_z   = s_r  + VEC;
    T *s_p   = s_z  + VEC;
    T *s_scr = s_p  + VEC;            // warp-dot scratch: ceil(blockDim/32) elems
    __shared__ T s_rho, s_rho_new, s_alpha, s_beta, s_rho_init;

    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

    set_const<T, 5 * VEC>(static_cast<T>(0), s_mem);   // zero the 5 vectors incl. pads
    __syncthreads();

    copy<T, VEC>(x, s_x);                              // s_x = x (initial guess)
    __syncthreads();

    bdmv<T, knot_points, state_size>(s_r, S, s_x);             // r = S x
    __syncthreads();
    axpby<T, VEC>(static_cast<T>(1), b, static_cast<T>(-1), s_r, s_r); // r = b - S x
    __syncthreads();

    bdmv<T, knot_points, state_size>(s_z, s_p, Pinv, s_r);     // z = p = Pinv r
    __syncthreads();

    dot_fast<T, VEC>(s_r, s_z, &s_rho, s_scr);                  // rho = rᵀ z
    __syncthreads();

    T arho = (s_rho < static_cast<T>(0)) ? -s_rho : s_rho;
    if (arho < abs_tol) {
        if (rank == 0) *iters = 0;
        copy<T, VEC>(s_x, x);
        __syncthreads();
        return;
    }
    if (rank == 0) s_rho_init = arho;
    __syncthreads();

    uint32_t it = 0;
    for (uint32_t i = 0; i < max_iters; i++) {
        it = i + 1;

        bdmv<T, knot_points, state_size>(s_Ap, S, s_p);        // Ap = S p
        __syncthreads();

        dot_fast<T, VEC>(s_p, s_Ap, &s_alpha, s_scr);          // pᵀ Ap
        __syncthreads();
        if (rank == 0) s_alpha = s_rho / s_alpha;                      // alpha
        __syncthreads();

        axpy<T, VEC>(s_alpha, s_p, s_x);                              // x += alpha p
        axpy<T, VEC>(-s_alpha, s_Ap, s_r);                           // r -= alpha Ap
        __syncthreads();

        bdmv<T, knot_points, state_size>(s_z, Pinv, s_r);     // z = Pinv r
        __syncthreads();

        dot_fast<T, VEC>(s_r, s_z, &s_rho_new, s_scr);        // rho_new = rᵀ z
        __syncthreads();

        T arho_new = (s_rho_new < static_cast<T>(0)) ? -s_rho_new : s_rho_new;
        if (arho_new < abs_tol + rel_tol * s_rho_init) break;

        if (rank == 0) { s_beta = s_rho_new / s_rho; s_rho = s_rho_new; }
        __syncthreads();

        axpby<T, VEC>(static_cast<T>(1), s_z, s_beta, s_p, s_p);      // p = z + beta p
        __syncthreads();
    }

    if (rank == 0) *iters = it;
    copy<T, VEC>(s_x, x);                                             // write solution back
    __syncthreads();
}

