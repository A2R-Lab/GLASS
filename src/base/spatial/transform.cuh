#pragma once
#include <cstdint>

// ─── Featherstone spatial (6-D) coordinate transforms (robotics ops) ─────────
//
// The frame-change companion of `spatial/cross.cuh`: the spatial motion
// transform `X`, the spatial force transform `X* = X⁻ᵀ`, and their fused
// applies in both directions. Together with the cross products and the
// spatial inertia (`spatial/inertia.cuh`) these are the complete inner-loop
// op set of RNEA/ABA/CRBA sweeps (cuRobo's `spatial_Xv`/`spatial_XTf`
// independently reinvented exactly this pair, which is the evidence this is
// the field's shared primitive layer; GRiD keeps its Xmats baked in codegen
// and can compose with these at will).
//
// CONVENTION — a transform is carried as the PAIR `(E, r)`, never a packed
// 6x6 (the fused applies exist so the 36-entry matrix never materializes):
//   * `E` is the 3x3 rotation FROM frame A TO frame B (column-major,
//     `x_B = E·x_A` for ordinary 3-vectors).
//   * `r` is the position of frame B's origin expressed IN FRAME A.
//   * The motion transform is Featherstone's `ᴮX_A = rot(E)·xlt(r)
//     = [[E, 0], [−E·[r]ₓ, E]]`; for `v = [ω; v_lin]` in A coordinates,
//     `X·v` gives the same motion in B coordinates.
//   * The force transform is `X* = X⁻ᵀ = [[E, −E·[r]ₓ], [0, E]]`.
//   * `INVERSE = true` applies the INVERSE map (B → A coordinates) directly
//     from the same `(E, r)` — nothing is materialized or factored:
//     `X⁻¹ = [[Eᵀ, 0], [[r]ₓ·Eᵀ, Eᵀ]]`, `X*⁻¹ = Xᵀ`. In Featherstone
//     pseudo-code terms: `motion_transform_mul` is `X·v`,
//     `force_transform_mul<INVERSE=true>` is the RNEA back-pass `Xᵀ·f`.
//   * Spatial vectors are ANGULAR-FIRST `[ω; v_lin]` / `[n; f_lin]` and 6x6
//     matrices column-major, as everywhere in this family (cross.cuh note).
//
// The fused `*_mul` applies carry BLAS-style `(alpha, beta)` with
// `beta_blend` beta==0 semantics (`HAS_BETA` gates the read of `y`), matching
// `motion_cross_mul`. Tiers: the fused applies use the redundant-core pattern
// (each active thread computes the 6-vector in registers, the tier strides
// the blended copy-out); the matrix materializers stride entries directly.
// Outputs must not alias inputs at block/warp scope.

namespace spatial_detail {
    // y = E·x and y = Eᵀ·x, 3x3 column-major (y must not alias x).
    template <typename T>
    __device__ __forceinline__ void rot_apply(const T *E, const T *x, T *y) {
        y[0] = E[0]*x[0] + E[3]*x[1] + E[6]*x[2];
        y[1] = E[1]*x[0] + E[4]*x[1] + E[7]*x[2];
        y[2] = E[2]*x[0] + E[5]*x[1] + E[8]*x[2];
    }
    template <typename T>
    __device__ __forceinline__ void rot_apply_t(const T *E, const T *x, T *y) {
        y[0] = E[0]*x[0] + E[1]*x[1] + E[2]*x[2];
        y[1] = E[3]*x[0] + E[4]*x[1] + E[5]*x[2];
        y[2] = E[6]*x[0] + E[7]*x[1] + E[8]*x[2];
    }
    // c = a × b (c must not alias a/b).
    template <typename T>
    __device__ __forceinline__ void cross3(const T *a, const T *b, T *c) {
        c[0] = a[1]*b[2] - a[2]*b[1];
        c[1] = a[2]*b[0] - a[0]*b[2];
        c[2] = a[0]*b[1] - a[1]*b[0];
    }

    // Entry (i, j) of the block `−E·[r]ₓ` (the motion transform's bottom-left
    // and the force transform's top-right): `Σ_k E(i,k)·(e_j × r)_k`.
    template <typename T>
    __device__ __forceinline__ T xform_skew_entry(uint32_t i, uint32_t j,
                                                  const T *E, const T *r) {
        switch (j) {
            case 0:  return -E[3 + i]*r[2] + E[6 + i]*r[1];
            case 1:  return  E[i]*r[2]     - E[6 + i]*r[0];
            default: return -E[i]*r[1]     + E[3 + i]*r[0];
        }
    }

    // Entry (i, j) of X (FORCE = false) or X* (FORCE = true), column-major.
    template <typename T, bool FORCE>
    __device__ __forceinline__ T xform_entry(uint32_t i, uint32_t j,
                                             const T *E, const T *r) {
        const uint32_t bi = i / 3, bj = j / 3;
        if (bi == bj) return E[(j % 3)*3 + (i % 3)];          // diagonal blocks: E
        if (FORCE ? (bi == 0) : (bi == 1))                    // the −E·[r]ₓ block
            return xform_skew_entry(i % 3, j % 3, E, r);
        return static_cast<T>(0);                             // the zero block
    }

    // serial cores: the fused applies, into a register 6-vector.
    template <typename T, bool INVERSE>
    __device__ __forceinline__ void motion_transform_mul_core(const T *E, const T *r,
                                                              const T *v, T *y) {
        T t[3], w[3];
        if constexpr (!INVERSE) {          // ω' = E·ω ; v' = E·(v_lin − r×ω)
            cross3(r, v, t);
            t[0] = v[3] - t[0]; t[1] = v[4] - t[1]; t[2] = v[5] - t[2];
            rot_apply(E, v, y);
            rot_apply(E, t, y + 3);
        } else {                           // ω = Eᵀ·ω' ; v_lin = Eᵀ·v' + r×ω
            rot_apply_t(E, v, y);
            rot_apply_t(E, v + 3, t);
            cross3(r, y, w);
            y[3] = t[0] + w[0]; y[4] = t[1] + w[1]; y[5] = t[2] + w[2];
        }
    }

    template <typename T, bool INVERSE>
    __device__ __forceinline__ void force_transform_mul_core(const T *E, const T *r,
                                                             const T *f, T *y) {
        T t[3], w[3];
        if constexpr (!INVERSE) {          // n' = E·(n − r×f_lin) ; f' = E·f_lin
            cross3(r, f + 3, t);
            t[0] = f[0] - t[0]; t[1] = f[1] - t[1]; t[2] = f[2] - t[2];
            rot_apply(E, t, y);
            rot_apply(E, f + 3, y + 3);
        } else {                           // f_lin = Eᵀ·f' ; n = Eᵀ·n' + r×f_lin
            rot_apply_t(E, f + 3, y + 3);
            rot_apply_t(E, f, t);
            cross3(r, y + 3, w);
            y[0] = t[0] + w[0]; y[1] = t[1] + w[1]; y[2] = t[2] + w[2];
        }
    }

    // tier glue: strided blended copy-out of a register tmp (the redundant-core
    // pattern with `(alpha, beta)` scaling). unroll 1: differing FMA contraction
    // between unroll copies would break bit-identity across thread counts.
    template <typename T, uint32_t N, bool HAS_BETA>
    __device__ __forceinline__ void blend_out(uint32_t rank, uint32_t size, T alpha,
                                              const T *tmp, T beta, T *y) {
        #pragma unroll 1
        for (uint32_t i = rank; i < N; i += size)
            y[i] = HAS_BETA ? beta_blend(alpha*tmp[i], beta, y[i]) : (alpha*tmp[i]);
    }

    template <typename T, bool FORCE>
    __device__ __forceinline__ void xform_mat_impl(uint32_t rank, uint32_t size,
                                                   const T *E, const T *r, T *X) {
        for (uint32_t i = rank; i < 36; i += size)
            X[i] = xform_entry<T, FORCE>(i % 6, i / 6, E, r);
    }
} // namespace spatial_detail

/**
 * @brief Spatial motion transform matrix: `X = [[E, 0], [−E·[r]ₓ, E]]` (6x6).
 *
 * Materializes Featherstone's `ᴮX_A` from the `(E, r)` pair (see the file
 * header for the frame convention). Column-major. Prefer the fused
 * `motion_transform_mul` when only a product is needed.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param E  3x3 rotation A→B (9 elements, column-major).
 * @param r  Origin of B expressed in A (3 elements).
 * @param X  Output 6x6 transform (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void motion_transform(const T *E, const T *r, T *X)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    spatial_detail::xform_mat_impl<T, false>(rank, size, E, r, X);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Spatial force transform matrix: `X* = X⁻ᵀ = [[E, −E·[r]ₓ], [0, E]]` (6x6).
 *
 * The dual transform for spatial FORCE vectors. Column-major. Prefer the
 * fused `force_transform_mul`.
 *
 * @tparam T,TRAILING_SYNC  See `motion_transform`.
 * @param E,r  The transform pair (see `motion_transform`).
 * @param X    Output 6x6 transform (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void force_transform(const T *E, const T *r, T *X)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    spatial_detail::xform_mat_impl<T, true>(rank, size, E, r, X);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Fused spatial motion transform apply: `y = alpha·X·v + beta·y`
 *        (or `X⁻¹·v` when `INVERSE`).
 *
 * Never materializes the 6x6: `X·v = [E·ω; E·(v_lin − r×ω)]`,
 * `X⁻¹·v = [Eᵀ·ω; Eᵀ·v_lin + r×(Eᵀ·ω)]`. The forward form is the RNEA/ABA
 * velocity down-sweep (`v_child = X·v_parent`); the inverse recovers
 * parent-frame coordinates. `beta` is only read when `HAS_BETA` (BLAS beta==0
 * semantics via `beta_blend`). Tested against the
 * `gemv(motion_transform(E, r), v)` composition.
 *
 * @tparam T  Scalar type.
 * @tparam INVERSE  Apply `X⁻¹` instead of `X` (default false).
 * @tparam HAS_BETA  Read/accumulate into `y` (default false = overwrite).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar on the product.
 * @param E,r    The transform pair (see `motion_transform`).
 * @param v      Spatial motion vector (6 elements, `[ω; v_lin]`).
 * @param beta   Scalar on the existing `y` (read only when `HAS_BETA`).
 * @param y      Output spatial motion vector (6 elements; no aliasing).
 */
template <typename T, bool INVERSE = false, bool HAS_BETA = false, bool TRAILING_SYNC = true>
__device__ void motion_transform_mul(T alpha, const T *E, const T *r,
                                     const T *v, T beta, T *y)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[6]; spatial_detail::motion_transform_mul_core<T, INVERSE>(E, r, v, tmp);
    spatial_detail::blend_out<T, 6, HAS_BETA>(rank, size, alpha, tmp, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Fused spatial force transform apply: `y = alpha·X*·f + beta·y`
 *        (or `X*⁻¹·f = Xᵀ·f` when `INVERSE`).
 *
 * `X*·f = [E·(n − r×f_lin); E·f_lin]`,
 * `Xᵀ·f = [Eᵀ·n + r×(Eᵀ·f_lin); Eᵀ·f_lin]`. The INVERSE form is THE RNEA
 * back-pass (`f_parent += Xᵀ·f_child` — use `HAS_BETA = true, beta = 1`).
 * Same `(alpha, beta)` semantics as `motion_transform_mul`. Tested against
 * the `gemv(force_transform(E, r), f)` composition.
 *
 * @tparam T,INVERSE,HAS_BETA,TRAILING_SYNC  See `motion_transform_mul`.
 * @param alpha  Scalar on the product.
 * @param E,r    The transform pair (see `motion_transform`).
 * @param f      Spatial force vector (6 elements, `[n; f_lin]`).
 * @param beta   Scalar on the existing `y` (read only when `HAS_BETA`).
 * @param y      Output spatial force vector (6 elements; no aliasing).
 */
template <typename T, bool INVERSE = false, bool HAS_BETA = false, bool TRAILING_SYNC = true>
__device__ void force_transform_mul(T alpha, const T *E, const T *r,
                                    const T *f, T beta, T *y)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[6]; spatial_detail::force_transform_mul_core<T, INVERSE>(E, r, f, tmp);
    spatial_detail::blend_out<T, 6, HAS_BETA>(rank, size, alpha, tmp, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /** @brief Single-warp motion transform matrix. See `glass::motion_transform`. */
    template <typename T>
    __device__ void motion_transform(const T *E, const T *r, T *X)
    {
        uint32_t lane = (flat_rank()) & 31;
        spatial_detail::xform_mat_impl<T, false>(lane, 32u, E, r, X);
        __syncwarp();
    }

    /** @brief Single-warp force transform matrix. See `glass::force_transform`. */
    template <typename T>
    __device__ void force_transform(const T *E, const T *r, T *X)
    {
        uint32_t lane = (flat_rank()) & 31;
        spatial_detail::xform_mat_impl<T, true>(lane, 32u, E, r, X);
        __syncwarp();
    }

    /** @brief Single-warp fused motion transform apply. See `glass::motion_transform_mul`. */
    template <typename T, bool INVERSE = false, bool HAS_BETA = false>
    __device__ void motion_transform_mul(T alpha, const T *E, const T *r,
                                         const T *v, T beta, T *y)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[6]; spatial_detail::motion_transform_mul_core<T, INVERSE>(E, r, v, tmp);
        spatial_detail::blend_out<T, 6, HAS_BETA>(lane, 32u, alpha, tmp, beta, y);
        __syncwarp();
    }

    /** @brief Single-warp fused force transform apply. See `glass::force_transform_mul`. */
    template <typename T, bool INVERSE = false, bool HAS_BETA = false>
    __device__ void force_transform_mul(T alpha, const T *E, const T *r,
                                        const T *f, T beta, T *y)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[6]; spatial_detail::force_transform_mul_core<T, INVERSE>(E, r, f, tmp);
        spatial_detail::blend_out<T, 6, HAS_BETA>(lane, 32u, alpha, tmp, beta, y);
        __syncwarp();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    /** @brief Single-thread motion transform matrix. See `glass::motion_transform`. */
    template <typename T>
    __device__ void motion_transform(const T *E, const T *r, T *X)
    { spatial_detail::xform_mat_impl<T, false>(0u, 1u, E, r, X); }

    /** @brief Single-thread force transform matrix. See `glass::force_transform`. */
    template <typename T>
    __device__ void force_transform(const T *E, const T *r, T *X)
    { spatial_detail::xform_mat_impl<T, true>(0u, 1u, E, r, X); }

    /** @brief Single-thread fused motion transform apply. See `glass::motion_transform_mul`. */
    template <typename T, bool INVERSE = false, bool HAS_BETA = false>
    __device__ void motion_transform_mul(T alpha, const T *E, const T *r,
                                         const T *v, T beta, T *y)
    {
        T tmp[6]; spatial_detail::motion_transform_mul_core<T, INVERSE>(E, r, v, tmp);
        spatial_detail::blend_out<T, 6, HAS_BETA>(0u, 1u, alpha, tmp, beta, y);
    }

    /** @brief Single-thread fused force transform apply. See `glass::force_transform_mul`. */
    template <typename T, bool INVERSE = false, bool HAS_BETA = false>
    __device__ void force_transform_mul(T alpha, const T *E, const T *r,
                                        const T *f, T beta, T *y)
    {
        T tmp[6]; spatial_detail::force_transform_mul_core<T, INVERSE>(E, r, f, tmp);
        spatial_detail::blend_out<T, 6, HAS_BETA>(0u, 1u, alpha, tmp, beta, y);
    }
}
