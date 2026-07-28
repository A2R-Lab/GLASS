#pragma once
#include <cstdint>

// ─── Featherstone spatial (6-D) cross products (robotics ops) ────────────────
//
// The rigid-body-dynamics core: motion and force cross-product matrices and
// their fused applies, plus the operand-swapped dual. Every RNEA/ABA/CRBA
// kernel and every analytic dynamics gradient is built from these; the
// formulas are promoted verbatim from GRiD's Pinocchio-validated codegen
// emitters (and the same op set was independently reinvented by cuRobo's RNEA
// kernels and ships in MuJoCo/mujoco-warp/Warp — this is the field's shared
// primitive layer).
//
// CONVENTION — Featherstone spatial vectors, ANGULAR-FIRST:
//   * A spatial MOTION vector is `v = [ω(3); v_lin(3)]`, a spatial FORCE
//     vector is `f = [n(3); f_lin(3)]` (moment first). This is the
//     Featherstone/GRiD/MuJoCo ordering — note Pinocchio's USER-facing order
//     is linear-first, and the SE(3)-tangent blocks in `lie/se3.cuh` are
//     linear-first: the two families keep their native literature
//     conventions; permute explicitly when crossing between them.
//   * 6x6 matrices are column-major (`M[c*6 + r]`), matching the rest of GLASS.
//   * `motion_cross(v)` (a.k.a. `v ×`, "crm") = `[[ωₓ, 0], [v_linₓ, ωₓ]]`.
//   * `force_cross(v)` (a.k.a. `v ×*`, "crf"/"fx") = `−motion_cross(v)ᵀ`
//     = `[[ωₓ, v_linₓ], [0, ωₓ]]`.
//   * `force_cross_dual(f)` (a.k.a. "icrf") is DEFINED by the identity
//     `force_cross(v)·f == force_cross_dual(f)·v` — it swaps which operand
//     becomes the matrix (the object inertia-gradient kernels need).
//
// The fused `*_mul` applies never materialize the 6x6 (each output row is a
// 2-4 term formula) and carry BLAS-style `(alpha, beta)` scaling with
// `beta_blend` beta==0 semantics — one signature replaces the historical
// `_peq`/`_scaled`/`_peq_scaled` twin explosion. The compile-time `AXIS`
// parameter on `motion_cross_mul` specializes the multiply to a cardinal
// basis column (`x = e_AXIS`), the motion-subspace fast path of joint sweeps.
//
// Tiers: block/warp/thread share the row/entry formulas (strided over the 6 or
// 36 outputs). Outputs must not alias inputs at block/warp scope.

namespace spatial_detail {
    // Entry (r, c) of motion_cross(v), column-major reading order. The table is
    // the GRiD `crm` emitter's (already column-major) index table, verbatim.
    template <typename T>
    __device__ __forceinline__ T motion_cross_entry(uint32_t r, uint32_t c, const T *v) {
        const T zero = static_cast<T>(0);
        switch (c*6 + r) {
            case  1: return  v[2];  case  2: return -v[1];
            case  4: return  v[5];  case  5: return -v[4];
            case  6: return -v[2];  case  8: return  v[0];
            case  9: return -v[5];  case 11: return  v[3];
            case 12: return  v[1];  case 13: return -v[0];
            case 15: return  v[4];  case 16: return -v[3];
            case 22: return  v[2];  case 23: return -v[1];
            case 27: return -v[2];  case 29: return  v[0];
            case 33: return  v[1];  case 34: return -v[0];
            default: return zero;
        }
    }

    // Row r of motion_cross(v)·x (the fused apply; GRiD `crm_mul` rows, verbatim).
    template <typename T>
    __device__ __forceinline__ T motion_cross_mul_row(uint32_t r, const T *v, const T *x) {
        switch (r) {
            case 0: return -v[2]*x[1] + v[1]*x[2];
            case 1: return  v[2]*x[0] - v[0]*x[2];
            case 2: return -v[1]*x[0] + v[0]*x[1];
            case 3: return -v[5]*x[1] + v[4]*x[2] - v[2]*x[4] + v[1]*x[5];
            case 4: return  v[5]*x[0] - v[3]*x[2] + v[2]*x[3] - v[0]*x[5];
            default: return -v[4]*x[0] + v[3]*x[1] - v[1]*x[3] + v[0]*x[4];
        }
    }

    // Entry (r, c) of force_cross(v) = −motion_cross(v)ᵀ (GRiD `fx` table, verbatim).
    template <typename T>
    __device__ __forceinline__ T force_cross_entry(uint32_t r, uint32_t c, const T *v) {
        return -motion_cross_entry(c, r, v);
    }

    // Row r of force_cross(v)·f (GRiD `fx_times_v` rows, verbatim).
    template <typename T>
    __device__ __forceinline__ T force_cross_mul_row(uint32_t r, const T *v, const T *f) {
        switch (r) {
            case 0: return -v[2]*f[1] + v[1]*f[2] - v[5]*f[4] + v[4]*f[5];
            case 1: return  v[2]*f[0] - v[0]*f[2] + v[5]*f[3] - v[3]*f[5];
            case 2: return -v[1]*f[0] + v[0]*f[1] - v[4]*f[3] + v[3]*f[4];
            case 3: return -v[2]*f[4] + v[1]*f[5];
            case 4: return  v[2]*f[3] - v[0]*f[5];
            default: return -v[1]*f[3] + v[0]*f[4];
        }
    }

    // Entry (r, c) of force_cross_dual(f) (GRiD `icrf` table incl. its global
    // negation, verbatim; column-major).
    template <typename T>
    __device__ __forceinline__ T force_cross_dual_entry(uint32_t r, uint32_t c, const T *f) {
        const T zero = static_cast<T>(0);
        T result;
        switch (c*6 + r) {
            case  1: result =  f[2]; break;  case  2: result = -f[1]; break;
            case  4: result =  f[5]; break;  case  5: result = -f[4]; break;
            case  6: result = -f[2]; break;  case  8: result =  f[0]; break;
            case  9: result = -f[5]; break;  case 11: result =  f[3]; break;
            case 12: result =  f[1]; break;  case 13: result = -f[0]; break;
            case 15: result =  f[4]; break;  case 16: result = -f[3]; break;
            case 19: result =  f[5]; break;  case 20: result = -f[4]; break;
            case 24: result = -f[5]; break;  case 26: result =  f[3]; break;
            case 30: result =  f[4]; break;  case 31: result = -f[3]; break;
            default: result = zero; break;
        }
        return -result;
    }

    // tier-shared bodies (strided over outputs; rank/size from the tier glue).
    template <typename T>
    __device__ __forceinline__ void motion_cross_impl(uint32_t rank, uint32_t size,
                                                      const T *v, T *M) {
        for (uint32_t i = rank; i < 36; i += size)
            M[i] = motion_cross_entry<T>(i % 6, i / 6, v);
    }

    template <typename T, int AXIS, bool HAS_BETA>
    __device__ __forceinline__ void motion_cross_mul_impl(uint32_t rank, uint32_t size,
                                                          T alpha, const T *v, const T *x,
                                                          T beta, T *y) {
        static_assert(AXIS >= -1 && AXIS < 6, "AXIS must be -1 (dense x) or 0..5");
        // unroll 1: differing FMA contraction between unroll copies would break
        // bit-identity across thread counts (see se3_retract_hessian_impl).
        #pragma unroll 1
        for (uint32_t r = rank; r < 6; r += size) {
            T res;
            if constexpr (AXIS >= 0) res = motion_cross_entry<T>(r, (uint32_t)AXIS, v);
            else                     res = motion_cross_mul_row<T>(r, v, x);
            y[r] = HAS_BETA ? beta_blend(alpha*res, beta, y[r]) : (alpha*res);
        }
    }

    template <typename T>
    __device__ __forceinline__ void force_cross_impl(uint32_t rank, uint32_t size,
                                                     const T *v, T *M) {
        for (uint32_t i = rank; i < 36; i += size)
            M[i] = force_cross_entry<T>(i % 6, i / 6, v);
    }

    template <typename T, bool HAS_BETA>
    __device__ __forceinline__ void force_cross_mul_impl(uint32_t rank, uint32_t size,
                                                         T alpha, const T *v, const T *f,
                                                         T beta, T *y) {
        #pragma unroll 1
        for (uint32_t r = rank; r < 6; r += size) {
            const T res = force_cross_mul_row<T>(r, v, f);
            y[r] = HAS_BETA ? beta_blend(alpha*res, beta, y[r]) : (alpha*res);
        }
    }

    template <typename T>
    __device__ __forceinline__ void force_cross_dual_impl(uint32_t rank, uint32_t size,
                                                          const T *f, T *M) {
        for (uint32_t i = rank; i < 36; i += size)
            M[i] = force_cross_dual_entry<T>(i % 6, i / 6, f);
    }
} // namespace spatial_detail

/**
 * @brief Spatial motion cross-product matrix: `M = motion_cross(v)` (6x6, "crm").
 *
 * `M·x = v ×ₘ x` for spatial motion vectors, `M = [[ωₓ, 0],[v_linₓ, ωₓ]]`
 * with `v = [ω; v_lin]` (angular-first) and column-major storage. Prefer the
 * fused `motion_cross_mul` when only the product is needed.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param v  Spatial motion vector (6 elements, `[ω; v_lin]`).
 * @param M  Output 6x6 matrix (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void motion_cross(const T *v, T *M)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    spatial_detail::motion_cross_impl(rank, size, v, M);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Fused spatial motion cross apply: `y = alpha·(v ×ₘ x) + beta·y`.
 *
 * Never materializes the 6x6 — each output row is its 2-4 term formula (the
 * hot inner op of RNEA/ABA velocity sweeps; ~70 call sites in a typical
 * generated dynamics suite). With compile-time `AXIS` in 0..5 the multiply
 * specializes to the cardinal basis column `x = e_AXIS` (the revolute/
 * prismatic motion-subspace fast path; `x` is ignored and may be nullptr).
 * `beta` is only read when `HAS_BETA` (BLAS beta==0 semantics via
 * `beta_blend`); the `(alpha, beta)` pair replaces the `_peq`/`_scaled`
 * variant explosion. Equivalent to `gemv(motion_cross(v), x)` — tested
 * against exactly that composition.
 *
 * @tparam T  Scalar type.
 * @tparam AXIS  −1 for a dense `x` (default), or 0..5 for `x = e_AXIS`.
 * @tparam HAS_BETA  Read/accumulate into `y` (default false = overwrite).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar on the product.
 * @param v      Spatial motion vector (6 elements, `[ω; v_lin]`).
 * @param x      Right operand (6 elements; ignored when `AXIS >= 0`).
 * @param beta   Scalar on the existing `y` (read only when `HAS_BETA`).
 * @param y      Output spatial vector (6 elements; no aliasing with `v`/`x`).
 */
template <typename T, int AXIS = -1, bool HAS_BETA = false, bool TRAILING_SYNC = true>
__device__ void motion_cross_mul(T alpha, const T *v, const T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    spatial_detail::motion_cross_mul_impl<T, AXIS, HAS_BETA>(rank, size, alpha, v, x, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Spatial force cross-product matrix: `M = force_cross(v)` (6x6, "crf"/"fx").
 *
 * The dual cross for spatial FORCE vectors: `force_cross(v) = −motion_cross(v)ᵀ
 * = [[ωₓ, v_linₓ],[0, ωₓ]]`. Column-major; always writes all 36 entries (no
 * pre-zeroed-destination variant). Prefer the fused `force_cross_mul`.
 *
 * @tparam T,TRAILING_SYNC  See `motion_cross`.
 * @param v  Spatial motion vector (6 elements, `[ω; v_lin]`).
 * @param M  Output 6x6 matrix (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void force_cross(const T *v, T *M)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    spatial_detail::force_cross_impl(rank, size, v, M);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Fused spatial force cross apply: `y = alpha·(v ×* f) + beta·y`.
 *
 * Row formulas of `force_cross(v)·f`, fused (no 6x6 materialized). Same
 * `(alpha, beta)`/`beta_blend` semantics as `motion_cross_mul`. Tested against
 * the `gemv(force_cross(v), f)` composition.
 *
 * @tparam T,HAS_BETA,TRAILING_SYNC  See `motion_cross_mul`.
 * @param alpha  Scalar on the product.
 * @param v      Spatial motion vector (6 elements, `[ω; v_lin]`).
 * @param f      Spatial force vector (6 elements, `[n; f_lin]`).
 * @param beta   Scalar on the existing `y` (read only when `HAS_BETA`).
 * @param y      Output spatial force vector (6 elements; no aliasing).
 */
template <typename T, bool HAS_BETA = false, bool TRAILING_SYNC = true>
__device__ void force_cross_mul(T alpha, const T *v, const T *f, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    spatial_detail::force_cross_mul_impl<T, HAS_BETA>(rank, size, alpha, v, f, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Operand-swapped force cross matrix: `M = force_cross_dual(f)` (6x6, "icrf").
 *
 * DEFINED by the identity `force_cross(v)·f == force_cross_dual(f)·v` for all
 * `v` — it swaps which operand becomes the matrix, the rearrangement
 * inertia-gradient kernels need (∂/∂q of `v ×* (I·v)` terms). Column-major.
 *
 * @tparam T,TRAILING_SYNC  See `motion_cross`.
 * @param f  Spatial force vector (6 elements, `[n; f_lin]`).
 * @param M  Output 6x6 matrix (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void force_cross_dual(const T *f, T *M)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    spatial_detail::force_cross_dual_impl(rank, size, f, M);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-thread spatial cross ops ─────────────────────────────────────────
namespace thread {
    // One thread owns the whole 6-/36-element result: the SAME row/entry
    // formulas run serially. No barriers, no threadIdx read; register operands
    // fine (all sizes are under the tier's element-count ceiling).

    /** @brief Single-thread motion cross matrix. See `glass::motion_cross`. */
    template <typename T>
    __device__ void motion_cross(const T *v, T *M)
    { spatial_detail::motion_cross_impl(0u, 1u, v, M); }

    /** @brief Single-thread fused motion cross apply. See `glass::motion_cross_mul`. */
    template <typename T, int AXIS = -1, bool HAS_BETA = false>
    __device__ void motion_cross_mul(T alpha, const T *v, const T *x, T beta, T *y)
    { spatial_detail::motion_cross_mul_impl<T, AXIS, HAS_BETA>(0u, 1u, alpha, v, x, beta, y); }

    /** @brief Single-thread force cross matrix. See `glass::force_cross`. */
    template <typename T>
    __device__ void force_cross(const T *v, T *M)
    { spatial_detail::force_cross_impl(0u, 1u, v, M); }

    /** @brief Single-thread fused force cross apply. See `glass::force_cross_mul`. */
    template <typename T, bool HAS_BETA = false>
    __device__ void force_cross_mul(T alpha, const T *v, const T *f, T beta, T *y)
    { spatial_detail::force_cross_mul_impl<T, HAS_BETA>(0u, 1u, alpha, v, f, beta, y); }

    /** @brief Single-thread operand-swapped force cross matrix. See `glass::force_cross_dual`. */
    template <typename T>
    __device__ void force_cross_dual(const T *f, T *M)
    { spatial_detail::force_cross_dual_impl(0u, 1u, f, M); }
}

// ─── single-warp spatial cross ops ───────────────────────────────────────────
namespace warp {
    // One 32-lane warp owns the result: lane-strided over the outputs,
    // `__syncwarp()` close. Outputs must not alias inputs.

    /** @brief Single-warp motion cross matrix. See `glass::motion_cross`. */
    template <typename T>
    __device__ void motion_cross(const T *v, T *M)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        spatial_detail::motion_cross_impl(lane, 32u, v, M);
        __syncwarp();
    }

    /** @brief Single-warp fused motion cross apply. See `glass::motion_cross_mul`. */
    template <typename T, int AXIS = -1, bool HAS_BETA = false>
    __device__ void motion_cross_mul(T alpha, const T *v, const T *x, T beta, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        spatial_detail::motion_cross_mul_impl<T, AXIS, HAS_BETA>(lane, 32u, alpha, v, x, beta, y);
        __syncwarp();
    }

    /** @brief Single-warp force cross matrix. See `glass::force_cross`. */
    template <typename T>
    __device__ void force_cross(const T *v, T *M)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        spatial_detail::force_cross_impl(lane, 32u, v, M);
        __syncwarp();
    }

    /** @brief Single-warp fused force cross apply. See `glass::force_cross_mul`. */
    template <typename T, bool HAS_BETA = false>
    __device__ void force_cross_mul(T alpha, const T *v, const T *f, T beta, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        spatial_detail::force_cross_mul_impl<T, HAS_BETA>(lane, 32u, alpha, v, f, beta, y);
        __syncwarp();
    }

    /** @brief Single-warp operand-swapped force cross matrix. See `glass::force_cross_dual`. */
    template <typename T>
    __device__ void force_cross_dual(const T *f, T *M)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        spatial_detail::force_cross_dual_impl(lane, 32u, f, M);
        __syncwarp();
    }
}
