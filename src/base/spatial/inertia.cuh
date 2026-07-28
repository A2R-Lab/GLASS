#pragma once
#include <cstdint>

// ─── spatial rigid-body inertia from the 10 standard parameters ──────────────
//
// The third leg of the spatial family (with cross.cuh and transform.cuh): the
// 6x6 spatial inertia assembled from — and applied directly from — the 10
// standard inertial parameters, the frozen "regressor basis" every sysID /
// payload / domain-randomization pipeline carries per body (GRiD's runtime
// inertia table and cuRobo's `spatial_inertia_times_vec` both use exactly
// this on-the-fly form so the 36-entry matrix never lives in memory).
//
// CONVENTION — the parameter 10-vector is GRiD's runtime table layout:
//   `pi = [m, h(3), I_O(6)]` with
//   * `m`     — mass,
//   * `h`     — first moment `m·c` (c = COM position in the body frame),
//   * `I_O`   — rotational inertia ABOUT THE BODY-FRAME ORIGIN (not the COM),
//               packed `[Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`.
//   NOTE this differs from Pinocchio's dynamic-parameter ordering
//   (`[m, h, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]` — Iyy and Ixz swapped); permute
//   when crossing.
//
// The spatial inertia (angular-first blocks, column-major) is
//   `I = [[I_O, [h]ₓ], [[h]ₓᵀ, m·1₃]]`,
// and the fused apply for `v = [ω; v_lin]` is
//   `I·v = [I_O·ω + h × v_lin ; m·v_lin − h × ω]`
// (a spatial FORCE vector `[n; f_lin]`). The apply carries `(alpha, beta)` +
// `beta_blend` like the other fused spatial ops — `HAS_BETA = true, beta = 1`
// is the RNEA momentum accumulation. Tiers: redundant core + strided blended
// copy-out (fused) / entry-strided (materializer). Outputs must not alias
// inputs at block/warp scope.

namespace spatial_detail {
    // Entry (i, j) of the 6x6 spatial inertia, column-major, from pi[10].
    template <typename T>
    __device__ __forceinline__ T inertia_entry(uint32_t i, uint32_t j, const T *pi) {
        const T zero = static_cast<T>(0);
        const uint32_t bi = i / 3, bj = j / 3, r = i % 3, c = j % 3;
        if (bi == 0 && bj == 0) {                    // I_O (symmetric from 6 params)
            const uint32_t lo = (r < c) ? r : c, hi = (r < c) ? c : r;
            // packed [Ixx,Ixy,Ixz,Iyy,Iyz,Izz]: (0,0)→4 (0,1)→5 (0,2)→6 (1,1)→7 (1,2)→8 (2,2)→9
            return pi[4 + lo*2 + hi - ((lo == 2) ? 1 : 0)];
        }
        if (bi == 1 && bj == 1)                      // m·1₃
            return (r == c) ? pi[0] : zero;
        // [h]ₓ (top-right) or [h]ₓᵀ = −[h]ₓ (bottom-left)
        T s;
        switch (c*3 + r) {
            case 1: s =  pi[3]; break;  case 2: s = -pi[2]; break;
            case 3: s = -pi[3]; break;  case 5: s =  pi[1]; break;
            case 6: s =  pi[2]; break;  case 7: s = -pi[1]; break;
            default: return zero;
        }
        return (bi == 0) ? s : -s;
    }

    template <typename T>
    __device__ __forceinline__ void inertia_mat_impl(uint32_t rank, uint32_t size,
                                                     const T *pi, T *M) {
        for (uint32_t i = rank; i < 36; i += size)
            M[i] = inertia_entry<T>(i % 6, i / 6, pi);
    }

    // serial core: f = I(pi)·v into a register 6-vector.
    template <typename T>
    __device__ __forceinline__ void spatial_inertia_mul_core(const T *pi, const T *v, T *f) {
        const T m = pi[0];
        const T hx = pi[1], hy = pi[2], hz = pi[3];
        const T Ixx = pi[4], Ixy = pi[5], Ixz = pi[6], Iyy = pi[7], Iyz = pi[8], Izz = pi[9];
        // n = I_O·ω + h × v_lin
        f[0] = Ixx*v[0] + Ixy*v[1] + Ixz*v[2] + hy*v[5] - hz*v[4];
        f[1] = Ixy*v[0] + Iyy*v[1] + Iyz*v[2] + hz*v[3] - hx*v[5];
        f[2] = Ixz*v[0] + Iyz*v[1] + Izz*v[2] + hx*v[4] - hy*v[3];
        // f_lin = m·v_lin − h × ω
        f[3] = m*v[3] - (hy*v[2] - hz*v[1]);
        f[4] = m*v[4] - (hz*v[0] - hx*v[2]);
        f[5] = m*v[5] - (hx*v[1] - hy*v[0]);
    }
} // namespace spatial_detail

/**
 * @brief Spatial inertia matrix from the 10 standard parameters (6x6).
 *
 * `M = [[I_O, [h]ₓ], [[h]ₓᵀ, m·1₃]]` with `pi = [m, h(3), I_O(6)]` (see the
 * file header for the packing and the Pinocchio-ordering caveat).
 * Column-major; symmetric by construction. Prefer the fused
 * `spatial_inertia_mul` when only a product is needed.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param pi  The 10 inertial parameters `[m, h(3), Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`.
 * @param M   Output 6x6 spatial inertia (36 elements, column-major; no aliasing).
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void spatial_inertia(const T *pi, T *M)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    spatial_detail::inertia_mat_impl(rank, size, pi, M);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Fused spatial inertia apply: `f = alpha·I(pi)·v + beta·f`.
 *
 * `I(pi)·v = [I_O·ω + h × v_lin ; m·v_lin − h × ω]` — momentum from motion
 * (RNEA's `I·v` and `I·a` terms) straight from the parameter 10-vector; the
 * 6x6 never materializes. `beta` is only read when `HAS_BETA` (BLAS beta==0
 * semantics via `beta_blend`). Tested against the
 * `gemv(spatial_inertia(pi), v)` composition.
 *
 * @tparam T  Scalar type.
 * @tparam HAS_BETA  Read/accumulate into `f` (default false = overwrite).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar on the product.
 * @param pi     The 10 inertial parameters (see `spatial_inertia`).
 * @param v      Spatial motion vector (6 elements, `[ω; v_lin]`).
 * @param beta   Scalar on the existing `f` (read only when `HAS_BETA`).
 * @param f      Output spatial force vector (6 elements; no aliasing).
 */
template <typename T, bool HAS_BETA = false, bool TRAILING_SYNC = true>
__device__ void spatial_inertia_mul(T alpha, const T *pi, const T *v, T beta, T *f)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[6]; spatial_detail::spatial_inertia_mul_core(pi, v, tmp);
    spatial_detail::blend_out<T, 6, HAS_BETA>(rank, size, alpha, tmp, beta, f);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-thread spatial inertia ───────────────────────────────────────────
namespace thread {
    /** @brief Single-thread spatial inertia matrix. See `glass::spatial_inertia`. */
    template <typename T>
    __device__ void spatial_inertia(const T *pi, T *M)
    { spatial_detail::inertia_mat_impl(0u, 1u, pi, M); }

    /** @brief Single-thread fused spatial inertia apply. See `glass::spatial_inertia_mul`. */
    template <typename T, bool HAS_BETA = false>
    __device__ void spatial_inertia_mul(T alpha, const T *pi, const T *v, T beta, T *f)
    {
        T tmp[6]; spatial_detail::spatial_inertia_mul_core(pi, v, tmp);
        spatial_detail::blend_out<T, 6, HAS_BETA>(0u, 1u, alpha, tmp, beta, f);
    }
}

// ─── single-warp spatial inertia ─────────────────────────────────────────────
namespace warp {
    /** @brief Single-warp spatial inertia matrix. See `glass::spatial_inertia`. */
    template <typename T>
    __device__ void spatial_inertia(const T *pi, T *M)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        spatial_detail::inertia_mat_impl(lane, 32u, pi, M);
        __syncwarp();
    }

    /** @brief Single-warp fused spatial inertia apply. See `glass::spatial_inertia_mul`. */
    template <typename T, bool HAS_BETA = false>
    __device__ void spatial_inertia_mul(T alpha, const T *pi, const T *v, T beta, T *f)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[6]; spatial_detail::spatial_inertia_mul_core(pi, v, tmp);
        spatial_detail::blend_out<T, 6, HAS_BETA>(lane, 32u, alpha, tmp, beta, f);
        __syncwarp();
    }
}
