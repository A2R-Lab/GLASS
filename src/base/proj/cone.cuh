#pragma once
#include <cstdint>

// ─── second-order (Lorentz) cone ops (robotics ops, projection family) ───────
//
// The convex-cone toolkit trajectory optimizers need for friction-cone /
// thrust-cone constraints: Euclidean projection onto the second-order cone
// `K = {g : ‖g[1:m]‖ ≤ g[0]}` (row 0 = axis), the cone violation metric, and
// the conic PHR augmented-Lagrangian value. Promoted from GATO's cone wave
// (bsqp `rowgroups.cuh`), where the set is numpy-oracle validated; no surveyed
// robotics GPU library ships a SOC projection — this is net-new shared ground.
//
// `soc_tail_norm` / `soc_violation` / `al_soc_value` are SCALAR-RETURNING,
// tier-free ops: every calling thread computes the (deterministic, serial)
// answer for itself — correct at any scope, so they live once at `glass::`
// with no tier mirrors. `soc_project` writes an m-vector and has the standard
// block/warp/thread forms: each active thread recomputes the (serial, small-m)
// tail norm redundantly and strides the writes — thread-count invariant
// because the scalar prelude is identical on every thread.

/**
 * @brief Norm of the cone tail: `‖g[1:m]‖`.
 *
 * Serial per-caller scalar (tier-free — see the header note). The `m ≤ ~16`
 * regime these ops serve makes a serial loop the right shape.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param g  Cone-space vector (m elements; row 0 is the axis).
 * @param m  Vector length (axis + m−1 tail rows).
 * @return `sqrt(Σ_{i>=1} g[i]²)`.
 */
template <typename T>
__device__ __forceinline__ T soc_tail_norm(const T *g, int32_t m)
{
    T s = static_cast<T>(0);
    for (int32_t i = 1; i < m; i++) s += g[i]*g[i];
    return sqrt(s);
}

/**
 * @brief Second-order-cone violation: `max(0, ‖g[1:]‖ − g[0])`.
 *
 * The margin metric (0 inside the cone). NOT the Euclidean distance to K —
 * outside both K and its polar the distance divides this by √2. Scalar,
 * tier-free.
 *
 * @tparam T  Scalar type.
 * @param g  Cone-space vector (m elements).
 * @param m  Vector length.
 * @return The violation (>= 0).
 */
template <typename T>
__device__ __forceinline__ T soc_violation(const T *g, int32_t m)
{
    const T v = soc_tail_norm(g, m) - g[0];
    return v > static_cast<T>(0) ? v : static_cast<T>(0);
}

namespace proj_detail {
    // tier-shared body: p = Π_K(w). Three cases (r = tail norm):
    //   r <= w0  : p = w                        (inside K)
    //   r <= −w0 : p = 0                        (inside the polar cone −K°)
    //   else     : p = ((w0+r)/2)·(1, w̄/r)     (project onto the boundary)
    // Every thread computes r redundantly (identical serial loop → identical
    // value on every thread) then strides the m writes.
    template <typename T>
    __device__ __forceinline__ void soc_project_impl(uint32_t rank, uint32_t size,
                                                     const T *w, T *p, int32_t m) {
        const T r = soc_tail_norm(w, m);
        if (r <= w[0]) {
            for (int32_t i = (int32_t)rank; i < m; i += (int32_t)size) p[i] = w[i];
        } else if (r <= -w[0]) {
            for (int32_t i = (int32_t)rank; i < m; i += (int32_t)size) p[i] = static_cast<T>(0);
        } else {
            const T half_sum = static_cast<T>(0.5)*(w[0] + r);
            const T scale = half_sum / r;   // r > 0 here
            for (int32_t i = (int32_t)rank; i < m; i += (int32_t)size)
                p[i] = (i == 0) ? half_sum : scale*w[i];
        }
    }
} // namespace proj_detail

/**
 * @brief Euclidean projection onto the second-order cone: `p = Π_K(w)`.
 *
 * `K = {g : ‖g[1:m]‖ ≤ g[0]}`. Idempotent; fixes points of K; maps the polar
 * cone to 0; on the intermediate region returns the boundary point
 * `((w0+r)/2)·(1, w̄/r)`. `w` and `p` MAY alias at thread:: scope (GATO's
 * calling pattern); at block/warp scope they must not (strided writers race a
 * reader). m = 1 degenerates to `max(0, ·)` — the scalar hinge.
 *
 * @tparam T  Scalar type.
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param w  Input vector (m elements).
 * @param p  Output projection (m elements).
 * @param m  Vector length.
 */
template <typename T, bool TRAILING_SYNC = true>
__device__ void soc_project(const T *w, T *p, int32_t m)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    proj_detail::soc_project_impl(rank, size, w, p, m);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Conic PHR augmented-Lagrangian value:
 *        `φ = (‖Π_K(λ − ρg)‖² − ‖λ‖²) / (2ρ)`.
 *
 * The exact conic generalization of the hinge PHR term (m = 1 reduces to the
 * `g >= 0` hinge). C¹ in `g`; `∇_g φ = −Π_K(λ − ρg)`; GN Hessian = ρ·JΠ on
 * the active set. Computed WITHOUT materializing the projection: `‖Π_K(w)‖²`
 * follows from the case split (inside: `‖w‖²`; polar: 0; boundary: `2α²` with
 * `α = (w0+r)/2`). Scalar, tier-free.
 *
 * @tparam T  Scalar type.
 * @param g    Constraint values in cone space (m elements).
 * @param lam  Multiplier estimate (m elements).
 * @param rho  Penalty parameter (> 0).
 * @param m    Vector length.
 * @return The AL value contribution.
 */
template <typename T>
__device__ __forceinline__ T al_soc_value(const T *g, const T *lam, T rho, int32_t m)
{
    T lam_sq = static_cast<T>(0), tail_sq = static_cast<T>(0);
    const T w0 = lam[0] - rho*g[0];
    for (int32_t i = 0; i < m; i++) lam_sq += lam[i]*lam[i];
    for (int32_t i = 1; i < m; i++) {
        const T wi = lam[i] - rho*g[i];
        tail_sq += wi*wi;
    }
    const T r = sqrt(tail_sq);
    T p_sq;
    if (r <= w0)       p_sq = w0*w0 + tail_sq;                    // inside K
    else if (r <= -w0) p_sq = static_cast<T>(0);                  // polar cone
    else {
        const T a = static_cast<T>(0.5)*(w0 + r);                 // boundary
        p_sq = static_cast<T>(2)*a*a;
    }
    return (p_sq - lam_sq) / (static_cast<T>(2)*rho);
}

// ─── single-thread / single-warp soc_project ─────────────────────────────────
namespace thread {
    /** @brief Single-thread SOC projection (aliasing-safe). See `glass::soc_project`. */
    template <typename T>
    __device__ void soc_project(const T *w, T *p, int32_t m)
    { proj_detail::soc_project_impl(0u, 1u, w, p, m); }
}

namespace warp {
    /** @brief Single-warp SOC projection. See `glass::soc_project`. */
    template <typename T>
    __device__ void soc_project(const T *w, T *p, int32_t m)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        proj_detail::soc_project_impl(lane, 32u, w, p, m);
        __syncwarp();
    }
}
