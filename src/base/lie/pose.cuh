#pragma once
#include <cstdint>

// ─── pose-error metrics (robotics ops, Lie family) ───────────────────────────
//
// The goal-cost kit of trajectory optimization, batched IK, and motion
// generation: the shortest-path rotation error between two orientations, the
// decoupled 6-D pose error, and the geodesic angle metric. Every GPU motion
// stack carries some version of these (cuRobo's pose_distance kernels, GATO's
// EE goal costs, HJCD-IK's orientation residual) with silently different sign
// and frame conventions — this is the one pinned home.
//
// CONVENTIONS:
//   * Errors are LOCAL (right/body convention): `quat_error(q, q_des) =
//     log(q_des⁻¹ ⊗ q)` — the tangent step that retracts q_des onto q
//     (`quat_retract(q_des, e) == q`); GTSAM localCoordinates / manif rminus.
//     Swap the arguments to negate.
//   * The double cover is folded: the error is always the SHORTEST path
//     (|e| ≤ π), regardless of the stored signs of q and q_des.
//   * `pose_error` is the DECOUPLED R³ × SO(3) error `[p − p_des ; rot]`
//     (linear-first, matching the SE(3)-tangent `[ρ; φ]` block order in
//     lie/se3.cuh) — what GPU goal costs actually minimize, NOT the coupled
//     SE(3) log (whose translation part twists through V(φ)⁻¹).
//   * Poses pack as `[p(3); q(4)]` (the se3_retract 7-element layout); the
//     quaternion layout is the usual compile-time `QuatLayout` tag.
//   * Exact tangent gradients when needed: for `e = quat_error(q, q_des)`,
//     `∂e/∂(local tangent of q) = Jr(e)⁻¹` — compose with
//     `so3_right_jacobian_inv(e)` from lie/so3.cuh; the translation block's
//     gradient is the identity.
//
// Tiers: `quat_error`/`pose_error` are array ops (redundant serial core +
// strided copy-out, block/warp/thread); `quat_angle` is a tier-free scalar.
// Inputs must be unit quaternions; outputs must not alias inputs at
// block/warp scope.

namespace lie_detail {
    // serial core: e = log(q_des⁻¹ ⊗ q), shortest path (see quat_log_core).
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void quat_error_core(const T *q, const T *q_des, T *e) {
        using QL = quat_detail::layout<L>;
        T dc[4];
        dc[QL::X] = -q_des[QL::X]; dc[QL::Y] = -q_des[QL::Y];
        dc[QL::Z] = -q_des[QL::Z]; dc[QL::W] =  q_des[QL::W];
        T qe[4]; quat_detail::quat_mul_core<T, L>(dc, q, qe);
        quat_detail::quat_log_core<T, L>(qe, e);
    }

    // serial core: e = [p − p_des ; quat_error] (linear-first).
    template <typename T, QuatLayout L>
    __device__ __forceinline__ void pose_error_core(const T *pose, const T *pose_des, T *e) {
        e[0] = pose[0] - pose_des[0];
        e[1] = pose[1] - pose_des[1];
        e[2] = pose[2] - pose_des[2];
        quat_error_core<T, L>(pose + 3, pose_des + 3, e + 3);
    }
} // namespace lie_detail

/**
 * @brief Shortest-path rotation error: `e = log(q_des⁻¹ ⊗ q)` (3-vector).
 *
 * The LOCAL (body-frame) tangent from `q_des` to `q`:
 * `quat_retract(q_des, e) == q`, `|e| ≤ π` always (double cover folded). The
 * orientation residual of IK / goal costs; its exact tangent Jacobian w.r.t.
 * `q`'s local perturbation is `Jr(e)⁻¹` (`so3_right_jacobian_inv`). Both
 * inputs must be unit. NumPy equivalent (xyzw, scipy):
 * `(Rotation.from_quat(q_des).inv() * Rotation.from_quat(q)).as_rotvec()`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam L  Quaternion storage layout (default `QuatLayout::xyzw`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param q      Current unit quaternion (4 elements).
 * @param q_des  Desired unit quaternion (4 elements).
 * @param e      Output rotation-error vector (3 elements; no aliasing).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void quat_error(const T *q, const T *q_des, T *e)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[3]; lie_detail::quat_error_core<T, L>(q, q_des, tmp);
    quat_detail::copy_out<T, 3>(rank, size, tmp, e);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Decoupled 6-D pose error: `e = [p − p_des ; quat_error(q, q_des)]`.
 *
 * Poses are `[p(3); q(4)]` (the `se3_retract` layout); the error is
 * linear-first, matching the SE(3)-tangent `[ρ; φ]` block order — but it is
 * the R³ × SO(3) product error every GPU goal cost minimizes, NOT the coupled
 * SE(3) log (see the file header). Exact tangent Jacobian: identity on the
 * translation block, `Jr(e_rot)⁻¹` on the rotation block.
 *
 * @tparam T,L,TRAILING_SYNC  See `quat_error`.
 * @param pose      Current pose `[p; q]` (7 elements, unit q).
 * @param pose_des  Desired pose `[p; q]` (7 elements, unit q).
 * @param e         Output error (6 elements, `[dp; drot]`; no aliasing).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, bool TRAILING_SYNC = true>
__device__ void pose_error(const T *pose, const T *pose_des, T *e)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T tmp[6]; lie_detail::pose_error_core<T, L>(pose, pose_des, tmp);
    quat_detail::copy_out<T, 6>(rank, size, tmp, e);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Geodesic angle between two orientations: `θ = |quat_error(q, q_des)|`.
 *
 * Computed as `2·atan2(|vec(q_des⁻¹ ⊗ q)|, |w(q_des⁻¹ ⊗ q)|)` — stable across
 * the whole range (no acos precision cliff near θ = 0), double cover folded,
 * `θ ∈ [0, π]`. Both inputs must be unit. Scalar, tier-free. NumPy
 * equivalent (xyzw, scipy):
 * `(Rotation.from_quat(q_des).inv() * Rotation.from_quat(q)).magnitude()`.
 *
 * @tparam T  Scalar type.
 * @tparam L  Quaternion storage layout (default `QuatLayout::xyzw`).
 * @param q      Current unit quaternion (4 elements).
 * @param q_des  Desired unit quaternion (4 elements).
 * @return The geodesic angle (radians).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw>
__device__ __forceinline__ T quat_angle(const T *q, const T *q_des)
{
    using QL = quat_detail::layout<L>;
    T dc[4];
    dc[QL::X] = -q_des[QL::X]; dc[QL::Y] = -q_des[QL::Y];
    dc[QL::Z] = -q_des[QL::Z]; dc[QL::W] =  q_des[QL::W];
    T qe[4]; quat_detail::quat_mul_core<T, L>(dc, q, qe);
    const T n = sqrt(qe[QL::X]*qe[QL::X] + qe[QL::Y]*qe[QL::Y] + qe[QL::Z]*qe[QL::Z]);
    const T w = (qe[QL::W] < static_cast<T>(0)) ? -qe[QL::W] : qe[QL::W];
    return static_cast<T>(2)*atan2(n, w);
}

// ─── single-thread pose errors ───────────────────────────────────────────────
namespace thread {
    /** @brief Single-thread shortest-path rotation error. See `glass::quat_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_error(const T *q, const T *q_des, T *e)
    {
        T tmp[3]; lie_detail::quat_error_core<T, L>(q, q_des, tmp);
        e[0] = tmp[0]; e[1] = tmp[1]; e[2] = tmp[2];
    }

    /** @brief Single-thread decoupled 6-D pose error. See `glass::pose_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void pose_error(const T *pose, const T *pose_des, T *e)
    {
        T tmp[6]; lie_detail::pose_error_core<T, L>(pose, pose_des, tmp);
        for (uint32_t i = 0; i < 6; i++) e[i] = tmp[i];
    }
}

// ─── single-warp pose errors ─────────────────────────────────────────────────
namespace warp {
    /** @brief Single-warp shortest-path rotation error. See `glass::quat_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void quat_error(const T *q, const T *q_des, T *e)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[3]; lie_detail::quat_error_core<T, L>(q, q_des, tmp);
        quat_detail::copy_out<T, 3>(lane, 32u, tmp, e);
        __syncwarp();
    }

    /** @brief Single-warp decoupled 6-D pose error. See `glass::pose_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw>
    __device__ void pose_error(const T *pose, const T *pose_des, T *e)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        T tmp[6]; lie_detail::pose_error_core<T, L>(pose, pose_des, tmp);
        quat_detail::copy_out<T, 6>(lane, 32u, tmp, e);
        __syncwarp();
    }
}
