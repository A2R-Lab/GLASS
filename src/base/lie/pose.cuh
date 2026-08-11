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
//   * The error FRAME is a compile-time `ErrorFrame` tag (mirroring the
//     `QuatLayout` pattern; pinocchio's ReferenceFrame::{LOCAL, WORLD} is the
//     field precedent). BOTH frames express the tangent step FROM q_des TO q
//     — only the frame it is resolved in differs:
//       LOCAL (default) : e = log(q_des⁻¹ ⊗ q)  — body frame of q_des;
//                         `quat_retract(q_des, e) == q` (right retract);
//                         GTSAM localCoordinates / manif rminus. Pair with
//                         BODY-frame Jacobians.
//       WORLD           : e = log(q ⊗ q_des⁻¹)  — world/spatial frame;
//                         `quat_mul(quat_exp(e), q_des) == q` (left retract);
//                         e_WORLD = R(q_des)·e_LOCAL. Pair with WORLD-frame
//                         (geometric) Jacobians — IK / visual servoing /
//                         task-space control.
//     Swapping the arguments negates the error IN EITHER frame (a residual
//     written as `log(q_des ⊗ q⁻¹)` — e.g. HJCD-IK's — is
//     `quat_error<…, ErrorFrame::WORLD>(q_des, q)`, arguments swapped).
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

/**
 * @brief Compile-time frame tag for the pose-error family.
 *
 * `LOCAL` (default) — the error resolved in the body frame of `q_des`
 * (`log(q_des⁻¹ ⊗ q)`; pair with body-frame Jacobians). `WORLD` — the same
 * from-desired-to-current step resolved in the world frame
 * (`log(q ⊗ q_des⁻¹)`; pair with world-frame geometric Jacobians). Field
 * precedent: pinocchio `ReferenceFrame::{LOCAL, WORLD}`.
 */
enum class ErrorFrame { LOCAL, WORLD };

namespace lie_detail {
    // serial core: LOCAL e = log(q_des⁻¹ ⊗ q); WORLD e = log(q ⊗ q_des⁻¹) —
    // both shortest path (see quat_log_core), both from-desired-to-current.
    template <typename T, QuatLayout L, ErrorFrame F = ErrorFrame::LOCAL>
    __device__ __forceinline__ void quat_error_core(const T *q, const T *q_des, T *e) {
        using QL = lie_detail::layout<L>;
        T dc[4];
        dc[QL::X] = -q_des[QL::X]; dc[QL::Y] = -q_des[QL::Y];
        dc[QL::Z] = -q_des[QL::Z]; dc[QL::W] =  q_des[QL::W];
        T qe[4];
        if constexpr (F == ErrorFrame::LOCAL) lie_detail::quat_mul_core<T, L>(dc, q, qe);
        else                                  lie_detail::quat_mul_core<T, L>(q, dc, qe);
        lie_detail::quat_log_core<T, L>(qe, e);
    }

    // serial core: e = [p − p_des ; quat_error] (linear-first).
    template <typename T, QuatLayout L, ErrorFrame F = ErrorFrame::LOCAL>
    __device__ __forceinline__ void pose_error_core(const T *pose, const T *pose_des, T *e) {
        e[0] = pose[0] - pose_des[0];
        e[1] = pose[1] - pose_des[1];
        e[2] = pose[2] - pose_des[2];
        quat_error_core<T, L, F>(pose + 3, pose_des + 3, e + 3);
    }
} // namespace lie_detail

/**
 * @brief Shortest-path rotation error, frame-tagged: LOCAL (default)
 *        `e = log(q_des⁻¹ ⊗ q)`; WORLD `e = log(q ⊗ q_des⁻¹)`.
 *
 * Both frames are the tangent step FROM `q_des` TO `q` (`|e| ≤ π` always,
 * double cover folded): LOCAL satisfies `quat_retract(q_des, e) == q` and
 * pairs with body-frame Jacobians (exact tangent Jacobian `Jr(e)⁻¹`,
 * `so3_right_jacobian_inv`); WORLD satisfies
 * `quat_mul(quat_exp(e), q_des) == q`, equals `R(q_des)·e_LOCAL`, and pairs
 * with world-frame geometric Jacobians (IK / visual servoing / task-space
 * control). Swapping the arguments negates the error in either frame. Both
 * inputs must be unit. NumPy equivalents (xyzw, scipy): LOCAL
 * `(Rotation.from_quat(q_des).inv() * Rotation.from_quat(q)).as_rotvec()`;
 * WORLD `(Rotation.from_quat(q) * Rotation.from_quat(q_des).inv()).as_rotvec()`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam L  Quaternion storage layout (default `QuatLayout::xyzw`).
 * @tparam F  Error frame (default `ErrorFrame::LOCAL`).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param q      Current unit quaternion (4 elements).
 * @param q_des  Desired unit quaternion (4 elements).
 * @param e      Output rotation-error vector (3 elements; no aliasing).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, ErrorFrame F = ErrorFrame::LOCAL,
          bool TRAILING_SYNC = true>
__device__ void quat_error(const T *q, const T *q_des, T *e)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[3]; lie_detail::quat_error_core<T, L, F>(q, q_des, tmp);
    lie_detail::copy_out<T, 3>(rank, size, tmp, e);
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
 * @tparam T,L,F,TRAILING_SYNC  See `quat_error` (the frame tag applies to the
 *                              rotation block; the translation difference is
 *                              world-frame in both).
 * @param pose      Current pose `[p; q]` (7 elements, unit q).
 * @param pose_des  Desired pose `[p; q]` (7 elements, unit q).
 * @param e         Output error (6 elements, `[dp; drot]`; no aliasing).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw, ErrorFrame F = ErrorFrame::LOCAL,
          bool TRAILING_SYNC = true>
__device__ void pose_error(const T *pose, const T *pose_des, T *e)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    T tmp[6]; lie_detail::pose_error_core<T, L, F>(pose, pose_des, tmp);
    lie_detail::copy_out<T, 6>(rank, size, tmp, e);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Geodesic angle between two orientations: `θ = |quat_error(q, q_des)|`.
 *
 * Computed as `2·atan2(|vec(q_des⁻¹ ⊗ q)|, |w(q_des⁻¹ ⊗ q)|)` — stable across
 * the whole range (no acos precision cliff near θ = 0), double cover folded,
 * `θ ∈ [0, π]`. FRAME-INVARIANT: the LOCAL and WORLD errors are the same
 * vector resolved in different frames, so their magnitude — this angle — is
 * identical (no `ErrorFrame` tag by design). Both inputs must be unit.
 * Scalar, tier-free. NumPy
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
    using QL = lie_detail::layout<L>;
    T dc[4];
    dc[QL::X] = -q_des[QL::X]; dc[QL::Y] = -q_des[QL::Y];
    dc[QL::Z] = -q_des[QL::Z]; dc[QL::W] =  q_des[QL::W];
    T qe[4]; lie_detail::quat_mul_core<T, L>(dc, q, qe);
    const T n = sqrt(qe[QL::X]*qe[QL::X] + qe[QL::Y]*qe[QL::Y] + qe[QL::Z]*qe[QL::Z]);
    const T w = (qe[QL::W] < static_cast<T>(0)) ? -qe[QL::W] : qe[QL::W];
    return static_cast<T>(2)*atan2(n, w);
}

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __shfl_*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /** @brief Single-warp frame-tagged rotation error. See `glass::quat_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, ErrorFrame F = ErrorFrame::LOCAL>
    __device__ void quat_error(const T *q, const T *q_des, T *e)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[3]; lie_detail::quat_error_core<T, L, F>(q, q_des, tmp);
        lie_detail::copy_out<T, 3>(lane, 32u, tmp, e);
        __syncwarp();
    }

    /** @brief Single-warp decoupled 6-D pose error. See `glass::pose_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, ErrorFrame F = ErrorFrame::LOCAL>
    __device__ void pose_error(const T *pose, const T *pose_des, T *e)
    {
        uint32_t lane = (flat_rank()) & 31;
        T tmp[6]; lie_detail::pose_error_core<T, L, F>(pose, pose_des, tmp);
        lie_detail::copy_out<T, 6>(lane, 32u, tmp, e);
        __syncwarp();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// thread:: — one problem per thread (serial, register-resident)
// ═══════════════════════════════════════════════════════════════════════

namespace thread {
    /** @brief Single-thread frame-tagged rotation error. See `glass::quat_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, ErrorFrame F = ErrorFrame::LOCAL>
    __device__ void quat_error(const T *q, const T *q_des, T *e)
    {
        T tmp[3]; lie_detail::quat_error_core<T, L, F>(q, q_des, tmp);
        e[0] = tmp[0]; e[1] = tmp[1]; e[2] = tmp[2];
    }

    /** @brief Single-thread decoupled 6-D pose error. See `glass::pose_error`. */
    template <typename T, QuatLayout L = QuatLayout::xyzw, ErrorFrame F = ErrorFrame::LOCAL>
    __device__ void pose_error(const T *pose, const T *pose_des, T *e)
    {
        T tmp[6]; lie_detail::pose_error_core<T, L, F>(pose, pose_des, tmp);
        for (uint32_t i = 0; i < 6; i++) e[i] = tmp[i];
    }
}
