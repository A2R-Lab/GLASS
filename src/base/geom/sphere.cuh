#pragma once
#include <cstdint>

// ─── sphere distance primitives (robotics ops, geometry family) ──────────────
//
// The narrow-phase kit of sphere-decomposed collision checking (the dominant
// GPU robot-collision representation — cuRobo, HJCD-IK, GRiD's collision
// spheres all use it): sphere-sphere and sphere-box signed distance with
// gradients, and the radius-preserving sphere transform. Semantics match the
// cuRobo kernels these are distilled from; signed distance is POSITIVE when
// clear and NEGATIVE in penetration.
//
// All ops here are PER-CALLER, tier-free: a collision check is inherently one
// (sphere, primitive) pair per calling thread (every surveyed library runs it
// that way), so each function computes its whole small answer serially with no
// `threadIdx` read — correct at any scope, one `glass::` home, no tier
// mirrors. Compose with `smooth_hinge` (proj/interval.cuh) for C¹ collision
// costs and with `quat_rotate`/`quat_to_basis` (lie/quat.cuh) for frame
// changes.

/**
 * @brief Sphere-sphere signed distance: `‖c1 − c2‖ − (r1 + r2)`.
 *
 * Positive = separated, negative = penetrating. Scalar, tier-free.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param c1  Center of sphere 1 (3 elements).
 * @param r1  Radius of sphere 1.
 * @param c2  Center of sphere 2 (3 elements).
 * @param r2  Radius of sphere 2.
 * @return The signed distance.
 */
template <typename T>
__device__ __forceinline__ T sphere_sphere_dist(const T *c1, T r1, const T *c2, T r2)
{
    const T dx = c1[0] - c2[0], dy = c1[1] - c2[1], dz = c1[2] - c2[2];
    return sqrt(dx*dx + dy*dy + dz*dz) - (r1 + r2);
}

/**
 * @brief Sphere-sphere signed distance with the gradient w.r.t. `c1`.
 *
 * `grad = ∂d/∂c1 = (c1 − c2)/‖c1 − c2‖` (the unit separating direction; the
 * gradient w.r.t. `c2` is its negation). Coincident centers (‖·‖ below 1e-12)
 * return a zero gradient — the caller sees the (measure-zero) degenerate case
 * explicitly rather than a NaN. Scalar, tier-free.
 *
 * @tparam T  Scalar type.
 * @param c1,r1,c2,r2  See `sphere_sphere_dist`.
 * @param grad  Output ∂d/∂c1 (3 elements).
 * @return The signed distance.
 */
template <typename T>
__device__ __forceinline__ T sphere_sphere_dist(const T *c1, T r1, const T *c2, T r2,
                                                T *grad)
{
    const T dx = c1[0] - c2[0], dy = c1[1] - c2[1], dz = c1[2] - c2[2];
    const T n = sqrt(dx*dx + dy*dy + dz*dz);
    if (n > static_cast<T>(1e-12)) {
        const T inv = static_cast<T>(1)/n;
        grad[0] = dx*inv; grad[1] = dy*inv; grad[2] = dz*inv;
    } else {
        grad[0] = grad[1] = grad[2] = static_cast<T>(0);
    }
    return n - (r1 + r2);
}

/**
 * @brief Sphere-box signed distance (box frame) with the gradient w.r.t. the center.
 *
 * The box is axis-aligned in ITS OWN frame, centered at the origin with half
 * extents `half[3]`; transform the sphere center into the box frame first for
 * an OBB (`quat_rotate` with the inverse box pose — the cuRobo pattern).
 * Uses the canonical box SDF: with `q_i = |c_i| − half_i`,
 * `d_box = ‖max(q, 0)‖ + min(max_i q_i, 0)`; the sphere distance is
 * `d_box − r`. Negative = penetrating. `grad = ∂d/∂c`: the normalized outward
 * offset outside the box, the (sign-carrying) max-penetration face normal
 * inside — the direction to move the CENTER to increase clearance. Scalar,
 * tier-free.
 *
 * @tparam T  Scalar type.
 * @param c     Sphere center IN THE BOX FRAME (3 elements).
 * @param r     Sphere radius.
 * @param half  Box half extents (3 elements, > 0).
 * @param grad  Output ∂d/∂c (3 elements; box frame).
 * @return The signed distance.
 */
template <typename T>
__device__ __forceinline__ T sphere_box_dist(const T *c, T r, const T *half, T *grad)
{
    const T zero = static_cast<T>(0), one = static_cast<T>(1);
    T q[3], outward[3];
    #pragma unroll
    for (uint32_t i = 0; i < 3; ++i) {
        const T a = (c[i] < zero) ? -c[i] : c[i];
        q[i] = a - half[i];
        outward[i] = (c[i] < zero) ? static_cast<T>(-1) : one;   // sign(c_i), sign(0) = +1
    }
    // outside part: ‖max(q,0)‖ along the sign-restored offset
    T ox = (q[0] > zero) ? q[0] : zero;
    T oy = (q[1] > zero) ? q[1] : zero;
    T oz = (q[2] > zero) ? q[2] : zero;
    const T outside = sqrt(ox*ox + oy*oy + oz*oz);
    // inside part: the least-negative q (0 when any q_i > 0)
    T qmax = q[0]; uint32_t kmax = 0;
    if (q[1] > qmax) { qmax = q[1]; kmax = 1; }
    if (q[2] > qmax) { qmax = q[2]; kmax = 2; }
    const T inside = (qmax < zero) ? qmax : zero;
    if (outside > static_cast<T>(1e-12)) {
        const T inv = one/outside;
        grad[0] = outward[0]*ox*inv;
        grad[1] = outward[1]*oy*inv;
        grad[2] = outward[2]*oz*inv;
    } else {
        // inside (or on the surface): push out through the nearest face
        grad[0] = grad[1] = grad[2] = zero;
        grad[kmax] = outward[kmax];
    }
    return outside + inside - r;
}

/**
 * @brief Sphere-box signed distance (box frame), distance only.
 *
 * See the gradient overload for semantics.
 *
 * @tparam T  Scalar type.
 * @param c     Sphere center IN THE BOX FRAME (3 elements).
 * @param r     Sphere radius.
 * @param half  Box half extents (3 elements, > 0).
 * @return The signed distance.
 */
template <typename T>
__device__ __forceinline__ T sphere_box_dist(const T *c, T r, const T *half)
{
    T grad[3];
    return sphere_box_dist(c, r, half, grad);
}

/**
 * @brief Rigid-transform a sphere: rotate + translate the center, keep the radius.
 *
 * `sph = [cx, cy, cz, r]` (the standard packed GPU collision-sphere layout);
 * `out[0:3] = R(q)·c + p`, `out[3] = r`. `q` must be unit. Scalar, tier-free;
 * `out` may alias `sph`.
 *
 * @tparam T  Scalar type.
 * @tparam L  Quaternion layout of `q` (default `xyzw`).
 * @param q    Unit rotation quaternion (4 elements).
 * @param p    Translation (3 elements).
 * @param sph  Input sphere `[c(3); r]` (4 elements).
 * @param out  Output sphere `[c(3); r]` (4 elements; aliasing allowed).
 */
template <typename T, QuatLayout L = QuatLayout::xyzw>
__device__ __forceinline__ void transform_sphere(const T *q, const T *p,
                                                 const T *sph, T *out)
{
    T c[3]; lie_detail::quat_rotate_core<T, L>(q, sph, c);
    out[0] = c[0] + p[0];
    out[1] = c[1] + p[1];
    out[2] = c[2] + p[2];
    out[3] = sph[3];
}
