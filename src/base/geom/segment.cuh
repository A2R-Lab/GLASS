#pragma once
#include <cstdint>

// ─── segment-segment closest points (robotics ops, geometry family) ──────────
//
// Closest points between two 3-D line segments — the capsule-capsule /
// swept-sphere distance core (capsule distance = segment distance − radii).
// Ericson's clamped-parametric algorithm (Real-Time Collision Detection §5.1.9)
// with explicit degenerate-segment and near-parallel epsilon guards — the two
// classic failure modes of naive ports.
//
// PER-CALLER, tier-free (see geom/sphere.cuh's header note).

/**
 * @brief Closest points between segments `[p1, q1]` and `[p2, q2]`.
 *
 * Returns the SQUARED distance between the closest points and writes the
 * parameters `s, t ∈ [0, 1]` plus the points themselves
 * (`c1 = p1 + s·(q1−p1)`, `c2 = p2 + t·(q2−p2)`). Degenerate (point-like)
 * segments and near-parallel pairs are handled via the `1e-12`-scaled guards
 * (Ericson §5.1.9); parallel overlapping segments return one valid
 * minimizing pair. Capsule-capsule signed distance is
 * `sqrt(result) − (r1 + r2)`. Scalar, tier-free.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param p1,q1  Endpoints of segment 1 (3 elements each).
 * @param p2,q2  Endpoints of segment 2 (3 elements each).
 * @param s   Output parameter on segment 1 (in [0, 1]).
 * @param t   Output parameter on segment 2 (in [0, 1]).
 * @param c1  Output closest point on segment 1 (3 elements).
 * @param c2  Output closest point on segment 2 (3 elements).
 * @return The squared distance `‖c1 − c2‖²`.
 */
template <typename T>
__device__ __forceinline__ T segment_segment_closest(const T *p1, const T *q1,
                                                     const T *p2, const T *q2,
                                                     T &s, T &t, T *c1, T *c2)
{
    const T zero = static_cast<T>(0), one = static_cast<T>(1);
    const T eps = static_cast<T>(1e-12);
    T d1[3], d2[3], r[3];
    #pragma unroll
    for (uint32_t i = 0; i < 3; ++i) {
        d1[i] = q1[i] - p1[i];
        d2[i] = q2[i] - p2[i];
        r[i]  = p1[i] - p2[i];
    }
    const T a = d1[0]*d1[0] + d1[1]*d1[1] + d1[2]*d1[2];   // ‖d1‖²
    const T e = d2[0]*d2[0] + d2[1]*d2[1] + d2[2]*d2[2];   // ‖d2‖²
    const T f = d2[0]*r[0] + d2[1]*r[1] + d2[2]*r[2];
    if (a <= eps && e <= eps) {           // both segments degenerate to points
        s = t = zero;
    } else if (a <= eps) {                // segment 1 is a point
        s = zero;
        t = f/e;
        t = (t < zero) ? zero : (t > one ? one : t);
    } else {
        const T c = d1[0]*r[0] + d1[1]*r[1] + d1[2]*r[2];
        if (e <= eps) {                   // segment 2 is a point
            t = zero;
            s = -c/a;
            s = (s < zero) ? zero : (s > one ? one : s);
        } else {                          // general case
            const T b = d1[0]*d2[0] + d1[1]*d2[1] + d1[2]*d2[2];
            const T denom = a*e - b*b;    // >= 0; ~0 when near-parallel
            s = (denom > eps*a*e) ? (b*f - c*e)/denom : zero;
            s = (s < zero) ? zero : (s > one ? one : s);
            t = (b*s + f)/e;
            if (t < zero) {
                t = zero;
                s = -c/a;
                s = (s < zero) ? zero : (s > one ? one : s);
            } else if (t > one) {
                t = one;
                s = (b - c)/a;
                s = (s < zero) ? zero : (s > one ? one : s);
            }
        }
    }
    T dx = zero, dy = zero, dz = zero;
    #pragma unroll
    for (uint32_t i = 0; i < 3; ++i) {
        c1[i] = p1[i] + s*d1[i];
        c2[i] = p2[i] + t*d2[i];
    }
    dx = c1[0] - c2[0]; dy = c1[1] - c2[1]; dz = c1[2] - c2[2];
    return dx*dx + dy*dy + dz*dz;
}
