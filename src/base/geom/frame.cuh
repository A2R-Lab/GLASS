#pragma once
#include <cstdint>

// ─── orthonormal frame from a vector (robotics ops, geometry family) ─────────
//
// Build a right-handed orthonormal basis `{t, b, n}` from a single unit
// vector `n` — the contact-frame / cone-basis / Unit3-tangent primitive every
// simulator and bearing-factor implementation hand-rolls, and a classic
// branch trap (the naive "cross with the smallest axis" construction is
// discontinuous and division-happy near coordinate axes). This is the
// BRANCHLESS Duff et al. / Frisvad-revised construction: one `copysign`, no
// normalization, no singular direction.
//
// PER-CALLER, tier-free (see geom/sphere.cuh's header note): serial, no
// `threadIdx` read, correct at any scope.

/**
 * @brief Right-handed orthonormal tangent basis from a unit vector (branchless).
 *
 * Given UNIT `n`, writes `t` and `b` with `{t, b, n}` orthonormal and
 * right-handed (`t × b = n`). Duff et al. (JCGT 2017) construction: exact
 * orthonormality to rounding for every input direction, no branch, no
 * normalization — continuity breaks only across the single `n_z` sign change
 * (unavoidable: no globally continuous tangent field exists on the sphere).
 * `n` must be normalized (‖n‖ = 1); the basis quality degrades smoothly with
 * ‖n‖ error. Scalar, tier-free.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Unit vector (3 elements).
 * @param t  Output tangent (3 elements).
 * @param b  Output bitangent (3 elements).
 */
template <typename T>
__device__ __forceinline__ void frame_from_vector(const T *n, T *t, T *b)
{
    const T s = copysign(static_cast<T>(1), n[2]);
    const T a = static_cast<T>(-1)/(s + n[2]);
    const T m = n[0]*n[1]*a;
    t[0] = static_cast<T>(1) + s*n[0]*n[0]*a;
    t[1] = s*m;
    t[2] = -s*n[0];
    b[0] = m;
    b[1] = s + n[1]*n[1]*a;
    b[2] = -n[1];
}
