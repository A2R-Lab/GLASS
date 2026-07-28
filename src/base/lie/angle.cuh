#pragma once
#include <cstdint>

// ─── planar-angle (SO(2)) scalar utilities (robotics ops, Lie family) ────────
//
// Tiny angle helpers that every trajectory optimizer, sampling controller, and
// IK line search reimplements: wrap-to-(−π,π], shortest signed angular
// difference, wraparound-aware interpolation, and the acos-domain clamp.
//
// These are SCALAR, TIER-FREE ops: they read no `threadIdx`, touch no shared
// state, and return by value — the SAME function is correct at block, warp, or
// thread scope (each calling thread computes its own answer). They therefore
// live once at `glass::` scope with no `warp::`/`thread::` mirrors; there is
// nothing a tier could change.

namespace angle_detail {
    template <typename T> __device__ __forceinline__ T pi() {
        return static_cast<T>(3.14159265358979323846);
    }
}

/**
 * @brief Wrap an angle to the principal branch `(−π, π]`.
 *
 * Scalar, tier-free (see the header note). NumPy equivalent:
 * `np.arctan2(np.sin(x), np.cos(x))` — but computed by shifting, cheaper and
 * exact for inputs already in range.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param x  Angle in radians (any magnitude).
 * @return The equivalent angle in `(−π, π]`.
 */
template <typename T>
__host__ __device__ __forceinline__ T angle_wrap(T x)
{
    const T pi  = angle_detail::pi<T>();
    const T two = static_cast<T>(2) * pi;
    x = x - two * floor((x + pi) / two);
    // floor puts x in [−π, π); move the open end so −π maps to +π.
    if (x <= -pi) x += two;
    return x;
}

/**
 * @brief Shortest signed angular difference `a − b`, wrapped to `(−π, π]`.
 *
 * The SO(2) "boxminus": the smallest-magnitude rotation taking `b` to `a`
 * (positive = counterclockwise). Scalar, tier-free.
 *
 * @tparam T  Scalar type.
 * @param a  First angle (radians).
 * @param b  Second angle (radians).
 * @return `angle_wrap(a − b)`.
 */
template <typename T>
__host__ __device__ __forceinline__ T angle_diff(T a, T b)
{
    return angle_wrap(a - b);
}

/**
 * @brief Wraparound-aware linear interpolation between two angles.
 *
 * Interpolates along the SHORTEST arc: `angle_lerp(a, b, t) = a + t·(b ⊖ a)`
 * wrapped back to `(−π, π]` — so interpolating 179° → −179° passes through
 * 180°, not 0°. Scalar, tier-free.
 *
 * @tparam T  Scalar type.
 * @param a  Start angle (radians).
 * @param b  End angle (radians).
 * @param t  Interpolation parameter (0 → `a`, 1 → `b`).
 * @return The interpolated angle in `(−π, π]`.
 */
template <typename T>
__host__ __device__ __forceinline__ T angle_lerp(T a, T b, T t)
{
    return angle_wrap(a + t * angle_diff(b, a));
}

/**
 * @brief Clamp to `[−1, 1]` — the acos/asin domain guard.
 *
 * Dot products of nominally-unit vectors drift past ±1 in floating point;
 * feeding `acos` a value like `1 + 1e-7` returns NaN. Clamp first:
 * `acos(clamp_unit(dot))`. Scalar, tier-free.
 *
 * @tparam T  Scalar type.
 * @param v  Input value.
 * @return `v` clamped to `[−1, 1]`.
 */
template <typename T>
__host__ __device__ __forceinline__ T clamp_unit(T v)
{
    return v > static_cast<T>(1) ? static_cast<T>(1)
         : (v < static_cast<T>(-1) ? static_cast<T>(-1) : v);
}
