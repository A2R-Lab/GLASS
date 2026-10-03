#pragma once
#include <cstdint>

// ─── homogeneous point transform, lanes striding points (geometry family) ────
//
// `out_i = R·p_i + t` for n points under one 4x4 column-major homogeneous
// transform `X` (`X[r], X[r+4], X[r+8]` = rotation columns, `X[12..14]` =
// translation; row 3 ignored). The sphere-placement step of sphere-decomposed
// collision checking: a warp holds a link's world transform(s) and places that
// link's collision spheres before the narrow-phase kit in `geom/sphere.cuh`
// runs. Distilled from HJCD-IK's `warp_config_free`, and GRiD's warp
// `config_free` consumes it.
//
// PRECISION CONTRACT: points and outputs are `float` even when `X` is `double`
// (robotics callers hold double transforms and float sphere clouds). The
// per-component expression is EXACTLY
//     static_cast<float>(X[r]*p0 + X[r+4]*p1 + X[r+8]*p2 + X[r+12])
// evaluated in `T` — the same operand order as the reference it replaces, so
// a caller that previously inlined that line gets bit-identical results under
// the same `--fmad` setting. Do not reorder the terms.
//
// Warp tier only: the callers are warp-per-problem and n is small (tens to a
// few hundred spheres). No `__syncwarp` inside — the caller fences before a
// dependent read of `out` (every lane writes only its own points).

namespace warp {
    /**
     * @brief Single-warp homogeneous transform of `n` points: `out_i = R·p_i + t`.
     *
     * Lanes stride the points (lane `l` handles `i = l, l+32, …`); each lane
     * writes only its own `out[3i..3i+2]`. No cross-lane communication and NO
     * trailing `__syncwarp` — the caller fences before reading `out` from other
     * lanes. Full 32-lane warp required. Points/outputs are `float`; the
     * arithmetic runs in `T` and rounds once on store (see header note for the
     * exact expression). `out` must not alias `pts`.
     *
     * @tparam T  Transform scalar type (`float` or `double`).
     * @param X    Column-major 4x4 homogeneous transform (16 elements).
     * @param pts  Input points, packed `xyz` (`3*n` floats).
     * @param out  Output points, packed `xyz` (`3*n` floats).
     * @param n    Number of points.
     */
    template <typename T>
    __device__ void transform_points(const T *X, const float *pts, float *out, int n)
    {
        const int lane = static_cast<int>(flat_rank() & 31u);
        for (int i = lane; i < n; i += 32) {
            const float p0 = pts[3*i], p1 = pts[3*i + 1], p2 = pts[3*i + 2];
            out[3*i]     = static_cast<float>(X[0]*p0 + X[4]*p1 + X[8]*p2  + X[12]);
            out[3*i + 1] = static_cast<float>(X[1]*p0 + X[5]*p1 + X[9]*p2  + X[13]);
            out[3*i + 2] = static_cast<float>(X[2]*p0 + X[6]*p1 + X[10]*p2 + X[14]);
        }
    }

    /**
     * @brief Single-warp homogeneous transform of `n` points, each under its own
     *        transform: `out_i = R_{k_i}·p_i + t_{k_i}` with `k_i = Xidx[i]`.
     *
     * The multi-link form: `Xs` is an array of column-major 4x4 transforms
     * (16 elements apiece, e.g. one per movable joint) and `Xidx[i]` selects the
     * transform for point `i` — a sphere table's anchor column. Same striding,
     * precision, aliasing and no-trailing-sync contract as the single-`X`
     * overload; `Xidx[i]` must be a valid transform index (no base-link
     * sentinel handling — drop base points before the call).
     *
     * @tparam T  Transform scalar type (`float` or `double`).
     * @param Xs    Array of column-major 4x4 transforms (`16 * ntransforms`).
     * @param Xidx  Per-point transform index (`n` entries).
     * @param pts   Input points, packed `xyz` (`3*n` floats).
     * @param out   Output points, packed `xyz` (`3*n` floats).
     * @param n     Number of points.
     */
    template <typename T>
    __device__ void transform_points(const T *Xs, const int *Xidx, const float *pts,
                                     float *out, int n)
    {
        const int lane = static_cast<int>(flat_rank() & 31u);
        for (int i = lane; i < n; i += 32) {
            const T *X = Xs + 16*Xidx[i];
            const float p0 = pts[3*i], p1 = pts[3*i + 1], p2 = pts[3*i + 2];
            out[3*i]     = static_cast<float>(X[0]*p0 + X[4]*p1 + X[8]*p2  + X[12]);
            out[3*i + 1] = static_cast<float>(X[1]*p0 + X[5]*p1 + X[9]*p2  + X[13]);
            out[3*i + 2] = static_cast<float>(X[2]*p0 + X[6]*p1 + X[10]*p2 + X[14]);
        }
    }
}  // namespace warp
