#pragma once
#include <cstdint>

// ─── interval / hinge constraint scalars (robotics ops, projection family) ───
//
// The scalar constraint-handling kit for augmented-Lagrangian and barrier
// trajectory optimizers: interval violation, PHR augmented-Lagrangian value /
// gradient / Hessian for hinge and two-sided interval rows (with optional L1
// elastic softening), the C² relaxed log-barrier, and the smooth hinge used
// as a collision-cost activation. Promoted from GATO's AL/barrier scalars
// (numpy-oracle validated; cuRobo's independent AL kernel cross-validates the
// PHR form) plus cuRobo's η-activation.
//
// ALL ops here are SCALAR, TIER-FREE: no `threadIdx`, no shared state, return
// by value — the same function is correct at block, warp, or thread scope
// (each calling thread computes its own row). They live once at `glass::`.
//
// Conventions (kept from the validated GATO source):
//   * An interval row is `lo <= g <= hi`; an INFINITE bound contributes 0
//     (`isfinite` gate). `lo == hi` (finite) marks an EQUALITY row whose
//     signed multiplier lives in the `lam_hi` slot.
//   * PHR inequality side: `φ = (max(0, λ + ρc)² − λ²)/(2ρ)` with the signed
//     constraint `c` (`g − hi` above, `lo − g` below); equality:
//     `φ = λc + ρc²/2`. The `−λ²` offset keeps φ C¹ across activation.
//   * SOFT rows (`sigma > 0`): L1 elastic slack, analytically minimized — the
//     activation `a = λ + ρc` SATURATES at `sigma` (gradient caps at ±sigma,
//     Hessian 0 beyond the seam; C¹ at the seam).
//   * Relaxed barrier: one-sided barrier on the distance `d` to a bound —
//     `−μ·log(d)` for `d > δ`, the C² quadratic extension below (defined for
//     ALL d, infeasible-start safe; Hessian bounded by μ/δ²).

/**
 * @brief Interval violation: `max(0, g − hi) + max(0, lo − g)`.
 *
 * The true (unsigned) constraint violation of `lo <= g <= hi`; 0 when
 * feasible. Scalar, tier-free.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param g   Constraint value.
 * @param lo  Lower bound (may be −inf).
 * @param hi  Upper bound (may be +inf).
 * @return The violation (>= 0).
 */
template <typename T>
__device__ __forceinline__ T interval_violation(T g, T lo, T hi)
{
    T v = static_cast<T>(0);
    const T over = g - hi, under = lo - g;
    if (over  > static_cast<T>(0)) v += over;
    if (under > static_cast<T>(0)) v += under;
    return v;
}

/**
 * @brief Is this interval row an equality (`lo == hi`, finite)?
 *
 * Scalar, tier-free. Equality rows carry a signed multiplier in the `lam_hi`
 * slot of the interval AL ops.
 */
template <typename T>
__device__ __forceinline__ bool al_is_eq_row(T lo, T hi)
{
    return isfinite(lo) && lo == hi;
}

/**
 * @brief One hinge side's PHR augmented-Lagrangian value (optionally elastic).
 *
 * For the signed constraint `c` (feasible when `c <= 0`) with multiplier
 * `lam >= 0` and penalty `rho`: `φ = (max(0, λ + ρc)² − λ²)/(2ρ)`, so
 * `dφ/dc = max(0, λ + ρc)` and the GN Hessian is ρ on the active set. With
 * `sigma > 0` (L1 elastic slack weight) the activation saturates at `sigma`:
 * beyond it `φ = σc − (σ−λ)²/(2ρ)` (linear; C¹ at the seam). `sigma <= 0`
 * keeps the exact hard path. Scalar, tier-free.
 *
 * @tparam T  Scalar type.
 * @param c      Signed constraint value (`g − hi`, or `lo − g`).
 * @param lam    Multiplier (>= 0).
 * @param rho    Penalty parameter (> 0).
 * @param sigma  Elastic saturation weight (<= 0 disables softening).
 * @return The AL value contribution.
 */
template <typename T>
__device__ __forceinline__ T al_hinge_value(T c, T lam, T rho, T sigma)
{
    T a = lam + rho*c;
    if (a < static_cast<T>(0)) a = static_cast<T>(0);
    if (sigma > static_cast<T>(0) && a > sigma) {
        const T d = sigma - lam;
        return sigma*c - d*d/(static_cast<T>(2)*rho);
    }
    return (a*a - lam*lam)/(static_cast<T>(2)*rho);
}

/**
 * @brief PHR augmented-Lagrangian VALUE of an interval row `lo <= g <= hi`.
 *
 * Splits into independent hinge sides (`lam_hi` for `g <= hi`, `lam_lo` for
 * `g >= lo`); infinite bounds contribute 0; `lo == hi` rows are always-active
 * equalities (`φ = λc + ρc²/2`, signed multiplier in `lam_hi`, elastic
 * saturation symmetric at ±sigma). Scalar, tier-free. The matching outer
 * multiplier update is `λ ← max(0, λ + ρc)` (equalities unclamped, elastic
 * capped at sigma) on the ACCEPTED iterate.
 *
 * @tparam T  Scalar type.
 * @param g       Constraint value.
 * @param lo,hi   Interval bounds (either may be infinite).
 * @param lam_hi  Upper-side multiplier (equality rows: the signed multiplier).
 * @param lam_lo  Lower-side multiplier (unused on equality rows).
 * @param rho     Penalty parameter (> 0).
 * @param sigma   Elastic saturation weight (<= 0 disables softening).
 * @return The AL value contribution.
 */
template <typename T>
__device__ __forceinline__ T al_interval_value(T g, T lo, T hi, T lam_hi, T lam_lo,
                                               T rho, T sigma)
{
    if (al_is_eq_row(lo, hi)) {
        const T c = g - hi;
        const T a = lam_hi + rho*c;
        if (sigma > static_cast<T>(0) && a > sigma) {
            const T d = sigma - lam_hi;
            return sigma*c - d*d/(static_cast<T>(2)*rho);
        }
        if (sigma > static_cast<T>(0) && a < -sigma) {
            const T d = sigma + lam_hi;
            return -sigma*c - d*d/(static_cast<T>(2)*rho);
        }
        return lam_hi*c + static_cast<T>(0.5)*rho*c*c;
    }
    T v = static_cast<T>(0);
    if (isfinite(hi)) v += al_hinge_value(g - hi, lam_hi, rho, sigma);
    if (isfinite(lo)) v += al_hinge_value(lo - g, lam_lo, rho, sigma);
    return v;
}

/**
 * @brief PHR augmented-Lagrangian GRADIENT and GN HESSIAN of an interval row.
 *
 * `gr = dφ/dg`, `h = d²φ/dg²` (GN: ρ per active side, 0 in the elastic
 * saturation region). Same row semantics as `al_interval_value`. Scalar,
 * tier-free.
 *
 * @tparam T  Scalar type.
 * @param g,lo,hi,lam_hi,lam_lo,rho,sigma  See `al_interval_value`.
 * @param gr  Output gradient.
 * @param h   Output GN Hessian.
 */
template <typename T>
__device__ __forceinline__ void al_interval_grad_hess(T g, T lo, T hi, T lam_hi, T lam_lo,
                                                      T rho, T sigma, T &gr, T &h)
{
    gr = static_cast<T>(0);
    h  = static_cast<T>(0);
    const bool soft = sigma > static_cast<T>(0);
    if (al_is_eq_row(lo, hi)) {
        const T a = lam_hi + rho*(g - hi);
        if (soft && a >  sigma) { gr =  sigma; return; }
        if (soft && a < -sigma) { gr = -sigma; return; }
        gr = a;
        h  = rho;
        return;
    }
    if (isfinite(hi)) {
        const T a = lam_hi + rho*(g - hi);
        if (a > static_cast<T>(0)) {
            if (soft && a > sigma) { gr += sigma; }
            else                   { gr += a; h += rho; }
        }
    }
    if (isfinite(lo)) {
        const T a = lam_lo + rho*(lo - g);
        if (a > static_cast<T>(0)) {
            if (soft && a > sigma) { gr -= sigma; }
            else                   { gr -= a; h += rho; }
        }
    }
}

/**
 * @brief C² relaxed log-barrier VALUE on a bound distance `d`.
 *
 * `B(d) = −μ·log(d)` for `d > δ`; the C² quadratic extension
 * `−μ·(log δ − 3/2 + 2d/δ − d²/(2δ²))` for `d <= δ` — defined for ALL `d`
 * (including infeasible `d <= 0`), Hessian bounded by `μ/δ²`. Chain the sign
 * of `dd/dg` at the call site (+1 lower bound, −1 upper). Scalar, tier-free.
 *
 * @tparam T  Scalar type.
 * @param d      Distance to the bound (`g − lo` or `hi − g`).
 * @param mu     Barrier weight (> 0).
 * @param delta  Relaxation threshold (> 0).
 * @return The barrier value.
 */
template <typename T>
__device__ __forceinline__ T relaxed_barrier_value(T d, T mu, T delta)
{
    if (d > delta) return -mu*log(d);
    const T r = d/delta;
    return -mu*(log(delta) - static_cast<T>(1.5) + static_cast<T>(2)*r
                - static_cast<T>(0.5)*r*r);
}

/**
 * @brief Relaxed log-barrier derivative `dB/dd`. See `relaxed_barrier_value`.
 */
template <typename T>
__device__ __forceinline__ T relaxed_barrier_grad(T d, T mu, T delta)
{
    if (d > delta) return -mu/d;
    return -mu*(static_cast<T>(2) - d/delta)/delta;
}

/**
 * @brief Relaxed log-barrier second derivative `d²B/dd²` (sign-free in the
 *        bound direction). See `relaxed_barrier_value`.
 */
template <typename T>
__device__ __forceinline__ T relaxed_barrier_hess(T d, T mu, T delta)
{
    if (d > delta) return mu/(d*d);
    return mu/(delta*delta);
}

/**
 * @brief Two-sided relaxed-barrier VALUE on `g ∈ [lo, hi]` (infinite bound → 0).
 *
 * Scalar, tier-free. See `relaxed_barrier_value` for the one-sided form.
 */
template <typename T>
__device__ __forceinline__ T relaxed_barrier_interval_value(T g, T lo, T hi, T mu, T delta)
{
    T v = static_cast<T>(0);
    if (isfinite(lo)) v += relaxed_barrier_value(g - lo, mu, delta);
    if (isfinite(hi)) v += relaxed_barrier_value(hi - g, mu, delta);
    return v;
}

/**
 * @brief Two-sided relaxed-barrier GRADIENT `dB/dg`. See `relaxed_barrier_interval_value`.
 */
template <typename T>
__device__ __forceinline__ T relaxed_barrier_interval_grad(T g, T lo, T hi, T mu, T delta)
{
    T v = static_cast<T>(0);
    if (isfinite(lo)) v += relaxed_barrier_grad(g - lo, mu, delta);
    if (isfinite(hi)) v -= relaxed_barrier_grad(hi - g, mu, delta);
    return v;
}

/**
 * @brief Two-sided relaxed-barrier HESSIAN `d²B/dg²`. See `relaxed_barrier_interval_value`.
 */
template <typename T>
__device__ __forceinline__ T relaxed_barrier_interval_hess(T g, T lo, T hi, T mu, T delta)
{
    T v = static_cast<T>(0);
    if (isfinite(lo)) v += relaxed_barrier_hess(g - lo, mu, delta);
    if (isfinite(hi)) v += relaxed_barrier_hess(hi - g, mu, delta);
    return v;
}

/**
 * @brief Smooth hinge on a signed distance (collision-cost activation).
 *
 * The CHOMP/cuRobo η-metric: for signed distance `d` (positive = clear) and
 * activation width `eta > 0`,
 *   `d <= 0`      : `−d + η/2`        (linear in penetration)
 *   `0 < d < η`   : `(d−η)²/(2η)`     (quadratic taper)
 *   `d >= η`      : `0`               (inactive)
 * C¹ at both seams. Compose with `sphere_*_dist` for collision costs. Scalar,
 * tier-free.
 *
 * @tparam T  Scalar type.
 * @param d    Signed distance (positive = clear of contact).
 * @param eta  Activation width (> 0).
 * @return The activation cost (>= 0).
 */
template <typename T>
__device__ __forceinline__ T smooth_hinge(T d, T eta)
{
    if (d <= static_cast<T>(0)) return -d + static_cast<T>(0.5)*eta;
    if (d >= eta) return static_cast<T>(0);
    const T r = d - eta;
    return r*r/(static_cast<T>(2)*eta);
}

/**
 * @brief Smooth-hinge derivative `d(cost)/dd`. See `smooth_hinge`.
 */
template <typename T>
__device__ __forceinline__ T smooth_hinge_grad(T d, T eta)
{
    if (d <= static_cast<T>(0)) return static_cast<T>(-1);
    if (d >= eta) return static_cast<T>(0);
    return (d - eta)/eta;
}

/**
 * @brief Smooth absolute value: `log(cosh(x))` (overflow-safe).
 *
 * The classic C-infinity |x| surrogate of pose/tracking costs (cuRobo's
 * smooth pose-distance metric): quadratic (`x²/2`) near zero, asymptotically
 * `|x| − log 2`. Evaluated as `|x| + log1p(exp(−2|x|)) − log 2`, which never
 * overflows (naive `log(cosh(x))` dies at |x| ≈ 89 in f32). For a scaled
 * width use `log_cosh(x/s)·s`. Scalar, tier-free.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param x  Input value.
 * @return `log(cosh(x))` (>= 0).
 */
template <typename T>
__device__ __forceinline__ T log_cosh(T x)
{
    const T a = (x < static_cast<T>(0)) ? -x : x;
    return a + log1p(exp(static_cast<T>(-2)*a)) - static_cast<T>(0.6931471805599453);
}

/**
 * @brief `log_cosh` derivative: `tanh(x)`. See `log_cosh`.
 */
template <typename T>
__device__ __forceinline__ T log_cosh_grad(T x)
{
    return tanh(x);
}
