#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// Differential checkers that non-finite values cannot fool.
//
// The former per-driver helpers computed `m = std::max(m, fabs(a[i] - b[i]))`.
// A NaN difference compares false, so std::max kept the previous maximum and a
// NaN output read as zero error (equal infinities did the same: inf - inf is
// NaN). Found by injection on 2026-10-30. Any non-finite element on either
// side, or a size mismatch, now reports +inf and fails every tolerance; a
// driver whose op legitimately returns non-finite values must compare those
// elements explicitly rather than through these helpers.

template <typename T>
inline double checked_max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double x = static_cast<double>(a[i]), y = static_cast<double>(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) return INFINITY;
        m = std::max(m, std::fabs(x - y));
    }
    return m;
}

template <typename T>
inline double checked_max_rel_diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return INFINITY;
    double m = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double x = static_cast<double>(a[i]), y = static_cast<double>(b[i]);
        if (!std::isfinite(x) || !std::isfinite(y)) return INFINITY;
        m = std::max(m, std::fabs(x - y) / std::max(1.0, std::fabs(y)));
    }
    return m;
}

// Self-test exposed by each driver as the `checker_selftest` op: every injected
// corruption must report +inf, and a finite perturbation must be measured
// exactly. Prints one line per case and a final PASS/FAIL verdict.
inline int checker_selftest() {
    const float nan = NAN, inf = INFINITY;
    const std::vector<float> ref{1.f, 2.f, 3.f};
    struct Case { const char* name; std::vector<float> a, b; bool expect_inf; double expect_abs; };
    const Case cases[] = {
        {"nan_first",           {nan, 2.f, 3.f},  ref,              true,  0.0},
        {"nan_middle",          {1.f, nan, 3.f},  ref,              true,  0.0},
        {"nan_last",            {1.f, 2.f, nan},  ref,              true,  0.0},
        {"nan_reference",       ref,              {1.f, nan, 3.f},  true,  0.0},
        {"pos_inf",             {1.f, inf, 3.f},  ref,              true,  0.0},
        {"neg_inf",             {-inf, 2.f, 3.f}, ref,              true,  0.0},
        {"equal_inf",           {inf, 2.f, 3.f},  {inf, 2.f, 3.f},  true,  0.0},
        {"size_mismatch",       {1.f, 2.f},       ref,              true,  0.0},
        {"finite_perturbation", {1.f, 2.5f, 3.f}, ref,              false, 0.5},
        {"identical",           ref,              ref,              false, 0.0},
    };
    int fail = 0;
    for (const Case& c : cases) {
        const double da = checked_max_abs_diff(c.a, c.b);
        const double dr = checked_max_rel_diff(c.a, c.b);
        const bool ok = c.expect_inf ? (std::isinf(da) && std::isinf(dr))
                                     : (da == c.expect_abs && std::isfinite(dr));
        std::printf("checker %-20s abs=%g rel=%g %s\n", c.name, da, dr, ok ? "ok" : "WRONG");
        if (!ok) fail = 1;
    }
    std::printf("%s\n", fail ? "FAIL" : "PASS");
    return fail;
}
