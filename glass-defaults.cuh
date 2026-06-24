#pragma once
/**
 * @file glass-defaults.cuh
 * @brief Queryable backend-selection defaults — the measured warp/block/nvidia ladder
 *        (bench/MEGA_SWEEP_RESULTS.md) exposed as `constexpr` so callers and GRiD-style
 *        codegen pick a backend + launch config instead of hand-copying a table.
 *
 * The pick CANNOT be a device function: warp / block / nvidia need different
 * `<<<grid, block>>>` launches, so the decision happens host-side / at codegen time.
 * These helpers answer "given (op, N, T) on this SM, which backend and how many threads?"
 *
 *   constexpr auto be = glass::suggested_backend<glass::op::chol, N, float>();
 *   if      constexpr (be == glass::backend::nvidia) { ... cuSOLVERDx launch ... }
 *   else if constexpr (be == glass::backend::warp)   { ... <<<ceil(P/WPB), {32,WPB}>>> ... }
 *   else                                             { ... <<<P, TB>>> ... }
 *
 * INCLUDE ORDER: include this AFTER glass.cuh, and after glass-nvidia.cuh if you want the
 * `nvidia` tier to be eligible (it reads GLASS_HAVE_CUBLASDX / GLASS_HAVE_CUSOLVERDX, which
 * glass-nvidia.cuh defines). With only glass.cuh, the nvidia tier collapses to its warp/block
 * runner-up, so a no-MathDx caller always gets a backend it can actually launch.
 *
 * Numbers are seeded from **RTX 5090 / sm_120** (the sweep). For SM != 1200 the helpers fall
 * back to a coarse size heuristic; regenerate a per-host table with bench/autotune.py (see
 * docs/.../concepts/tuning.rst). When `nvidia` IS returned, the call still flows through
 * `glass::nvidia::<op>`, which refines SIMT-vs-cuBLASDx internally via should_use_cublasdx.
 */

#include <cstdint>

namespace glass {

enum class op : int      { dot, gemv, gemm, chol, trsv, posv };
enum class backend : int { warp, block, nvidia };

// SM the table is keyed on: the build's SMS (nvidia builds) else the measured sm_120.
#ifndef GLASS_DEFAULTS_SM
  #ifdef SMS
    #define GLASS_DEFAULTS_SM (SMS)
  #else
    #define GLASS_DEFAULTS_SM (1200u)
  #endif
#endif

namespace defaults {

// Vendor availability per family (auto-detected from include order; absent => no nvidia tier).
constexpr bool have_nv_blas =
#if defined(GLASS_HAVE_CUBLASDX) && GLASS_HAVE_CUBLASDX
    true;
#else
    false;
#endif
constexpr bool have_nv_lapack =
#if defined(GLASS_HAVE_CUSOLVERDX) && GLASS_HAVE_CUSOLVERDX
    true;
#else
    false;
#endif

constexpr bool nv_available(op o) {
    return (o == op::gemm || o == op::gemv) ? have_nv_blas
         : (o == op::chol || o == op::trsv || o == op::posv) ? have_nv_lapack
         : false;  // dot: nvidia never wins
}

// ── sm_120 measured ladder (bench/MEGA_SWEEP_RESULTS.md, f32/f64 throughput) ──
// Returns the *ideal* tier assuming nvidia is linked; nv_available() filters after.
constexpr backend ideal_sm120(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:  return backend::warp;                                  // warp all N
        case op::gemv: return N <= 32 ? backend::warp : backend::block;       // nvidia never wins
        case op::gemm:
            if (!f64) return N <= 8  ? backend::warp
                           : N <= 12 ? backend::block
                           : N <= 64 ? backend::nvidia : backend::block;      // smem-cap >64
            else      return N <= 8  ? backend::warp
                           : N <= 32 ? backend::nvidia : backend::block;      // f64 band 16–32
        case op::chol:
            if (!f64) return N <= 12 ? backend::warp : backend::nvidia;       // nvidia to 128
            else      return N <= 16 ? backend::block
                           : N <= 64 ? backend::nvidia : backend::block;      // f64: block tiny, nv 32–64
        case op::trsv:
            return N <= 12 ? backend::warp
                 : N <= 32 ? backend::nvidia : backend::warp;                 // nv only mid-band
        case op::posv:
            if (!f64) return N <= 12 ? backend::warp : backend::nvidia;
            else      return N <= 64 ? backend::nvidia : backend::block;      // f64 nv to 64
    }
    return backend::block;
}

// Coarse fallback for unmeasured SMs: warp tiny, block large, nvidia mid for the
// parallel/factor ops when linked. Mirrors the sm_120 *shape*.
constexpr backend ideal_generic(op o, uint32_t N, bool /*f64*/) {
    switch (o) {
        case op::dot:  return backend::warp;
        case op::gemv: return N <= 32 ? backend::warp : backend::block;
        case op::gemm: return N <= 8  ? backend::warp : N <= 64 ? backend::nvidia : backend::block;
        case op::chol:
        case op::posv: return N <= 16 ? backend::warp : backend::nvidia;
        case op::trsv: return N <= 16 ? backend::warp : backend::block;
    }
    return backend::block;
}

// Runner-up when the ideal pick is nvidia but nvidia isn't linked (warp/block only).
constexpr backend without_nvidia(op o, uint32_t N) {
    switch (o) {
        case op::dot:  return backend::warp;
        case op::gemv: return N <= 32 ? backend::warp : backend::block;
        case op::gemm: return N <= 8  ? backend::warp : backend::block;
        case op::chol:
        case op::posv: return N <= 32 ? backend::warp : backend::block;       // crossover ~48
        case op::trsv: return backend::warp;                                  // warp wins w/o nvidia
    }
    return backend::block;
}

// Per-host override hook: a generated header (bench/autotune.py --emit-defaults) may
// `#define GLASS_DEFAULTS_HAVE_LOCAL` and provide `local_ideal(op,N,f64,sm)`. Point
// GLASS_DEFAULTS_TABLE_LOCAL at it to use your GPU's measured table instead of the seed.
#ifdef GLASS_DEFAULTS_TABLE_LOCAL
#include GLASS_DEFAULTS_TABLE_LOCAL
#endif

constexpr backend ideal(op o, uint32_t N, bool f64, uint32_t sm) {
#ifdef GLASS_DEFAULTS_HAVE_LOCAL
    return local_ideal(o, N, f64, sm);
#else
    return sm == 1200u ? ideal_sm120(o, N, f64) : ideal_generic(o, N, f64);
#endif
}

}  // namespace defaults

/// Suggested backend for (op, N, T) on `SM`. `nvidia` only when the vendor lib is linked.
template <op Op, uint32_t N, typename T, uint32_t SM = GLASS_DEFAULTS_SM>
constexpr backend suggested_backend() {
    constexpr bool f64 = sizeof(T) == 8;
    backend id = defaults::ideal(Op, N, f64, SM);
    if (id == backend::nvidia && !defaults::nv_available(Op))
        return defaults::without_nvidia(Op, N);
    return id;
}

/// Suggested block thread count for the `block` backend: factor/solve want 32 (extra
/// threads idle on the serial pivot); gemm grows with N; dot/gemv 64–128.
template <op Op, uint32_t N, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
constexpr uint32_t suggested_block_threads() {
    switch (Op) {
        case op::chol: case op::trsv: case op::posv: return 32u;
        case op::gemm: return N <= 8 ? 64u : N <= 16 ? 128u : 256u;
        case op::dot:  return 64u;
        case op::gemv: return N <= 16 ? 64u : 128u;
    }
    return 64u;
}

/// Suggested warps-per-block for the `warp` backend (intra-block problem packing).
template <op Op, uint32_t N = 0, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
constexpr uint32_t suggested_warps_per_block() {
    return Op == op::dot ? 8u : 2u;  // dot packs more (8–16); others 2–4
}

}  // namespace glass
