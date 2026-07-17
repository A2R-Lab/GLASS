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
 *   else if constexpr (be == glass::backend::thread) { ... <<<ceil(P/TPB), TPB>>> ... }
 *   else                                             { ... <<<P, TB>>> ... }
 *
 * NOTE ON `thread`: no shipped table returns it yet. The tier exists and is swept by
 * bench_mega_sweep.cu, but the in-tree ladders (`ideal_sm120`, `ideal_generic`,
 * `without_nvidia`) predate it and were measured with only warp/block/nvidia
 * contending — inventing `thread` entries here would fabricate a verdict nobody
 * measured. Run `bench/tune.py --legs ladder --sm auto` to regenerate this arch's
 * table WITH the thread column; the marker-block machinery and `_ladder_expr` already
 * emit `backend::thread` unchanged. Until then callers get the measured warp/block/
 * nvidia answer, and a `thread` caller opts in explicitly.
 *
 * INCLUDE ORDER: include this AFTER glass.cuh, and after glass-nvidia.cuh if you want the
 * `nvidia` tier to be eligible (it reads GLASS_HAVE_CUBLASDX / GLASS_HAVE_CUSOLVERDX, which
 * glass-nvidia.cuh defines). With only glass.cuh, the nvidia tier collapses to its warp/block
 * runner-up, so a no-MathDx caller always gets a backend it can actually launch.
 *
 * Tables are **per-arch**: every swept SM gets its own constexpr ladder (`ideal_sm120`
 * today, measured on an RTX 5090), and `bench/tune.py --sm auto` adds or refreshes the
 * table + dispatch case for whatever GPU it runs on (e.g. `ideal_sm87` on a Jetson Orin)
 * without touching other arches' tables. Unmeasured SMs fall back to a coarse size
 * heuristic (`ideal_generic`). When `nvidia` IS returned, the call still flows through
 * `glass::nvidia::<op>`, which refines SIMT-vs-cuBLASDx internally via should_use_cublasdx.
 */

#include <cstdint>

namespace glass {

enum class op : int      { dot, gemv, gemm, chol, trsv, posv };
// APPEND-ONLY: `thread` is last so the pre-existing warp/block/nvidia ordinals are
// unchanged. Scope ladder (most→least problem packing): thread (1 problem/thread,
// 32 per warp) → warp (1/warp) → block (1/block) → nvidia (1/block, vendor).
enum class backend : int { warp, block, nvidia, thread };

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

// ─── measured ladders: one constexpr table per swept arch. bench/tune.py's ladder
// leg owns the marker blocks — it replaces the block for the arch it measured (and
// inserts a new block + dispatch case for a first-time arch), leaving the rest alone. ───

// === BEGIN tune.py ladder sm_120 ===
// Source sweep: mega_sweep_20260704_2300.txt   tie margin: ±5% (nvidia must clear it)
// Returns the *ideal* tier assuming nvidia is linked; nv_available() filters after.
constexpr backend ideal_sm120(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot: return backend::warp;
        case op::gemv:
            if (!f64) return N <= 32u ? backend::warp : N <= 48u ? backend::block : backend::warp;
            else      return N <= 24u ? backend::warp : N <= 32u ? backend::block : N <= 64u ? backend::warp : backend::block;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : N <= 24u ? backend::block : N <= 64u ? backend::nvidia : backend::block;
            else      return N <= 4u ? backend::warp : N <= 12u ? backend::block : N <= 16u ? backend::warp : backend::block;
        case op::chol:
            if (!f64) return N <= 12u ? backend::warp : backend::nvidia;
            else      return N <= 24u ? backend::block : N <= 64u ? backend::nvidia : backend::block;
        case op::trsv:
            if (!f64) return N <= 16u ? backend::warp : N <= 32u ? backend::nvidia : backend::warp;
            else      return N <= 48u ? backend::nvidia : backend::block;
        case op::posv:
            if (!f64) return N <= 12u ? backend::warp : backend::nvidia;
            else      return N <= 64u ? backend::nvidia : backend::block;
    }
    return backend::block;
}
// === END tune.py ladder sm_120 ===

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
    switch (sm) {
        // === BEGIN tune.py ladder dispatch ===
        case 1200u: return ideal_sm120(o, N, f64);
        // === END tune.py ladder dispatch ===
        default:    return ideal_generic(o, N, f64);
    }
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

/// Suggested threads-per-block for the `thread` backend: launch `<<<ceil(P/TPB), TPB>>>`,
/// one problem per THREAD. Shrinks as N grows — the inverse of
/// `suggested_block_threads`: there, extra threads split ONE problem and idle on the
/// serial pivot; here every thread owns a whole problem, so the binding constraint is
/// the per-thread register footprint (~N*N live for a factor/solve; measured ceiling N<=7
/// — see CLAUDE.md), and a smaller
/// block keeps occupancy up. Seed heuristic, NOT measured — `bench/tune.py`'s ladder
/// leg does not tune this knob (it sweeps TPB but only records the winning tier).
template <op Op, uint32_t N = 0, typename T = float, uint32_t SM = GLASS_DEFAULTS_SM>
constexpr uint32_t suggested_threads_per_block() {
    switch (Op) {
        case op::chol: case op::posv: case op::trsv: case op::gemm:
            return N <= 4 ? 128u : N <= 6 ? 64u : 32u;   // N*N registers per thread
        case op::dot: case op::gemv:
            return 128u;                                  // ~N live registers, pack hard
    }
    return 64u;
}

}  // namespace glass
