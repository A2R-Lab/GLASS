#pragma once
/**
 * @file glass-defaults.cuh
 * @brief Queryable backend-selection defaults — the measured warp/block/nvidia ladder
 *        (bench/RESULTS.md) exposed as `constexpr` so callers and GRiD-style
 *        codegen pick a backend + launch config instead of hand-copying a table.
 *
 * The pick CANNOT be a device function: warp / block / nvidia need different
 * `<<<grid, block>>>` launches, so the decision happens host-side / at codegen time.
 * These helpers answer "given (op, N, T) on this SM, which backend and how many threads?"
 *
 *   constexpr auto be = glass::suggested_backend<glass::op::chol, N, float>();
 *   if      constexpr (be == glass::backend::nvidia) { ... cuSOLVERDx block launch ... }
 *   else if constexpr (be == glass::backend::nvidia_thread) { ... cuSOLVERDx thread launch ... }
 *   else if constexpr (be == glass::backend::warp)   { ... <<<ceil(P/WPB), {32,WPB}>>> ... }
 *   else if constexpr (be == glass::backend::thread) { ... <<<ceil(P/TPB), TPB>>> ... }
 *   else                                             { ... <<<P, TB>>> ... }
 *
 * NOTE ON `thread`: measured and shipped for sm_120 (2026-07-18 sweep) — the tier
 * takes the low-DOF corner of every op except gemm (up to 7.5x on posv f64 at
 * N<=6; see the docs sweep-results page). `ideal_generic` and `without_nvidia`
 * still predate the tier (warp/block/nvidia only) — a thread verdict appears on an
 * arch once `bench/tune.py --sm auto` sweeps it there. A `backend::thread` pick
 * means a thread-per-problem launch: <<<ceil(P/TPB), TPB>>> with
 * suggested_threads_per_block<>().
 *
 * INCLUDE ORDER: include this AFTER glass.cuh, and after glass-nvidia.cuh if you want the
 * NVIDIA tiers to be eligible (it reads GLASS_HAVE_CUBLASDX,
 * GLASS_HAVE_CUSOLVERDX, and GLASS_HAVE_CUSOLVERDX_THREAD, which
 * glass-nvidia.cuh defines). With only glass.cuh, either dependency-backed tier
 * collapses to its warp/block runner-up, so a no-MathDx caller always gets a
 * backend it can actually launch.
 *
 * Tables are **per-arch**: every swept SM gets its own constexpr ladder (`ideal_sm120`
 * today, measured on an RTX 5090), and `bench/tune.py --sm auto` adds or refreshes the
 * table + dispatch case for whatever GPU it runs on (e.g. `ideal_sm87` on a Jetson Orin)
 * without touching other arches' tables. Unmeasured SMs fall back to a coarse size
 * heuristic (`ideal_generic`). When `nvidia` IS returned, the call still flows through
 * `glass::nvidia::<op>`, which refines SIMT-vs-cuBLASDx internally via should_use_cublasdx.
 */

#include <cstdint>

#include "glass-dispatch.cuh"  // shared `op` enum + GLASS_DEFAULTS_SM + dispatch_body()

namespace glass {

// (`op` lives in glass-dispatch.cuh — shared with the bare face's body table.)
// APPEND-ONLY: new values stay at the end so every shipped ordinal remains
// unchanged. Execution scopes are thread (one problem/thread), warp (one/warp),
// and block (one/block); NVIDIA provides measured implementations at block and,
// with cuSOLVERDx 0.4+, thread scope.
enum class backend : int { warp, block, nvidia, thread, nvidia_thread };

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
constexpr bool have_nv_thread =
#if defined(GLASS_HAVE_CUSOLVERDX_THREAD) && GLASS_HAVE_CUSOLVERDX_THREAD
    true;
#else
    false;
#endif

constexpr bool nv_available(op o) {
    return (o == op::gemm || o == op::gemv) ? have_nv_blas
         : (o == op::chol || o == op::trsv || o == op::posv) ? have_nv_lapack
         : false;  // dot: nvidia never wins
}

constexpr bool nv_thread_available(op o) {
    return have_nv_thread &&
           (o == op::chol || o == op::trsv || o == op::posv);
}

// ─── measured ladders: one constexpr table per swept arch. bench/tune.py's ladder
// leg owns the marker blocks — it replaces the block for the arch it measured (and
// inserts a new block + dispatch case for a first-time arch), leaving the rest alone. ───

// === BEGIN tune.py ladder sm_120 ===
// Source sweep: mega_sweep_20260830_042156.txt   tie margin: ±5% (NVIDIA block/thread must clear it; SIMT ties ±2% prefer thread>warp>block)
// Returns the *ideal* tier assuming NVIDIA dependencies are linked;
// availability predicates collapse either vendor tier after selection.
constexpr backend ideal_sm120(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 12u ? backend::thread : N <= 16u ? backend::warp : N <= 24u ? backend::thread : backend::warp;
            else      return N <= 32u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : N <= 32u ? backend::warp : N <= 48u ? backend::block : backend::warp;
            else      return N <= 6u ? backend::thread : backend::warp;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : N <= 24u ? backend::block : N <= 32u ? backend::nvidia : backend::block;
            else      return N <= 8u ? backend::warp : backend::block;
        case op::chol:
            if (!f64) return N <= 4u ? backend::thread : N <= 8u ? backend::nvidia_thread : N <= 24u ? backend::warp : backend::nvidia;
            else      return N <= 8u ? backend::nvidia_thread : N <= 24u ? backend::thread : backend::block;
        case op::trsv:
            if (!f64) return N <= 16u ? backend::thread : N <= 24u ? backend::nvidia_thread : N <= 32u ? backend::nvidia : backend::warp;
            else      return N <= 32u ? backend::nvidia_thread : N <= 48u ? backend::nvidia : backend::warp;
        case op::posv:
            if (!f64) return N <= 6u ? backend::thread : N <= 8u ? backend::nvidia_thread : N <= 12u ? backend::thread : backend::nvidia;
            else      return N <= 8u ? backend::nvidia_thread : N <= 24u ? backend::thread : N <= 32u ? backend::nvidia : backend::block;
    }
    return backend::block;
}
// === END tune.py ladder sm_120 ===

// === BEGIN tune.py ladder sm_87 ===
// Source sweep: mega_sweep_orin_tegra_20260830_035819.txt   tie margin: ±5% (NVIDIA block/thread must clear it; SIMT ties ±2% prefer thread>warp>block)
// Returns the *ideal* tier assuming NVIDIA dependencies are linked;
// availability predicates collapse either vendor tier after selection.
constexpr backend ideal_sm87(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 24u ? backend::thread : backend::warp;
            else      return N <= 64u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : N <= 32u ? backend::warp : N <= 48u ? backend::nvidia : backend::warp;
            else      return N <= 12u ? backend::thread : backend::warp;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : N <= 96u ? backend::nvidia : backend::block;
            else      return N <= 96u ? backend::warp : backend::block;
        case op::chol:
            if (!f64) return N <= 6u ? backend::thread : N <= 12u ? backend::nvidia_thread : backend::nvidia;
            else      return N <= 12u ? backend::nvidia_thread : N <= 48u ? backend::thread : N <= 64u ? backend::block : N <= 96u ? backend::warp : backend::block;
        case op::trsv:
            if (!f64) return N <= 12u ? backend::thread : N <= 32u ? backend::nvidia_thread : backend::warp;
            else      return N <= 32u ? backend::nvidia_thread : N <= 64u ? backend::thread : N <= 96u ? backend::warp : backend::block;
        case op::posv:
            if (!f64) return N <= 16u ? backend::thread : backend::nvidia;
            else      return N <= 8u ? backend::nvidia_thread : N <= 64u ? backend::thread : N <= 96u ? backend::warp : backend::block;
    }
    return backend::block;
}
// === END tune.py ladder sm_87 ===

// ─── blas2 family (syrk / syr2k / ldlt / ldltsv): warp-vs-block only — no
// vendor tier exists for these ops (nv_available() is already false for them).
// tune.py's blas2 leg owns the marker blocks; unmeasured arches stay block
// (the always-correct incumbent). inv/trmv/ger are single-impl (block-only)
// and deliberately have no table — measured and reported, never picked. ───

// === BEGIN tune.py blas2 sm_120 ===
// Source sweep: blas2_sweep_20260814_011659.txt   tie margin: ±5% (SIMT ties ±2% prefer the simpler tier)
constexpr backend blas2_sm120(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::syrk:
            if (!f64) return N <= 12u ? backend::warp : backend::block;
            else      return N <= 8u ? backend::warp : backend::block;
        case op::syr2k:
            if (!f64) return N <= 8u ? backend::warp : backend::block;
            else      return N <= 6u ? backend::warp : backend::block;
        case op::ldlt:
            if (!f64) return N <= 64u ? backend::warp : backend::block;
            else      return backend::block;
        case op::ldltsv:
            if (!f64) return N <= 64u ? backend::warp : backend::block;
            else      return backend::block;
    }
    return backend::block;
}
// === END tune.py blas2 sm_120 ===

constexpr bool is_blas2(op o) {
    return o == op::syrk || o == op::syr2k || o == op::ldlt || o == op::ldltsv;
}
constexpr backend blas2_ideal(op o, uint32_t N, bool f64, uint32_t sm) {
    switch (sm) {
        // === BEGIN tune.py blas2 dispatch ===
        case 1200u: return blas2_sm120(o, N, f64);
        // === END tune.py blas2 dispatch ===
        default:    return backend::block;  // unmeasured arch: block incumbent
    }
}

// ─── rectangular gemv/gemm: measured per EXACT shape by tune.py's rect leg
// (warp-vs-block; the vendor leg is per-shape machinery that belongs to the
// `shapes` table). Unmeasured shapes stay block. ───

// === BEGIN tune.py rect sm_120 ===
// Source sweep: rect_sweep_20260814_031430.txt   tie margin: ±5% (SIMT ties ±2% prefer the simpler tier); exact shapes only
constexpr backend rect_gemv_sm120(uint32_t M, uint32_t N, bool f64) {
    if (!f64) {
        if (M == 8u && N == 64u) return backend::warp;
        if (M == 16u && N == 128u) return backend::warp;
        if (M == 32u && N == 256u) return backend::warp;
        if (M == 64u && N == 8u) return backend::warp;
        if (M == 128u && N == 16u) return backend::block;
        if (M == 256u && N == 32u) return backend::warp;
    }
    if (f64) {
        if (M == 8u && N == 64u) return backend::warp;
        if (M == 16u && N == 128u) return backend::warp;
        if (M == 32u && N == 256u) return backend::warp;
        if (M == 64u && N == 8u) return backend::warp;
        if (M == 128u && N == 16u) return backend::warp;
        if (M == 256u && N == 32u) return backend::warp;
    }
    return backend::block;
}
constexpr backend rect_gemm_sm120(uint32_t M, uint32_t K, uint32_t N, bool f64) {
    if (!f64) {
        if (M == 6u && K == 6u && N == 64u) return backend::block;
        if (M == 8u && K == 32u && N == 8u) return backend::warp;
        if (M == 16u && K == 64u && N == 16u) return backend::block;
        if (M == 32u && K == 8u && N == 32u) return backend::warp;
        if (M == 64u && K == 6u && N == 6u) return backend::warp;
        if (M == 64u && K == 16u && N == 16u) return backend::warp;
    }
    if (f64) {
        if (M == 6u && K == 6u && N == 64u) return backend::block;
        if (M == 8u && K == 32u && N == 8u) return backend::block;
        if (M == 16u && K == 64u && N == 16u) return backend::block;
        if (M == 32u && K == 8u && N == 32u) return backend::block;
        if (M == 64u && K == 6u && N == 6u) return backend::warp;
        if (M == 64u && K == 16u && N == 16u) return backend::block;
    }
    return backend::block;
}
// === END tune.py rect sm_120 ===

constexpr backend rect_gemv_ideal(uint32_t M, uint32_t N, bool f64, uint32_t sm) {
    switch (sm) {
        // === BEGIN tune.py rect dispatch ===
        case 1200u: return rect_gemv_sm120(M, N, f64);
        // === END tune.py rect dispatch ===
        default:    return backend::block;
    }
}
constexpr backend rect_gemm_ideal(uint32_t M, uint32_t K, uint32_t N, bool f64, uint32_t sm) {
    switch (sm) {
        // === BEGIN tune.py rect gemm dispatch ===
        case 1200u: return rect_gemm_sm120(M, K, N, f64);
        // === END tune.py rect gemm dispatch ===
        default:    return backend::block;
    }
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
    if (is_blas2(o)) return blas2_ideal(o, N, f64, sm);
    switch (sm) {
        // === BEGIN tune.py ladder dispatch ===
        case 870u: return ideal_sm87(o, N, f64);
        case 1200u: return ideal_sm120(o, N, f64);
        // === END tune.py ladder dispatch ===
        default:    return ideal_generic(o, N, f64);
    }
#endif
}

}  // namespace defaults

// ─── bare-namespace body dispatch ────────────────────────────────────────────
// Moved to glass-dispatch.cuh (included above): the `body` enum and the
// tune.py-regenerated `dispatch_body()` table live there so glass.cuh's bare
// face can consume them without this header's vendor-macro include-order
// sensitivity. Determinism-sensitive consumers pin `glass::block::` explicitly.

/// Suggested backend for (op, N, T) on `SM`. Dependency-backed picks are
/// returned only when the required vendor implementation is available.
template <op Op, uint32_t N, typename T, uint32_t SM = GLASS_DEFAULTS_SM>
constexpr backend suggested_backend() {
    constexpr bool f64 = sizeof(T) == 8;
    backend id = defaults::ideal(Op, N, f64, SM);
    if (id == backend::nvidia && !defaults::nv_available(Op))
        return defaults::without_nvidia(Op, N);
    if (id == backend::nvidia_thread && !defaults::nv_thread_available(Op))
        return defaults::without_nvidia(Op, N);
    return id;
}

/// Rectangular gemv: measured per exact (M, N) shape by the rect leg; unmeasured
/// shapes (and unmeasured arches) return `block`. Never returns `nvidia`.
template <uint32_t M, uint32_t N, typename T, uint32_t SM = GLASS_DEFAULTS_SM>
constexpr backend suggested_backend_rect_gemv() {
    return defaults::rect_gemv_ideal(M, N, sizeof(T) == 8, SM);
}

/// Rectangular gemm (C is MxN, contraction K): measured per exact (M, K, N) shape.
template <uint32_t M, uint32_t K, uint32_t N, typename T, uint32_t SM = GLASS_DEFAULTS_SM>
constexpr backend suggested_backend_rect_gemm() {
    return defaults::rect_gemm_ideal(M, K, N, sizeof(T) == 8, SM);
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
