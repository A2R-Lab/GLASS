#pragma once
#include <cstdint>

// Per-SM lookup tables for the auto-dispatch decisions in glass::nvidia.
// Tell the wrapper primary templates (gemm<>, gemv<>, gemm_batched_1d<>,
// row_strided_*<>) whether to dispatch to cuBLASDx (true) or fall through
// to the SIMT path (false) for a given shape on a given SM.
//
// EDITING POLICY
// --------------
// The hand-curated values below ship as defaults. To regenerate this file
// for your specific machine (recommended for production deployments — small
// shapes are quite SM-dependent), run:
//
//   python bench/autotune.py --sm AUTO --out src/nvidia/tuning_table.cuh
//
// The autotune script measures both backends for the (M,N,K) grid you
// specify, emits one explicit specialization per measured shape, and leaves
// each `cublasdx_wins*<>` primary template (the conservative heuristic) as
// the fallback for unmeasured shapes.
//
// GLASS_TUNING_TABLE_LOCAL — per-build override (round-2)
// -------------------------------------------------------
// If you want to override entries without editing this file (e.g. for a
// per-host build on shared CI infrastructure), compile with:
//   -DGLASS_TUNING_TABLE_LOCAL='"path/to/your_tuning.cuh"'
// The named header is `#include`d at the bottom of this namespace and may
// add specializations for shapes NOT already specialized in the in-tree
// table above. (C++ disallows re-specialization, so to override a shape
// the in-tree table already covers, you must edit this file directly —
// or remove the in-tree entry first.)
//
// THE HEURISTICS
// --------------
// For shapes with no per-SM measurement, each API has its own heuristic
// reflecting its arithmetic intensity:
//
//   gemm:        max(M,N,K) >= 16 AND min(M,N,K) >= 4
//   gemv:        max(M,N)   >= 32             (lower intensity → SIMT scales farther)
//   batched_1d:  BATCH >= 8 AND max(M,N,K) >= 8
//
// Float-only because cuBLASDx's small-shape kernels are tuned for FP32.

namespace _glass_tuning {

// ─── gemm ─────────────────────────────────────────────────────────────────
// Primary template — conservative shape heuristic for unmeasured (M,N,K,SM).
template <uint32_t M, uint32_t N, uint32_t K, uint32_t SM_VAL>
constexpr bool cublasdx_wins() {
    constexpr uint32_t mx = (M > N ? (M > K ? M : K) : (N > K ? N : K));
    constexpr uint32_t mn = (M < N ? (M < K ? M : K) : (N < K ? N : K));
    return mx >= 16 && mn >= 4;
}

// ─── gemv (M×N → M, no K) ─────────────────────────────────────────────────
// GEMV has lower arithmetic intensity per element than GEMM, so SIMT
// competes farther up the size range.
template <uint32_t M, uint32_t N, uint32_t SM_VAL>
constexpr bool cublasdx_wins_gemv() {
    return (M >= 32) || (N >= 32);
}

// ─── gemm_batched_1d (1D-launch SIMT batched + optional cuBLASDx) ────────
// BATCH multiplies effective compute density. RFC starting heuristic.
template <uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH, uint32_t SM_VAL>
constexpr bool cublasdx_wins_batched() {
    constexpr uint32_t mx = (M > N ? (M > K ? M : K) : (N > K ? N : K));
    return (BATCH >= 8) && (mx >= 8);
}

// ─── row_strided_* (delegate to base APIs) ────────────────────────────────
// Packing cost is roughly constant per call so the tipping shape matches
// the unstrided variant. Users may still specialize these directly if their
// strided cases skew differently.
template <uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS, uint32_t B_RS, uint32_t SM_VAL>
constexpr bool cublasdx_wins_row_strided_gemm() {
    return cublasdx_wins<M, N, K, SM_VAL>();
}

template <uint32_t M, uint32_t N, uint32_t ROW_STRIDE, uint32_t SM_VAL>
constexpr bool cublasdx_wins_row_strided_gemv() {
    return cublasdx_wins_gemv<M, N, SM_VAL>();
}


// ---------------------------------------------------------------------------
// sm_120 — measured by bench/autotune.py at 2026-05-14 00:50.
// ---------------------------------------------------------------------------
template <> constexpr bool cublasdx_wins< 3,  3,  3, 1200>() { return false; } // simt wins (0.211us vs cublasdx 0.467us, 121.0%)
template <> constexpr bool cublasdx_wins< 4,  4,  4, 1200>() { return false; } // simt wins (0.205us vs cublasdx 0.455us, 121.7%)
template <> constexpr bool cublasdx_wins< 5,  5,  5, 1200>() { return false; } // simt wins (0.204us vs cublasdx 0.482us, 136.9%)
template <> constexpr bool cublasdx_wins< 6,  6,  6, 1200>() { return false; } // simt wins (0.220us vs cublasdx 0.504us, 129.1%)
template <> constexpr bool cublasdx_wins< 7,  7,  7, 1200>() { return false; } // simt wins (0.222us vs cublasdx 0.531us, 139.0%)
template <> constexpr bool cublasdx_wins< 8,  8,  8, 1200>() { return false; } // simt wins (0.207us vs cublasdx 0.392us, 89.3%)
template <> constexpr bool cublasdx_wins<12, 12, 12, 1200>() { return false; } // simt wins (0.491us vs cublasdx 0.543us, 10.5%)
template <> constexpr bool cublasdx_wins<14, 14, 14, 1200>() { return false; } // tie within ±5% → SIMT default
template <> constexpr bool cublasdx_wins<16, 16, 16, 1200>() { return true; } // cublasdx wins (0.447us vs simt 0.677us, 33.9%)
template <> constexpr bool cublasdx_wins<24, 24, 24, 1200>() { return true; } // cublasdx wins (0.798us vs simt 1.933us, 58.7%)
template <> constexpr bool cublasdx_wins<32, 32, 32, 1200>() { return true; } // cublasdx wins (1.055us vs simt 3.870us, 72.7%)
template <> constexpr bool cublasdx_wins<48, 48, 48, 1200>() { return true; } // cublasdx wins (2.332us vs simt 12.409us, 81.2%)
template <> constexpr bool cublasdx_wins<64, 64, 64, 1200>() { return true; } // cublasdx wins (5.319us vs simt 33.115us, 83.9%)

// ---------------------------------------------------------------------------
// Per-build local override (round-2). Define GLASS_TUNING_TABLE_LOCAL to a
// header path to append specializations for shapes not already covered
// above. Useful for ephemeral / per-host tuning without polluting source.
// See bench/TUNING.md.
// ---------------------------------------------------------------------------
#ifdef GLASS_TUNING_TABLE_LOCAL
#include GLASS_TUNING_TABLE_LOCAL
#endif

} // namespace _glass_tuning
