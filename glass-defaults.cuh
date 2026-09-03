#pragma once
/**
 * @file glass-defaults.cuh
 * @brief Queryable execution plans for the measured placement ladder.
 *
 * The pick CANNOT be a device function: warp / block / nvidia need different
 * `<<<grid, block>>>` launches, so the decision happens host-side / at codegen time.
 * `recommend()` answers "given an operation, shape, scalar, dependency set,
 * and target SM, which measured implementation family/scope should the caller
 * use, and what legal baseline launch packs it?" It is metadata for a host
 * launcher or code generator, not
 * a device-function dispatcher.
 *
 *   constexpr auto p = glass::recommend<glass::op::potrf, float, N>(
 *       glass::dependency_set::mathdx);
 *   // p.implementation, p.execution_scope, p.block_threads,
 *   // p.problems_per_block, p.shared_bytes
 *
 * A thread recommendation means a thread-per-problem launch:
 * `<<<ceil(P / plan.problems_per_block), plan.block_threads>>>`. Vendor-backed
 * candidates are admitted only when the caller explicitly passes
 * `dependency_set::mathdx`. Each measured architecture also ships its measured
 * native runner-up table, so either policy is data-backed and neither depends
 * on header include order.
 *
 * Tables are **per-arch**: every swept SM gets its own constexpr ladder (`ideal_sm120`
 * today, measured on an RTX 5090), and `bench/tune.py --sm auto` adds or refreshes the
 * table + dispatch case for whatever GPU it runs on (e.g. `ideal_sm87` on a Jetson Orin)
 * without touching other arches' tables. Unmeasured SMs fall back to a coarse size
 * heuristic (`ideal_generic`). When NVIDIA block is returned, the call flows through
 * `glass::nvidia::block::<op>`, whose implementation may refine
 * SIMT-vs-cuBLASDx internally via `should_use_cublasdx`.
 */

#include <cstdint>

#include "glass-dispatch.cuh"  // shared `op` enum + GLASS_TARGET_SM + dispatch_body()

namespace glass {

enum class family : uint8_t { native, nvidia };
enum class scope : uint8_t { thread, warp, block };
enum class dependency_set : uint8_t { native_only, mathdx };

/// A measured placement plus a ready-to-use legal launch shape. The packing
/// fields are defaults, not a claim that every caller's best block size was
/// measured. `dynamic_requirement` means that the
/// selected explicit NVIDIA block wrapper's `*_threads()` or
/// `*_scratch_bytes()` query owns the exact value.
struct execution_plan {
    static constexpr uint32_t dynamic_requirement = UINT32_MAX;
    family implementation;
    scope execution_scope;
    uint32_t block_threads;
    uint32_t problems_per_block;
    uint32_t shared_bytes;
};

namespace defaults {

// Generated-table implementation detail. Ordinals stay stable so archived
// captures and local override fragments remain readable across this API cleanup.
enum class backend : int { warp, block, nvidia_block, thread, nvidia_thread };

// ─── measured ladders: one constexpr table per swept arch. bench/tune.py's ladder
// leg owns the marker blocks — it replaces the block for the arch it measured (and
// inserts a new block + dispatch case for a first-time arch), leaving the rest alone. ───

// === BEGIN tune.py ladder sm_120 ===
// Source sweep: mega_sweep_20260902_090755.txt   tie margin: ±5% (NVIDIA block/thread must clear it; SIMT ties ±2% prefer thread>warp>block)
// Fresh-input solver sweep: solver_ladder_20260902_103003.txt (1104 measured execution plans; symmetric selection)
// Interval confirmation: 1 ambiguous NVIDIA pick(s) demoted to the capture's native winner
// Paired tables preserve the measured native runner-up for callers that
// do not opt into MathDx; both use the same capture and SIMT tie rule.
constexpr backend ideal_sm120(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 12u ? backend::thread : backend::warp;
            else      return N <= 32u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : N <= 32u ? backend::warp : N <= 48u ? backend::block : backend::warp;
            else      return N <= 6u ? backend::thread : N <= 96u ? backend::warp : backend::block;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : N <= 24u ? backend::block : N <= 32u ? backend::nvidia_block : backend::block;
            else      return N <= 12u ? backend::warp : backend::block;
        case op::potrf:
            if (!f64) return N <= 4u ? backend::thread : N <= 8u ? backend::nvidia_thread : N <= 12u ? backend::warp : backend::nvidia_block;
            else      return N <= 8u ? backend::nvidia_thread : N <= 24u ? backend::thread : backend::block;
        case op::trsv:
            if (!f64) return N <= 16u ? backend::thread : N <= 48u ? backend::nvidia_block : backend::warp;
            else      return N <= 6u ? backend::thread : N <= 24u ? backend::nvidia_thread : N <= 32u ? backend::nvidia_block : backend::warp;
        case op::posv:
            if (!f64) return N <= 12u ? backend::thread : backend::nvidia_block;
            else      return N <= 8u ? backend::nvidia_thread : N <= 24u ? backend::thread : N <= 48u ? backend::warp : backend::block;
    }
    return backend::block;
}

constexpr backend native_sm120(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 12u ? backend::thread : backend::warp;
            else      return N <= 32u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : N <= 32u ? backend::warp : N <= 48u ? backend::block : backend::warp;
            else      return N <= 6u ? backend::thread : N <= 96u ? backend::warp : backend::block;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : backend::block;
            else      return N <= 12u ? backend::warp : backend::block;
        case op::potrf:
            if (!f64) return N <= 6u ? backend::thread : N <= 48u ? backend::warp : backend::block;
            else      return N <= 24u ? backend::thread : backend::block;
        case op::trsv: return N <= 16u ? backend::thread : backend::warp;
        case op::posv:
            if (!f64) return N <= 16u ? backend::thread : N <= 64u ? backend::warp : backend::block;
            else      return N <= 24u ? backend::thread : N <= 48u ? backend::warp : backend::block;
    }
    return backend::block;
}
// === END tune.py ladder sm_120 ===

// === BEGIN tune.py ladder sm_87 ===
// Source sweep: mega_sweep_20260901_163654.txt   tie margin: ±5% (NVIDIA block/thread must clear it; SIMT ties ±2% prefer thread>warp>block)
// Fresh-input solver sweep: solver_ladder_20260901_194600.txt (1104 measured execution plans; symmetric selection)
// Paired tables preserve the measured native runner-up for callers that
// do not opt into MathDx; both use the same capture and SIMT tie rule.
constexpr backend ideal_sm87(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 24u ? backend::thread : backend::warp;
            else      return N <= 64u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : N <= 32u ? backend::warp : N <= 48u ? backend::nvidia_block : backend::warp;
            else      return N <= 12u ? backend::thread : backend::warp;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : N <= 96u ? backend::nvidia_block : backend::block;
            else      return N <= 96u ? backend::warp : backend::block;
        case op::potrf:
            if (!f64) return N <= 4u ? backend::nvidia_thread : N <= 6u ? backend::thread : N <= 12u ? backend::nvidia_thread : backend::nvidia_block;
            else      return N <= 8u ? backend::nvidia_thread : N <= 48u ? backend::thread : N <= 96u ? backend::block : backend::warp;
        case op::trsv:
            if (!f64) return N <= 6u ? backend::thread : N <= 32u ? backend::nvidia_thread : backend::warp;
            else      return N <= 32u ? backend::nvidia_thread : N <= 48u ? backend::thread : N <= 96u ? backend::warp : backend::block;
        case op::posv:
            if (!f64) return N <= 16u ? backend::thread : backend::nvidia_block;
            else      return N <= 64u ? backend::thread : backend::block;
    }
    return backend::block;
}

constexpr backend native_sm87(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 24u ? backend::thread : backend::warp;
            else      return N <= 64u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : backend::warp;
            else      return N <= 12u ? backend::thread : backend::warp;
        case op::gemm:
            if (!f64) return N <= 32u ? backend::warp : backend::block;
            else      return N <= 96u ? backend::warp : backend::block;
        case op::potrf:
            if (!f64) return N <= 8u ? backend::thread : N <= 96u ? backend::warp : backend::block;
            else      return N <= 48u ? backend::thread : N <= 96u ? backend::block : backend::warp;
        case op::trsv:
            if (!f64) return N <= 16u ? backend::thread : backend::warp;
            else      return N <= 48u ? backend::thread : N <= 96u ? backend::warp : backend::block;
        case op::posv:
            if (!f64) return N <= 16u ? backend::thread : N <= 96u ? backend::warp : backend::block;
            else      return N <= 64u ? backend::thread : backend::block;
    }
    return backend::block;
}
// === END tune.py ladder sm_87 ===

// === BEGIN tune.py ladder sm_72 ===
// Source sweep: mega_sweep_20260901_163656.txt   tie margin: ±5% (NVIDIA block/thread must clear it; SIMT ties ±2% prefer thread>warp>block)
// Fresh-input solver sweep: solver_ladder_20260901_230339.txt (876 measured execution plans; symmetric selection)
// Paired tables preserve the measured native runner-up for callers that
// do not opt into MathDx; both use the same capture and SIMT tie rule.
constexpr backend ideal_sm72(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 16u ? backend::thread : backend::warp;
            else      return N <= 32u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : N <= 96u ? backend::warp : backend::block;
            else      return N <= 6u ? backend::thread : N <= 32u ? backend::warp : N <= 48u ? backend::block : N <= 64u ? backend::warp : backend::block;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : backend::block;
            else      return N <= 64u ? backend::warp : backend::block;
        case op::potrf:
            if (!f64) return N <= 6u ? backend::thread : N <= 48u ? backend::warp : N <= 96u ? backend::block : backend::warp;
            else      return N <= 32u ? backend::thread : backend::block;
        case op::trsv:
            if (!f64) return N <= 12u ? backend::thread : backend::warp;
            else      return N <= 32u ? backend::thread : N <= 96u ? backend::warp : backend::block;
        case op::posv:
            if (!f64) return N <= 16u ? backend::thread : N <= 48u ? backend::warp : N <= 64u ? backend::block : N <= 96u ? backend::warp : backend::block;
            else      return N <= 32u ? backend::thread : backend::block;
    }
    return backend::block;
}

constexpr backend native_sm72(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 16u ? backend::thread : backend::warp;
            else      return N <= 32u ? backend::thread : backend::warp;
        case op::gemv:
            if (!f64) return N <= 6u ? backend::thread : N <= 96u ? backend::warp : backend::block;
            else      return N <= 6u ? backend::thread : N <= 32u ? backend::warp : N <= 48u ? backend::block : N <= 64u ? backend::warp : backend::block;
        case op::gemm:
            if (!f64) return N <= 16u ? backend::warp : backend::block;
            else      return N <= 64u ? backend::warp : backend::block;
        case op::potrf:
            if (!f64) return N <= 6u ? backend::thread : N <= 48u ? backend::warp : N <= 96u ? backend::block : backend::warp;
            else      return N <= 32u ? backend::thread : backend::block;
        case op::trsv:
            if (!f64) return N <= 12u ? backend::thread : backend::warp;
            else      return N <= 32u ? backend::thread : N <= 96u ? backend::warp : backend::block;
        case op::posv:
            if (!f64) return N <= 16u ? backend::thread : N <= 48u ? backend::warp : N <= 64u ? backend::block : N <= 96u ? backend::warp : backend::block;
            else      return N <= 32u ? backend::thread : backend::block;
    }
    return backend::block;
}
// === END tune.py ladder sm_72 ===

// ─── blas2 family (syrk / syr2k / ldlt / ldlt_solve): warp-vs-block only;
// no vendor tier exists for these ops.
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
        case op::ldlt_solve:
            if (!f64) return N <= 64u ? backend::warp : backend::block;
            else      return backend::block;
    }
    return backend::block;
}
// === END tune.py blas2 sm_120 ===

constexpr bool is_blas2(op o) {
    return o == op::syrk || o == op::syr2k || o == op::ldlt || o == op::ldlt_solve;
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
constexpr backend rect_gemm_sm120(uint32_t M, uint32_t N, uint32_t K, bool f64) {
    if (!f64) {
        if (M == 6u && N == 64u && K == 6u) return backend::block;
        if (M == 8u && N == 8u && K == 32u) return backend::warp;
        if (M == 16u && N == 16u && K == 64u) return backend::block;
        if (M == 32u && N == 32u && K == 8u) return backend::warp;
        if (M == 64u && N == 6u && K == 6u) return backend::warp;
        if (M == 64u && N == 16u && K == 16u) return backend::warp;
    }
    if (f64) {
        if (M == 6u && N == 64u && K == 6u) return backend::block;
        if (M == 8u && N == 8u && K == 32u) return backend::block;
        if (M == 16u && N == 16u && K == 64u) return backend::block;
        if (M == 32u && N == 32u && K == 8u) return backend::block;
        if (M == 64u && N == 6u && K == 6u) return backend::warp;
        if (M == 64u && N == 16u && K == 16u) return backend::block;
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
constexpr backend rect_gemm_ideal(uint32_t M, uint32_t N, uint32_t K, bool f64, uint32_t sm) {
    switch (sm) {
        // === BEGIN tune.py rect gemm dispatch ===
        case 1200u: return rect_gemm_sm120(M, N, K, f64);
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
        case op::gemm: return N <= 8  ? backend::warp : N <= 64 ? backend::nvidia_block : backend::block;
        case op::potrf:
        case op::posv: return N <= 16 ? backend::warp : backend::nvidia_block;
        case op::trsv: return N <= 16 ? backend::warp : backend::block;
    }
    return backend::block;
}

// Conservative native fallback for an architecture without a measured table.
constexpr backend native_generic(op o, uint32_t N) {
    switch (o) {
        case op::dot:  return backend::warp;
        case op::gemv: return N <= 32 ? backend::warp : backend::block;
        case op::gemm: return N <= 8  ? backend::warp : backend::block;
        case op::potrf:
        case op::posv: return N <= 32 ? backend::warp : backend::block;       // crossover ~48
        case op::trsv: return backend::warp;                                  // warp wins w/o nvidia
    }
    return backend::block;
}

// Per-host override hook: a generated header (bench/autotune.py --emit-defaults) may
// `#define GLASS_DEFAULTS_HAVE_LOCAL` and provide
// `local_ideal(op,N,f64,sm,allow_nvidia)`. Point
// GLASS_DEFAULTS_TABLE_LOCAL at it to use your GPU's measured table instead of the seed.
#ifdef GLASS_DEFAULTS_TABLE_LOCAL
#include GLASS_DEFAULTS_TABLE_LOCAL
#endif

constexpr backend ideal(op o, uint32_t N, bool f64, uint32_t sm,
                        bool allow_nvidia = true) {
    if (is_blas2(o)) return blas2_ideal(o, N, f64, sm);
#ifdef GLASS_DEFAULTS_HAVE_LOCAL
    return local_ideal(o, N, f64, sm, allow_nvidia);
#else
    switch (sm) {
        // === BEGIN tune.py ladder dispatch ===
        case 720u: return allow_nvidia ? ideal_sm72(o, N, f64) : native_sm72(o, N, f64);
        case 870u: return allow_nvidia ? ideal_sm87(o, N, f64) : native_sm87(o, N, f64);
        case 1200u: return allow_nvidia ? ideal_sm120(o, N, f64) : native_sm120(o, N, f64);
        // === END tune.py ladder dispatch ===
        default:    return allow_nvidia ? ideal_generic(o, N, f64)
                                       : native_generic(o, N);
    }
#endif
}

constexpr uint32_t native_block_threads(op o, uint32_t N) {
    switch (o) {
        case op::potrf: case op::trsv: case op::posv: return 32u;
        case op::gemm: return N <= 8 ? 64u : N <= 16 ? 128u : 256u;
        case op::dot:  return 64u;
        case op::gemv: return N <= 16 ? 64u : 128u;
    }
    return 64u;
}

constexpr uint32_t native_warps_per_block(op o) {
    return o == op::dot ? 8u : 2u;
}

constexpr uint32_t native_threads_per_block(op o, uint32_t N) {
    switch (o) {
        case op::potrf: case op::posv: case op::trsv: case op::gemm:
            return N <= 4 ? 128u : N <= 6 ? 64u : 32u;
        case op::dot: case op::gemv:
            return 128u;
    }
    return 64u;
}

constexpr execution_plan make_plan(backend id, op o, uint32_t N) {
    switch (id) {
        case backend::thread: {
            const uint32_t t = native_threads_per_block(o, N);
            return {family::native, scope::thread, t, t, 0u};
        }
        case backend::warp: {
            const uint32_t w = native_warps_per_block(o);
            return {family::native, scope::warp, 32u * w, w, 0u};
        }
        case backend::nvidia_thread: {
            const uint32_t t = native_threads_per_block(o, N);
            return {family::nvidia, scope::thread, t, t, 0u};
        }
        case backend::nvidia_block:
            return {family::nvidia, scope::block,
                    execution_plan::dynamic_requirement, 1u,
                    execution_plan::dynamic_requirement};
        case backend::block:
        default:
            return {family::native, scope::block,
                    native_block_threads(o, N), 1u, 0u};
    }
}

}  // namespace defaults

/**
 * @brief Recommend a measured placement and a legal launch plan.
 *
 * Square ladder operations take one dimension. Rectangular GEMV takes M,N;
 * rectangular GEMM takes the conventional M,N,K order. `native_only` is the
 * safe default and never depends on header include order; pass `mathdx`
 * explicitly to admit NVIDIA block/thread candidates.
 */
template <op Op, typename T, uint32_t... Dims>
GLASS_DISPATCH_HD constexpr execution_plan recommend(
        dependency_set dependencies = dependency_set::native_only,
        uint32_t sm = GLASS_TARGET_SM) {
    static_assert(sizeof...(Dims) >= 1u, "recommend requires an operation shape");
    constexpr uint32_t d[] = {Dims...};
    constexpr bool is_gemm = Op == op::gemm;
    constexpr bool is_gemv = Op == op::gemv;
    static_assert((is_gemm && (sizeof...(Dims) == 1u || sizeof...(Dims) == 3u)) ||
                  (is_gemv && (sizeof...(Dims) == 1u || sizeof...(Dims) == 2u)) ||
                  ((!is_gemm && !is_gemv) && sizeof...(Dims) == 1u),
                  "recommend shape arity does not match the operation");

    defaults::backend id = defaults::backend::block;
    if constexpr (is_gemm && sizeof...(Dims) == 3u) {
        id = defaults::rect_gemm_ideal(d[0], d[1], d[2], sizeof(T) == 8u, sm);
    } else if constexpr (is_gemv && sizeof...(Dims) == 2u) {
        id = defaults::rect_gemv_ideal(d[0], d[1], sizeof(T) == 8u, sm);
    } else {
        id = defaults::ideal(Op, d[0], sizeof(T) == 8u, sm,
                             dependencies == dependency_set::mathdx);
    }
    return defaults::make_plan(id, Op, d[0]);
}

}  // namespace glass
