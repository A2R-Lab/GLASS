#pragma once
#include <cstdint>

// ─── warp vote wrappers (L1) ─────────────────────────────────────────────────
//
// `warp::any` / `warp::all` are thin, full-mask spellings of `__any_sync` /
// `__all_sync`, offered for symmetry with `warp::reduce` / `warp::argmin_pair`
// so warp-per-problem domain code (collision verdicts, convergence checks,
// early-exit flags) carries no raw vote intrinsics. No performance content:
// each is exactly one instruction under the full mask.

// ═══════════════════════════════════════════════════════════════════════
// warp:: — one warp per problem (32 lanes, __*_sync)
// ═══════════════════════════════════════════════════════════════════════

namespace warp {
    /**
     * @brief Warp-wide OR of a per-lane predicate: `true` on every lane iff ANY
     *        lane passed `true`.
     *
     * `__any_sync(0xffffffff, pred)`. Full 32-lane warp required and every lane
     * must reach the call (the mask names all lanes, so a lane that branched
     * away deadlocks/UB the vote) — reconverge first, then vote. Same contract
     * as the register `warp::reduce` family.
     *
     * @param pred  This lane's predicate.
     * @return The warp-wide OR, identical on every lane.
     */
    __device__ __forceinline__ bool any(bool pred)
    { return __any_sync(0xffffffffu, pred) != 0; }

    /**
     * @brief Warp-wide AND of a per-lane predicate: `true` on every lane iff ALL
     *        lanes passed `true`.
     *
     * `__all_sync(0xffffffff, pred)`. Full-mask contract as `warp::any`.
     *
     * @param pred  This lane's predicate.
     * @return The warp-wide AND, identical on every lane.
     */
    __device__ __forceinline__ bool all(bool pred)
    { return __all_sync(0xffffffffu, pred) != 0; }
}  // namespace warp
