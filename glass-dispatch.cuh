#pragma once
/**
 * @file glass-dispatch.cuh
 * @brief The bare-face body-dispatch table: `glass::dispatch_body()` picks the
 *        in-block implementation BODY (full-block SIMT / warp 0 / thread 0)
 *        for each (op, N, dtype) cell of the bare `glass::op` face.
 *
 * NAMESPACE CONTRACT (2026-07-30 restructure): explicit namespaces
 * (`glass::block::`, `glass::warp::`, `glass::thread::`,
 *  `glass::nvidia::{block,warp,thread}::`)
 * are the CONTRACT tier — bit-exact, never re-dispatched. The BARE `glass::op`
 * face keeps a block-scope CALLING contract (launch one block per problem,
 * any thread count) but its body is chosen from this measured table: a
 * `warp_in_block` / `thread_in_block` verdict means warp 0 / thread 0 of the
 * block executes the op's warp/thread twin, followed by a block-wide sync —
 * same contract, fewer participating lanes. `bench/tune.py --legs body`
 * regenerates the table (receipt-attested); the wrappers that consume it live
 * in `src/base/dispatch.cuh` and only shadow ops with at least one moved cell.
 *
 * This header is deliberately tiny and dependency-free: `glass.cuh` includes
 * it for the wrappers, and `glass-defaults.cuh` includes it for the shared
 * `op` enum — keeping it independent of the vendor-availability macros that
 * make glass-defaults.cuh include-order-sensitive.
 */

#include <cstdint>

// Device-callable constexpr without demanding --expt-relaxed-constexpr from
// every consumer; harmless no-op in pure host translation units.
#ifdef __CUDACC__
  #define GLASS_DISPATCH_HD __host__ __device__
#else
  #define GLASS_DISPATCH_HD
#endif

// One architecture selector for dispatch, advisors, and MathDx descriptors.
// Build systems should define GLASS_TARGET_SM (e.g. 1200 for sm_120). Legacy
// SMS remains an input alias during the source migration. An unspecified
// target uses the measured sm_120 seed table, matching prior GLASS releases.
#ifndef GLASS_TARGET_SM
  #ifdef SMS
    #define GLASS_TARGET_SM SMS
  #else
    #define GLASS_TARGET_SM 1200u
  #endif
#endif

// MathDx descriptor templates historically consume SMS. Define it once from
// the same selector so advisors, native dispatch, and vendor descriptors can
// no longer silently target different architectures.
#ifndef SMS
  #define SMS GLASS_TARGET_SM
#endif

namespace glass {

// APPEND-ONLY (ordinals are load-bearing for the per-arch ladder tables in
// glass-defaults.cuh): the six ladder ops first, then the body-sweep additions,
// then the blas2 family (warp-vs-block only; measured by tune.py's blas2 leg).
enum class op : int      { dot, gemv, gemm, potrf, trsv, posv, eig3, softmax,
                           syrk, syr2k, ldlt, ldlt_solve };

// The bare face's implementation bodies under the fixed block-scope contract.
// `warp_in_block` / `thread_in_block` = the op's warp/thread twin executed by
// warp 0 / thread 0 of the block, then a block-wide sync.
enum class body : int { block, warp_in_block, thread_in_block };

// === BEGIN tune.py body sm_120 ===
// Source sweep: body_dispatch_sweep_20260814_010815.txt   margin: ±5%
// RULE: never slower than block by >margin at ANY measured (NPROB, TB);
// faster by >margin at >=1 TB in the NPROB=8192 section. Else block.
// Verdicts are BOUNDED: N beyond the largest measured point stays block.
GLASS_DISPATCH_HD constexpr body body_sm120(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 4u ? body::block : N <= 16u ? body::thread_in_block : N <= 64u ? body::warp_in_block : body::block;
            else      return N <= 8u ? body::thread_in_block : N <= 64u ? body::warp_in_block : body::block;
        case op::gemv:
            if (!f64) return N <= 16u ? body::block : N <= 32u ? body::warp_in_block : body::block;
            else      return N <= 4u ? body::block : N <= 8u ? body::warp_in_block : N <= 16u ? body::block : N <= 32u ? body::warp_in_block : body::block;
        case op::gemm: return body::block;
        case op::potrf:
            if (!f64) return N <= 4u ? body::thread_in_block : body::block;
            else      return body::block;
        case op::trsv:
            if (!f64) return N <= 4u ? body::thread_in_block : N <= 32u ? body::warp_in_block : body::block;
            else      return N <= 8u ? body::block : N <= 16u ? body::warp_in_block : body::block;
        case op::posv:
            if (!f64) return N <= 4u ? body::thread_in_block : N <= 8u ? body::block : N <= 32u ? body::warp_in_block : body::block;
            else      return body::block;
        case op::eig3:
            if (!f64) return body::block;
            else      return N <= 3u ? body::thread_in_block : body::block;
        case op::softmax:
            if (!f64) return N <= 16u ? body::warp_in_block : body::block;
            else      return body::block;
        default: break;
    }
    return body::block;
}
// === END tune.py body sm_120 ===

// === BEGIN tune.py body sm_87 ===
// Source sweep: body_dispatch_sweep_20260803_0936.txt   margin: ±5%
// RULE: never slower than block by >margin at ANY measured (NPROB, TB);
// faster by >margin at >=1 TB in the NPROB=8192 section. Else block.
// Verdicts are BOUNDED: N beyond the largest measured point stays block.
GLASS_DISPATCH_HD constexpr body body_sm87(op o, uint32_t N, bool f64) {
    switch (o) {
        case op::dot:
            if (!f64) return N <= 16u ? body::thread_in_block : N <= 64u ? body::warp_in_block : body::block;
            else      return N <= 4u ? body::thread_in_block : N <= 16u ? body::block : N <= 64u ? body::warp_in_block : body::block;
        case op::gemv: return N <= 4u ? body::block : N <= 16u ? body::warp_in_block : body::block;
        case op::gemm:
            if (!f64) return N <= 4u ? body::warp_in_block : body::block;
            else      return body::block;
        case op::potrf: return body::block;
        case op::trsv:
            if (!f64) return N <= 4u ? body::thread_in_block : N <= 16u ? body::warp_in_block : body::block;
            else      return N <= 8u ? body::block : N <= 16u ? body::warp_in_block : body::block;
        case op::posv:
            if (!f64) return N <= 32u ? body::warp_in_block : body::block;
            else      return body::block;
        case op::eig3:
            if (!f64) return body::block;
            else      return N <= 3u ? body::thread_in_block : body::block;
        case op::softmax:
            if (!f64) return N <= 16u ? body::warp_in_block : body::block;
            else      return body::block;
        default: break;
    }
    return body::block;
}
// === END tune.py body sm_87 ===

// === BEGIN tune.py body dispatch ===
// Bodies for the bare block-scope face; unmeasured arches stay block.
GLASS_DISPATCH_HD constexpr body dispatch_body(op o, uint32_t N, bool f64,
                                               uint32_t sm = GLASS_TARGET_SM) {
    switch (sm) {
        case 870u: return body_sm87(o, N, f64);
        case 1200u: return body_sm120(o, N, f64);
        default: break;
    }
    return body::block;
}
// === END tune.py body dispatch ===

}  // namespace glass
