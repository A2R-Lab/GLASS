// test_defaults.cu — compile-time validation of glass-defaults.cuh. The helpers are
// constexpr, so the static_asserts ARE the test: if this compiles, the picks match the
// sweep (bench/RESULTS.md). No MathDx is linked: recommend() still checks both
// measured dependency policies because it returns metadata only.
#include <cstdio>
#include "glass.cuh"
#include "glass-defaults.cuh"

using glass::op;
namespace gd = glass::defaults;
using backend = gd::backend;

// Generated-table ordinals are serialized by archived tuner artifacts: append only.
static_assert(static_cast<int>(backend::warp) == 0, "backend ordinal: warp");
static_assert(static_cast<int>(backend::block) == 1, "backend ordinal: block");
static_assert(static_cast<int>(backend::nvidia_block) == 2, "backend ordinal: nvidia_block");
static_assert(static_cast<int>(backend::thread) == 3, "backend ordinal: thread");
static_assert(static_cast<int>(backend::nvidia_thread) == 4,
              "backend ordinal: nvidia_thread appended");

// ── measured sm_120 ladder (2026-08-30, independent of what's linked) ──
// The higher-repetition throughput replication agreed on 131/132 policy
// winners. The lone disagreement was dot f32 N=16 inside the SIMT tie band;
// these pins follow the 500-repetition capture used to generate the table.
static_assert(gd::ideal_sm120(op::gemm, 8,  false) == backend::warp,   "gemm8 f32");
static_assert(gd::ideal_sm120(op::gemm, 12, false) == backend::warp,   "gemm12 f32");
static_assert(gd::ideal_sm120(op::gemm, 24, false) == backend::block,  "gemm24 f32");
static_assert(gd::ideal_sm120(op::gemm, 32, false) == backend::nvidia_block, "gemm32 f32");
static_assert(gd::ideal_sm120(op::gemm, 96, false) == backend::block,  "gemm96 f32 (smem cap)");
// NVIDIA thread wins 14 throughput cells after the valid-input confirmation
// vetoes three repeated-mutation ladder picks.
static_assert(gd::ideal_sm120(op::potrf, 4,   false) == backend::thread, "potrf4 f32");
static_assert(gd::ideal_sm120(op::potrf, 8,   false) == backend::nvidia_thread, "potrf8 f32 -> NVIDIA thread");
static_assert(gd::ideal_sm120(op::potrf, 24,  false) == backend::warp,   "potrf24 f32");
static_assert(gd::ideal_sm120(op::potrf, 128, false) == backend::nvidia_block, "potrf128 f32");
static_assert(gd::ideal_sm120(op::trsv, 12, false) == backend::thread, "trsv12 f32");
static_assert(gd::ideal_sm120(op::trsv, 24, false) == backend::nvidia_thread, "trsv24 f32 -> NVIDIA thread");
static_assert(gd::ideal_sm120(op::trsv, 32, false) == backend::nvidia_block, "trsv32 f32 -> NVIDIA block");
static_assert(gd::ideal_sm120(op::trsv, 64, false) == backend::warp,   "trsv64 f32");
static_assert(gd::ideal_sm120(op::trsv, 6, true) == backend::thread,
              "trsv6 f64 -> native after valid-input veto");
static_assert(gd::ideal_sm120(op::dot,  8,   false) == backend::thread, "dot8");
static_assert(gd::ideal_sm120(op::dot,  16,  false) == backend::warp,   "dot16 higher-repetition tie verdict");
static_assert(gd::ideal_sm120(op::dot,  24,  false) == backend::thread, "dot24");
static_assert(gd::ideal_sm120(op::dot,  128, false) == backend::warp,  "dot128");
static_assert(gd::ideal_sm120(op::gemv, 4,   false) == backend::thread, "gemv4");
static_assert(gd::ideal_sm120(op::gemv, 32,  false) == backend::warp,  "gemv32");
static_assert(gd::ideal_sm120(op::gemv, 48,  false) == backend::block, "gemv48");
static_assert(gd::ideal_sm120(op::potrf, 8,  true) == backend::nvidia_thread, "potrf8 f64 -> NVIDIA thread");
static_assert(gd::ideal_sm120(op::potrf, 48, true) == backend::block,  "potrf48 f64");
static_assert(gd::ideal_sm120(op::gemm, 64, true) == backend::block,  "gemm64 f64");
static_assert(gd::ideal_sm120(op::posv, 8,  true) == backend::nvidia_thread, "posv8 f64 -> NVIDIA thread");
static_assert(gd::ideal_sm120(op::posv, 8,  false) == backend::thread,
              "posv8 f32 -> native after valid-input veto");
static_assert(gd::ideal_sm120(op::posv, 12, true) == backend::thread, "posv12 f64");
static_assert(gd::ideal_sm120(op::posv, 64, true) == backend::block,  "posv64 f64");

// ── sm_87 (Jetson AGX Orin, pinned 50W mode; 2026-08-30) ──
// The vendor thread path is linked through MathDx's architecture-neutral
// LTO-IR fatbin. The valid-input confirmation retains 15 of the original 19
// repeated-mutation ladder winners.
static_assert(gd::ideal_sm87(op::dot,   8,  false) == backend::thread, "dot8 f32");
static_assert(gd::ideal_sm87(op::dot,   128, false) == backend::warp,  "dot128 f32");
static_assert(gd::ideal_sm87(op::potrf,  8,  false) == backend::nvidia_thread, "potrf8 f32 -> NVIDIA thread");
static_assert(gd::ideal_sm87(op::potrf,  12, false) == backend::nvidia_thread, "potrf12 f32 -> NVIDIA thread");
static_assert(gd::ideal_sm87(op::potrf,  48, false) == backend::nvidia_block, "potrf48 f32 -> vendor");
static_assert(gd::ideal_sm87(op::trsv,  16, false) == backend::nvidia_thread, "trsv16 f32 -> NVIDIA thread");
static_assert(gd::ideal_sm87(op::posv,  32, false) == backend::nvidia_block, "posv32 f32 -> vendor");
static_assert(gd::ideal_sm87(op::posv,  16, false) == backend::thread, "posv16 f32");
static_assert(gd::ideal_sm87(op::gemm,  64, false) == backend::nvidia_block, "gemm64 f32 -> vendor");
static_assert(gd::ideal_sm87(op::gemm,  64, true)  == backend::warp,   "gemm64 f64 (SIMT tie: warp within 1% of block to N=96)");
static_assert(gd::ideal_sm87(op::gemm, 128, true)  == backend::block,  "gemm128 f64 (block's only real win, 24% faster)");
static_assert(gd::ideal_sm87(op::potrf,  8,  true)   == backend::nvidia_thread, "potrf8 f64 -> NVIDIA thread");
static_assert(gd::ideal_sm87(op::potrf,  12, true)  == backend::thread,
              "potrf12 f64 -> native after valid-input veto");
static_assert(gd::ideal_sm87(op::potrf,  24, true)  == backend::thread, "potrf24 f64 (thread reaches further than sm_120)");
static_assert(gd::ideal_sm87(op::posv,  8,  true)   == backend::thread,
              "posv8 f64 -> native after valid-input veto");

// ── per-arch dispatch: a measured SM hits its table, an unmeasured SM falls to generic ──
static_assert(gd::ideal(op::gemm, 32, false, 1200u) == gd::ideal_sm120(op::gemm, 32, false), "sm_120 dispatches to its table");
static_assert(gd::ideal(op::posv, 64, true,  1200u) == gd::ideal_sm120(op::posv, 64, true),  "sm_120 dispatches to its table (f64)");
static_assert(gd::ideal(op::potrf, 48, false, 870u)  == gd::ideal_sm87(op::potrf, 48, false),  "sm_87 dispatches to its table");
static_assert(gd::ideal(op::posv, 64, true,  870u)  == gd::ideal_sm87(op::posv, 64, true),   "sm_87 dispatches to its table (f64)");
// (sentinel SMs no sweep will ever produce — a real new arch, e.g. sm_87 on Jetson,
//  gets its own table + dispatch case from tune.py and must NOT be asserted generic here)
static_assert(gd::ideal(op::gemm, 32, false, 0u) == gd::ideal_generic(op::gemm, 32, false), "unmeasured SM falls to generic");
static_assert(gd::ideal(op::potrf, 24, false, 1u) == gd::ideal_generic(op::potrf, 24, false), "unmeasured SM falls to generic");

// ── blas2 family (warp-vs-block; tune.py blas2 leg, blas2_sweep_20260718_0327) ──
static_assert(gd::blas2_sm120(op::syrk,  16, false) == backend::block, "syrk16 f32 -> block (2.2% gap just outside SIMT tie band, 2026-08-12 capture; band-edge cell)");
static_assert(gd::blas2_sm120(op::syrk,  24, false) == backend::block, "syrk24 f32 -> block");
static_assert(gd::blas2_sm120(op::ldlt,  32, false) == backend::warp,  "ldlt32 f32 -> warp");
static_assert(gd::blas2_sm120(op::ldlt,  32, true)  == backend::block, "ldlt32 f64 -> block");
static_assert(gd::blas2_sm120(op::syr2k,  8, true)  == backend::block, "syr2k8 f64 -> block (2.5% gap just outside SIMT tie band, 2026-08-12 capture; band-edge cell)");
static_assert(gd::ideal(op::syrk, 16, false, 1200u) == gd::blas2_sm120(op::syrk, 16, false), "blas2 ops route through ideal()");
static_assert(gd::ideal(op::ldlt_solve, 32, false, 870u) == backend::block, "blas2 unmeasured arch -> block incumbent");
static_assert(glass::recommend<op::ldlt, float, 32>(glass::dependency_set::native_only, 1200u).execution_scope == glass::scope::warp,
              "public plan reaches blas2 table");

// ── rect exact-shape pickers (tune.py rect leg, rect_sweep_20260718_0328) ──
static_assert(gd::rect_gemv_sm120( 64,  8, false) == backend::warp,  "gemv 64x8 f32 tall -> warp");
static_assert(gd::rect_gemv_sm120(128, 16, false) == backend::block, "gemv 128x16 f32 -> block");
static_assert(gd::rect_gemv_sm120(128, 16, true)  == backend::warp,  "gemv 128x16 f64 -> warp");
static_assert(gd::rect_gemm_sm120( 6, 64,  6, false) == backend::block, "gemm M=6,N=64,K=6 wide -> block");
static_assert(gd::rect_gemm_sm120(32, 32,  8, false) == backend::warp,  "gemm M=32,N=32,K=8 -> warp");
static_assert(glass::recommend<op::gemv, float, 64, 8>(glass::dependency_set::native_only, 1200u).execution_scope == glass::scope::warp,
              "public rectangular GEMV plan");
static_assert(glass::recommend<op::gemm, float, 7, 7, 7>(glass::dependency_set::native_only, 1200u).execution_scope == glass::scope::block,
              "unmeasured rectangular GEMM shape -> block");
static_assert(glass::recommend<op::gemv, float, 64, 8>(glass::dependency_set::native_only, 870u).execution_scope == glass::scope::block,
              "rectangular unmeasured arch -> block");

// ── bare-namespace face: measured body pins (2026-08-14 sweep, sm_120) ──
// dispatch_body() now carries the measured body_sm120 table
// (body_dispatch_sweep_20260814_010815.txt; rule = never worse than block
// by >5% at any measured (NPROB, TB), >5% faster at >=1 TB @ NPROB=8192,
// bounded at the largest measured N). Spot-pin the moved cells + the rule's
// conservative refusals:
using glass::body;
static_assert(glass::dispatch_body(op::dot,   4, false, 1200u) == body::block,           "dot4 f32 -> block body");
static_assert(glass::dispatch_body(op::dot,   8, false, 1200u) == body::thread_in_block, "dot8 f32 -> thread body");
static_assert(glass::dispatch_body(op::dot,  16, false, 1200u) == body::thread_in_block, "dot16 f32 -> thread body");
static_assert(glass::dispatch_body(op::dot,  32, false, 1200u) == body::warp_in_block,   "dot32 f32 -> warp body");
static_assert(glass::dispatch_body(op::dot,  64, false, 1200u) == body::warp_in_block,   "dot64 f32 -> warp body");
static_assert(glass::dispatch_body(op::dot, 128, false, 1200u) == body::block,           "dot128 f32 BOUNDED -> block");
static_assert(glass::dispatch_body(op::trsv, 16, false, 1200u) == body::warp_in_block,   "trsv16 f32 -> warp body");
static_assert(glass::dispatch_body(op::trsv, 64, false, 1200u) == body::block,           "trsv64 f32 stays block");
static_assert(glass::dispatch_body(op::posv,  4, false, 1200u) == body::thread_in_block, "posv4 f32 -> thread body");
static_assert(glass::dispatch_body(op::posv, 16, true, 1200u)  == body::block,           "posv f64 all block");
static_assert(glass::dispatch_body(op::eig3,  3, true, 1200u)  == body::thread_in_block, "eig3 f64 -> thread body");
static_assert(glass::dispatch_body(op::eig3,  3, false, 1200u) == body::block,           "eig3 f32 TB-unstable -> block");
static_assert(glass::dispatch_body(op::softmax, 16, false, 1200u) == body::warp_in_block, "softmax16 f32 -> warp body");
static_assert(glass::dispatch_body(op::softmax, 4096, false, 1200u) == body::block,      "softmax large-n BOUNDED -> block");
static_assert(glass::dispatch_body(op::gemm, 16, false, 1200u) == body::block,           "gemm never moves");
// sm_87 body table (body_dispatch_sweep_20260803_0936.txt, 50W, archived externally): the same rule
// moves 23 cells there; softmax/eig3 land identically, dot's warp band runs wider.
static_assert(glass::dispatch_body(op::dot,   8, false, 870u) == body::thread_in_block, "sm_87 dot8 -> thread body");
static_assert(glass::dispatch_body(op::dot,  32, false, 870u) == body::warp_in_block,   "sm_87 dot32 -> warp body");
static_assert(glass::dispatch_body(op::dot, 128, false, 870u) == body::block,           "sm_87 dot128 BOUNDED -> block");
static_assert(glass::dispatch_body(op::potrf, 16, false, 870u) == body::block,           "sm_87 potrf never moves");
static_assert(glass::dispatch_body(op::eig3,  3, true,  870u) == body::thread_in_block, "sm_87 eig3 f64 -> thread body");
static_assert(glass::dispatch_body(op::softmax, 16, false, 870u) == body::warp_in_block, "sm_87 softmax16 -> warp body");
// unmeasured arch: every cell stays the block body
static_assert(glass::dispatch_body(op::dot, 32, false, 0u) == body::block, "unmeasured SM -> block body");
// bare == block identity is pinned through an op with NO moved cells (the
// moved names are now distinct wrapper entities; unmoved names must remain
// the very same block:: functions via the using-directive).
static_assert(&glass::symmetrize<float, 8, true> == &glass::block::symmetrize<float, 8, true>,
              "bare (unmoved) IS block");
// tier aliases: glass::warp/thread are namespace aliases of block::warp/thread
static_assert(&glass::warp::dot<float, 8>  == &glass::block::warp::dot<float, 8>,  "warp alias");
static_assert(&glass::thread::dot<float, 8> == &glass::block::thread::dot<float, 8>, "thread alias");

// ── public dependency policies ──
constexpr auto native_potrf = glass::recommend<op::potrf, float, 8>();
static_assert(native_potrf.implementation == glass::family::native &&
              native_potrf.execution_scope == glass::scope::warp &&
              native_potrf.block_threads == 64u && native_potrf.problems_per_block == 2u,
              "native-only plan uses the measured native runner-up");
constexpr auto mathdx_potrf = glass::recommend<op::potrf, float, 8>(
    glass::dependency_set::mathdx, 1200u);
static_assert(mathdx_potrf.implementation == glass::family::nvidia &&
              mathdx_potrf.execution_scope == glass::scope::thread,
              "MathDx plan admits the measured NVIDIA-thread winner");
constexpr auto native_posv = glass::recommend<op::posv, float, 8>();
static_assert(native_posv.implementation == glass::family::native &&
              native_posv.execution_scope == glass::scope::thread,
              "native-only plan preserves the measured native thread winner");
constexpr auto native_gemm = glass::recommend<op::gemm, float, 32>();
static_assert(native_gemm.implementation == glass::family::native &&
              native_gemm.execution_scope == glass::scope::block &&
              native_gemm.block_threads == 256u && native_gemm.problems_per_block == 1u,
              "native-only plan includes the block launch");
constexpr auto native_dot = glass::recommend<op::dot, float, 24>(glass::dependency_set::native_only, 1200u);
static_assert(native_dot.execution_scope == glass::scope::thread &&
              native_dot.block_threads == native_dot.problems_per_block,
              "thread plan packs one problem per thread");

// ── host-side query/size helpers: constexpr, so the asserts ARE the test.
// Property-based (positive, monotone in threads/size) rather than exact —
// the formulas are the implementation's business, launchability is ours. ──
static_assert(glass::dot_fast_scratch_bytes<float>(64) == 8, "2 warps -> 2 floats");
static_assert(glass::dot_fast_scratch_bytes<double>(256) >= glass::dot_fast_scratch_bytes<double>(32), "monotone in threads");
static_assert(glass::iamax_scratch_bytes<float>(64) > 0 && glass::iamax_fast_scratch_bytes<float>(64) > 0, "iamax scratch positive");
static_assert(glass::argreduce_scratch_bytes<float>(64) > 0 && glass::argreduce_fast_scratch_bytes<float>(64) > 0, "argreduce scratch positive");
static_assert(glass::congruence_scratch_bytes<float, 12, 4>() > 0, "congruence scratch positive");
static_assert(glass::gn_step_scratch_bytes<float, 7>() > 0, "gn_step scratch positive");
static_assert(glass::inv_dense_scratch_bytes<float>(8) >= glass::inv_dense_scratch_bytes<float>(4), "inv_dense monotone in dim");
static_assert(glass::trmv_scratch_bytes<double>(6) == 6 * sizeof(double), "trmv scratch = n*T");
// scratch-formula pins (audit 2026-08-11: covered in test/cuda so the coverage
// badge never depends on the examples that also call them)
static_assert(glass::ldlt_scratch_bytes<float>(8) == 9 * sizeof(float), "ldlt scratch = (n+1)*T");
static_assert(glass::inv_scratch_bytes<double>(8) == 17 * sizeof(double), "inv scratch = (2n+1)*T");
static_assert(glass::inv_pivoted_scratch_bytes<float>(8) == 25 * sizeof(float), "inv_pivoted scratch = (3n+1)*T");
static_assert(glass::softmax_scratch_bytes<double>(16) == 16 * sizeof(double), "softmax scratch = n*T");
constexpr uint32_t k_inv_dims[] = {4u, 6u};
static_assert(glass::inv_fused_scratch_bytes<float>(2, k_inv_dims) > 0, "K-way fused inv scratch positive");
static_assert(glass::eigh_sweeps<double>() > glass::eigh_sweeps<float>(), "f64 needs more Jacobi sweeps");
static_assert(glass::syev_eps<float>() > 0 && glass::syev_eps<double>() < glass::syev_eps<float>(), "syev eps ordered by precision");
int main() { printf("ok\n"); return 0; }
