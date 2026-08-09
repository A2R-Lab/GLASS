// test_defaults.cu — compile-time validation of glass-defaults.cuh. The helpers are
// constexpr, so the static_asserts ARE the test: if this compiles, the picks match the
// sweep (bench/MEGA_SWEEP_RESULTS.md). No MathDx needed — this TU links no vendor lib, so
// suggested_backend<> exercises the no-nvidia COLLAPSE, while ideal_sm120() is checked
// directly for the nvidia-tier picks (it's availability-independent).
#include <cstdio>
#include "glass.cuh"
#include "glass-defaults.cuh"

using glass::op; using glass::backend;
namespace gd = glass::defaults;

// ── measured sm_120 ladder (the ideal tier, independent of what's linked) ──
//   (2026-07-18 retune — first sweep with the THREAD contender: it takes the
//    low-DOF corner of every op except gemm.)
//   gemm f32: warp <=16, block <=24, nvidia 25..32, block >=48 — no thread
//   (the old warp/block interleave <=24 was sub-noise; the ±2% SIMT tie band
//    resolves it to the simpler tier — see bench/tune_pick.py)
static_assert(gd::ideal_sm120(op::gemm, 8,  false) == backend::warp,   "gemm8 f32");
static_assert(gd::ideal_sm120(op::gemm, 12, false) == backend::warp,   "gemm12 f32 (SIMT tie -> simpler tier; old warp/block zigzag collapsed)");
static_assert(gd::ideal_sm120(op::gemm, 24, false) == backend::block,  "gemm24 f32");
static_assert(gd::ideal_sm120(op::gemm, 32, false) == backend::nvidia, "gemm32 f32");
static_assert(gd::ideal_sm120(op::gemm, 96, false) == backend::block,  "gemm96 f32 (smem cap)");
//   chol f32: thread<=6, warp<=12, nvidia>=16 (through 128)
static_assert(gd::ideal_sm120(op::chol, 4,   false) == backend::thread, "chol4 f32");
static_assert(gd::ideal_sm120(op::chol, 8,   false) == backend::warp,   "chol8 f32");
static_assert(gd::ideal_sm120(op::chol, 24,  false) == backend::nvidia, "chol24 f32");
static_assert(gd::ideal_sm120(op::chol, 128, false) == backend::nvidia, "chol128 f32");
//   trsv f32: thread<=16, nvidia 17..32, warp above
static_assert(gd::ideal_sm120(op::trsv, 12, false) == backend::thread, "trsv12 f32");
static_assert(gd::ideal_sm120(op::trsv, 24, false) == backend::nvidia, "trsv24 f32");
static_assert(gd::ideal_sm120(op::trsv, 64, false) == backend::warp,   "trsv64 f32");
//   dot: thread<=16, warp above ; gemv: thread<=6, warp<=32, block@48
static_assert(gd::ideal_sm120(op::dot,  8,   false) == backend::thread, "dot8");
static_assert(gd::ideal_sm120(op::dot,  128, false) == backend::warp,  "dot128");
static_assert(gd::ideal_sm120(op::gemv, 4,   false) == backend::thread, "gemv4");
static_assert(gd::ideal_sm120(op::gemv, 32,  false) == backend::warp,  "gemv32");
static_assert(gd::ideal_sm120(op::gemv, 48,  false) == backend::block, "gemv48");
//   f64: thread reaches N<=16 on chol/trsv/posv (block/warp f64 small-N is slow
//   enough that even the spilled thread path wins); nvidia bands as before
static_assert(gd::ideal_sm120(op::chol, 8,  true) == backend::thread, "chol8 f64");
static_assert(gd::ideal_sm120(op::chol, 48, true) == backend::nvidia, "chol48 f64");
static_assert(gd::ideal_sm120(op::gemm, 64, true) == backend::block,  "gemm64 f64");
static_assert(gd::ideal_sm120(op::posv, 12, true) == backend::thread, "posv12 f64");
static_assert(gd::ideal_sm120(op::posv, 64, true) == backend::nvidia, "posv64 f64");

// ── sm_87 (Jetson AGX Orin, 50W standard mode; mega_sweep_50W_merged.txt) ──
// The integrated-memory Orin keeps the thread tier winning further out than
// sm_120 does, and the vendor tier (reachable on Tegra only via the cuSOLVERDx
// LTO-IR fatbin device link) takes the small-N factor/solve band outright.
static_assert(gd::ideal_sm87(op::dot,   8,  false) == backend::thread, "dot8 f32");
static_assert(gd::ideal_sm87(op::dot,   128, false) == backend::warp,  "dot128 f32");
static_assert(gd::ideal_sm87(op::chol,  8,  false) == backend::thread, "chol8 f32");
static_assert(gd::ideal_sm87(op::chol,  48, false) == backend::nvidia, "chol48 f32 -> vendor");
static_assert(gd::ideal_sm87(op::posv,  32, false) == backend::nvidia, "posv32 f32 -> vendor");
static_assert(gd::ideal_sm87(op::posv,  16, false) == backend::thread, "posv16 f32");
static_assert(gd::ideal_sm87(op::gemm,  64, false) == backend::nvidia, "gemm64 f32 -> vendor");
static_assert(gd::ideal_sm87(op::gemm,  64, true)  == backend::warp,   "gemm64 f64 (SIMT tie: warp within 1% of block to N=96)");
static_assert(gd::ideal_sm87(op::gemm, 128, true)  == backend::block,  "gemm128 f64 (block's only real win, 24% faster)");
static_assert(gd::ideal_sm87(op::chol,  24, true)  == backend::thread, "chol24 f64 (thread reaches further than sm_120)");

// ── per-arch dispatch: a measured SM hits its table, an unmeasured SM falls to generic ──
static_assert(gd::ideal(op::gemm, 32, false, 1200u) == gd::ideal_sm120(op::gemm, 32, false), "sm_120 dispatches to its table");
static_assert(gd::ideal(op::posv, 64, true,  1200u) == gd::ideal_sm120(op::posv, 64, true),  "sm_120 dispatches to its table (f64)");
static_assert(gd::ideal(op::chol, 48, false, 870u)  == gd::ideal_sm87(op::chol, 48, false),  "sm_87 dispatches to its table");
static_assert(gd::ideal(op::posv, 64, true,  870u)  == gd::ideal_sm87(op::posv, 64, true),   "sm_87 dispatches to its table (f64)");
// (sentinel SMs no sweep will ever produce — a real new arch, e.g. sm_87 on Jetson,
//  gets its own table + dispatch case from tune.py and must NOT be asserted generic here)
static_assert(gd::ideal(op::gemm, 32, false, 0u) == gd::ideal_generic(op::gemm, 32, false), "unmeasured SM falls to generic");
static_assert(gd::ideal(op::chol, 24, false, 1u) == gd::ideal_generic(op::chol, 24, false), "unmeasured SM falls to generic");

// ── blas2 family (warp-vs-block; tune.py blas2 leg, blas2_sweep_20260718_0327) ──
static_assert(gd::blas2_sm120(op::syrk,  16, false) == backend::warp,  "syrk16 f32 -> warp");
static_assert(gd::blas2_sm120(op::syrk,  24, false) == backend::block, "syrk24 f32 -> block");
static_assert(gd::blas2_sm120(op::ldlt,  32, false) == backend::warp,  "ldlt32 f32 -> warp");
static_assert(gd::blas2_sm120(op::ldlt,  32, true)  == backend::block, "ldlt32 f64 -> block");
static_assert(gd::blas2_sm120(op::syr2k,  8, true)  == backend::warp,  "syr2k8 f64 (2.0% gap inside SIMT tie band -> simpler tier)");
static_assert(gd::ideal(op::syrk, 16, false, 1200u) == gd::blas2_sm120(op::syrk, 16, false), "blas2 ops route through ideal()");
static_assert(gd::ideal(op::ldltsv, 32, false, 870u) == backend::block, "blas2 unmeasured arch -> block incumbent");
static_assert(glass::suggested_backend<op::ldlt, 32, float, 1200u>() == backend::warp, "public picker reaches blas2 table");

// ── rect exact-shape pickers (tune.py rect leg, rect_sweep_20260718_0328) ──
static_assert(gd::rect_gemv_sm120( 64,  8, false) == backend::warp,  "gemv 64x8 f32 tall -> warp");
static_assert(gd::rect_gemv_sm120(128, 16, false) == backend::block, "gemv 128x16 f32 -> block");
static_assert(gd::rect_gemv_sm120(128, 16, true)  == backend::warp,  "gemv 128x16 f64 -> warp");
static_assert(gd::rect_gemm_sm120( 6,  6, 64, false) == backend::block, "gemm 6x6x64 wide -> block");
static_assert(gd::rect_gemm_sm120(32,  8, 32, false) == backend::warp,  "gemm 32x8x32 -> warp");
static_assert(glass::suggested_backend_rect_gemv<64, 8, float, 1200u>() == backend::warp, "public rect gemv picker");
static_assert(glass::suggested_backend_rect_gemm<7, 7, 7, float, 1200u>() == backend::block, "unmeasured rect shape -> block");
static_assert(glass::suggested_backend_rect_gemv<64, 8, float, 870u>() == backend::block, "rect unmeasured arch -> block");

// ── bare-namespace face: Phase-2 pins (2026-07-30 body sweep, sm_120) ──
// dispatch_body() now carries the measured body_sm120 table
// (bench/body_dispatch_sweep_20260730_2041.txt; rule = never worse than block
// by >5% at any measured (NPROB, TB), >5% faster at >=1 TB @ NPROB=8192,
// bounded at the largest measured N). Spot-pin the moved cells + the rule's
// conservative refusals:
using glass::body;
static_assert(glass::dispatch_body(op::dot,   8, false) == body::warp_in_block,   "dot8 f32 -> warp body");
static_assert(glass::dispatch_body(op::dot,  32, false) == body::thread_in_block, "dot32 f32 -> thread body");
static_assert(glass::dispatch_body(op::dot, 128, false) == body::block,           "dot128 f32 BOUNDED -> block");
static_assert(glass::dispatch_body(op::trsv, 16, false) == body::warp_in_block,   "trsv16 f32 -> warp body");
static_assert(glass::dispatch_body(op::trsv, 64, false) == body::block,           "trsv64 f32 stays block");
static_assert(glass::dispatch_body(op::posv,  4, false) == body::thread_in_block, "posv4 f32 -> thread body");
static_assert(glass::dispatch_body(op::posv, 16, true)  == body::block,           "posv f64 all block");
static_assert(glass::dispatch_body(op::eig3,  3, true)  == body::thread_in_block, "eig3 f64 -> thread body");
static_assert(glass::dispatch_body(op::eig3,  3, false) == body::block,           "eig3 f32 TB-unstable -> block");
static_assert(glass::dispatch_body(op::softmax, 16, false) == body::warp_in_block, "softmax16 f32 -> warp body");
static_assert(glass::dispatch_body(op::softmax, 4096, false) == body::block,      "softmax large-n BOUNDED -> block");
static_assert(glass::dispatch_body(op::gemm, 16, false) == body::block,           "gemm never moves");
// sm_87 body table (body_dispatch_sweep_20260803_0936.txt, 50W): the same rule
// moves 23 cells there; softmax/eig3 land identically, dot's warp band runs wider.
static_assert(glass::dispatch_body(op::dot,   8, false, 870u) == body::thread_in_block, "sm_87 dot8 -> thread body");
static_assert(glass::dispatch_body(op::dot,  32, false, 870u) == body::warp_in_block,   "sm_87 dot32 -> warp body");
static_assert(glass::dispatch_body(op::dot, 128, false, 870u) == body::block,           "sm_87 dot128 BOUNDED -> block");
static_assert(glass::dispatch_body(op::chol, 16, false, 870u) == body::block,           "sm_87 chol never moves");
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

// ── no-nvidia collapse (this TU links no vendor lib) ──
static_assert(glass::suggested_backend<op::chol, 24, float>() == backend::warp,  "chol24 collapses to warp");
static_assert(glass::suggested_backend<op::chol, 64, float>() == backend::block, "chol64 collapses to block");
static_assert(glass::suggested_backend<op::gemm, 32, float>() == backend::block, "gemm32 collapses to block");
static_assert(glass::suggested_backend<op::trsv, 24, float>() == backend::warp,  "trsv24 collapses to warp");
static_assert(glass::suggested_backend<op::dot,  32, float>() == backend::warp,  "dot stays warp");
static_assert(glass::suggested_backend<op::gemv, 48, float>() == backend::block, "gemv48 stays block");

// ── launch-config helpers ──
static_assert(glass::suggested_block_threads<op::chol, 32, float>() == 32u, "chol TB=32");
static_assert(glass::suggested_block_threads<op::posv, 64, float>() == 32u, "posv TB=32");
static_assert(glass::suggested_block_threads<op::gemm, 8,  float>() == 64u, "gemm8 TB=64");
static_assert(glass::suggested_block_threads<op::gemm, 32, float>() == 256u, "gemm32 TB=256");
static_assert(glass::suggested_warps_per_block<op::dot>()  == 8u, "dot WPB=8");
static_assert(glass::suggested_warps_per_block<op::chol>() == 2u, "chol WPB=2");

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
constexpr uint32_t k_inv_dims[] = {4u, 6u};
static_assert(glass::inv_fused_scratch_bytes<float>(2, k_inv_dims) > 0, "K-way fused inv scratch positive");
static_assert(glass::eigh_sweeps<double>() > glass::eigh_sweeps<float>(), "f64 needs more Jacobi sweeps");
static_assert(glass::syev_eps<float>() > 0 && glass::syev_eps<double>() < glass::syev_eps<float>(), "syev eps ordered by precision");
static_assert(!glass::suggested_use_reduced<4, 8, 128>(), "reduced corner empty on sm_120 (REDUCED_SWEEP_RESULTS)");

int main() { printf("ok\n"); return 0; }
