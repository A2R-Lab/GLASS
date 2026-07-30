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
//   gemm f32: warp/block interleave <=24, nvidia 25..32, block >=48 — no thread
static_assert(gd::ideal_sm120(op::gemm, 8,  false) == backend::warp,   "gemm8 f32");
static_assert(gd::ideal_sm120(op::gemm, 12, false) == backend::block,  "gemm12 f32");
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

// ── per-arch dispatch: a measured SM hits its table, an unmeasured SM falls to generic ──
static_assert(gd::ideal(op::gemm, 32, false, 1200u) == gd::ideal_sm120(op::gemm, 32, false), "sm_120 dispatches to its table");
static_assert(gd::ideal(op::posv, 64, true,  1200u) == gd::ideal_sm120(op::posv, 64, true),  "sm_120 dispatches to its table (f64)");
// (sentinel SMs no sweep will ever produce — a real new arch, e.g. sm_87 on Jetson,
//  gets its own table + dispatch case from tune.py and must NOT be asserted generic here)
static_assert(gd::ideal(op::gemm, 32, false, 0u) == gd::ideal_generic(op::gemm, 32, false), "unmeasured SM falls to generic");
static_assert(gd::ideal(op::chol, 24, false, 1u) == gd::ideal_generic(op::chol, 24, false), "unmeasured SM falls to generic");

// ── bare-namespace face (2026-07-30 restructure): Phase-1 pins ──
// Every dispatch_body cell is the block body, and the bare glass:: names are
// the SAME entities as glass::block:: (using-directive re-export) — pinned by
// function-pointer identity. When a Phase-2 sweep moves cells, tune.py must
// regenerate the table AND add shadowing wrappers; these asserts then update
// as part of that attested retune.
static_assert(glass::dispatch_body(op::gemm, 8,  false) == glass::body::block, "phase-1: all cells block");
static_assert(glass::dispatch_body(op::chol, 64, true)  == glass::body::block, "phase-1: all cells block");
static_assert(glass::dispatch_body(op::dot,  4,  false) == glass::body::block, "phase-1: all cells block");
// (one op suffices: the re-export is a single using-directive, so it covers
//  every name identically — heavily-overloaded names like gemv/potrf can't be
//  address-compared without a disambiguating cast, but resolve the same way)
static_assert(&glass::dot<float, 8, true>  == &glass::block::dot<float, 8, true>,  "bare dot IS block::dot");
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

int main() { printf("ok\n"); return 0; }
