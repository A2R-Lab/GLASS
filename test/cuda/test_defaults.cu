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
//   gemm f32: warp<=8, block@12, nvidia 16..64, block>=96
static_assert(gd::ideal_sm120(op::gemm, 8,  false) == backend::warp,   "gemm8 f32");
static_assert(gd::ideal_sm120(op::gemm, 12, false) == backend::block,  "gemm12 f32");
static_assert(gd::ideal_sm120(op::gemm, 32, false) == backend::nvidia, "gemm32 f32");
static_assert(gd::ideal_sm120(op::gemm, 96, false) == backend::block,  "gemm96 f32 (smem cap)");
//   chol f32: warp<=12, nvidia>=16 (through 128)
static_assert(gd::ideal_sm120(op::chol, 8,   false) == backend::warp,   "chol8 f32");
static_assert(gd::ideal_sm120(op::chol, 24,  false) == backend::nvidia, "chol24 f32");
static_assert(gd::ideal_sm120(op::chol, 128, false) == backend::nvidia, "chol128 f32");
//   trsv f32: nvidia only mid-band 16..32, warp otherwise
static_assert(gd::ideal_sm120(op::trsv, 12, false) == backend::warp,   "trsv12 f32");
static_assert(gd::ideal_sm120(op::trsv, 24, false) == backend::nvidia, "trsv24 f32");
static_assert(gd::ideal_sm120(op::trsv, 64, false) == backend::warp,   "trsv64 f32");
//   dot: warp always ; gemv: warp<=32 / block>=48
static_assert(gd::ideal_sm120(op::dot,  128, false) == backend::warp,  "dot128");
static_assert(gd::ideal_sm120(op::gemv, 32,  false) == backend::warp,  "gemv32");
static_assert(gd::ideal_sm120(op::gemv, 48,  false) == backend::block, "gemv48");
//   f64: narrower nvidia band (chol tiny->block, gemm band 16..32, posv nv to 64)
static_assert(gd::ideal_sm120(op::chol, 8,  true) == backend::block,  "chol8 f64");
static_assert(gd::ideal_sm120(op::chol, 48, true) == backend::nvidia, "chol48 f64");
static_assert(gd::ideal_sm120(op::gemm, 64, true) == backend::block,  "gemm64 f64");
static_assert(gd::ideal_sm120(op::posv, 64, true) == backend::nvidia, "posv64 f64");

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
