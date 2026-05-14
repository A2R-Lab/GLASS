#pragma once
#include "glass.cuh"

// System headers must be included before namespace wrapping so their internal
// declarations land at global scope. The re-includes inside the namespace below
// are no-ops due to #pragma once.
#include <cstdint>
#include <cub/cub.cuh>

#ifdef __CUBLASDX_HPP__
#define GLASS_HAVE_CUBLASDX 1
#else
// cublasdx.hpp not yet included; try to include it now.
// Set MATHDX_ROOT and pass -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#define GLASS_HAVE_CUBLASDX 1
#else
#define GLASS_HAVE_CUBLASDX 0
#endif
#endif

// cuSOLVERDx detection (for L3 LAPACK wrappers: chol_inplace, trsm).
// Both cusolverdx.hpp AND cusolverdx_io.hpp are required at GLOBAL scope so
// their symbols don't get nested inside `namespace glass::nvidia` when
// lapack.cuh is included below.
#ifdef CUSOLVERDX_HPP
#define GLASS_HAVE_CUSOLVERDX 1
#include <cusolverdx_io.hpp>
#else
#ifdef GLASS_BENCH_CUSOLVERDX
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>
#define GLASS_HAVE_CUSOLVERDX 1
#else
#define GLASS_HAVE_CUSOLVERDX 0
#endif
#endif

// types.cuh defines `glass::nvidia::layout` and the shared helper macros
// (_GLASS_CUBLAS_LAYOUT, _GLASS_ASSERT_BLOCKDIM_GEQ). It must be included
// before any of l1/l2/l3/lapack so the `layout` enum is in scope inside the
// `glass::nvidia` namespace where the wrapper headers expand.
#include "./src/nvidia/types.cuh"

namespace glass {
namespace nvidia {

    // L1: CUB-backed reduce / dot / l2norm
    #include "./src/nvidia/l1.cuh"

    // SIMT-only query helpers: should_use_cublasdx<>, print_dispatch<>,
    // gemm_batched_1d_block_threads_valid<>. No cuBLASDx dependency.
    #include "./src/nvidia/query_simt.cuh"

    // L3 SIMT-only: 1D-launch batched GEMMs (no cuBLASDx dependency, so
    // available even in builds that don't link cuBLASDx).
    #include "./src/nvidia/l3_simt.cuh"

#if GLASS_HAVE_CUBLASDX
    // L2: cuBLASDx-backed gemv  (primary templates + DEFINE_NVIDIA_GEMV* macros)
    #include "./src/nvidia/l2.cuh"

    // L3: cuBLASDx-backed gemm  (primary templates + DEFINE_NVIDIA_GEMM* macros,
    // including BLOCKDIM, LAYOUT, SM, and BATCHED variants)
    #include "./src/nvidia/l3.cuh"

    // Host-side BlockDim query API (constexpr — no DEFINE macro required).
    #include "./src/nvidia/query.cuh"

    // Pre-instantiated standard sizes for the GEMV benches.
    DEFINE_NVIDIA_GEMV(4,   4)
    DEFINE_NVIDIA_GEMV(6,   6)
    DEFINE_NVIDIA_GEMV(8,   8)
    DEFINE_NVIDIA_GEMV(12, 12)
    DEFINE_NVIDIA_GEMV(14, 14)
    DEFINE_NVIDIA_GEMV(24, 24)
    DEFINE_NVIDIA_GEMV(64, 64)

    // GEMM pre-instantiations: only emit cuBLASDx specializations for shapes
    // where should_use_cublasdx<float,M,N,K,SMS>() returns true. Smaller
    // shapes are SIMT-dispatched by the primary template (P1-4) and don't
    // need a cuBLASDx specialization. The shape list below mirrors what
    // bench/autotune.py would emit; regenerate src/nvidia/tuning_table.cuh
    // and update this list to retune for a particular SM.
    //
    // Default heuristic (max(M,N,K) >= 16): all shapes below qualify.
    DEFINE_NVIDIA_GEMM(16, 16, 16)
    DEFINE_NVIDIA_GEMM(24, 24, 24)
    DEFINE_NVIDIA_GEMM(32, 32, 32)
    DEFINE_NVIDIA_GEMM(64, 64, 64)
    // Shapes (4×4×4) through (14×14×14) are now SIMT-dispatched — no DEFINE
    // needed. To force cuBLASDx for any of those shapes, add an explicit
    // DEFINE_NVIDIA_GEMM(M,N,K) in your .cu file (it will override the SIMT
    // fallback in the primary template).
#endif // GLASS_HAVE_CUBLASDX

#if GLASS_HAVE_CUSOLVERDX
    // LAPACK: cuSOLVERDx-backed chol_inplace + trsm
    // (primary templates + DEFINE_NVIDIA_CHOL* / DEFINE_NVIDIA_TRSM* macros)
    #include "./src/nvidia/lapack.cuh"
#endif // GLASS_HAVE_CUSOLVERDX

} // namespace nvidia
} // namespace glass
