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

#if GLASS_HAVE_CUBLASDX
    // Host-side query API (constexpr) — defines should_use_cublasdx*<> and
    // BlockDim helpers. Included BEFORE l2/l3 because the auto-dispatching
    // primary templates in those headers consult should_use_cublasdx<>.
    #include "./src/nvidia/query.cuh"

    // L2: cuBLASDx-backed gemv (primary templates + DEFINE_NVIDIA_GEMV* macros).
    // Primary templates auto-dispatch to ::glass::gemv (SIMT) for shapes where
    // should_use_cublasdx_gemv<>() returns false, else require a DEFINE macro.
    #include "./src/nvidia/l2.cuh"

    // L3: cuBLASDx-backed gemm (primary templates + DEFINE_NVIDIA_GEMM* macros,
    // including BLOCKDIM, LAYOUT, SM, BATCHED, and the 1D-launch batched
    // variant). Same auto-dispatch story as L2.
    #include "./src/nvidia/l3.cuh"

    // Pre-instantiated cuBLASDx specializations for shapes the default
    // heuristic flags as "cuBLASDx wins". Smaller shapes intentionally fall
    // through to the auto-dispatch primary template, which routes to SIMT.
    // Consumers wanting cuBLASDx for a small size add DEFINE_NVIDIA_GEMM*
    // in their own TU; consumers wanting SIMT for a large size override
    // tuning_table.cuh (or define GLASS_TUNING_TABLE_LOCAL).
    DEFINE_NVIDIA_GEMV(24, 24)
    DEFINE_NVIDIA_GEMV(64, 64)

    DEFINE_NVIDIA_GEMM(24, 24, 24)
    DEFINE_NVIDIA_GEMM(64, 64, 64)

    // ---------------------------------------------------------------------
    // required_smem_for_dispatch_*<> — explicit-intent aliases.
    //
    // The *_smem_size<> functions already encode the dispatch decision
    // (return 0 for SIMT-routed shapes, cuBLASDx scratch size otherwise),
    // but these aliases make consumer code self-documenting:
    //
    //   constexpr std::size_t smem =
    //       glass::nvidia::required_smem_for_dispatch_gemm<float, M, N, K>();
    //   __shared__ char buf[smem];  // 0 bytes if every call SIMT-routes
    //
    // Codegen that accumulates scratch across many call sites can take
    // max() of these constexprs and either emit an `extern __shared__` only
    // when nonzero, or skip the buffer entirely.
    // ---------------------------------------------------------------------

    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              uint32_t BLOCK_THREADS = 0,
              layout LA = layout::col_major,
              layout LB = layout::col_major,
              layout LC = layout::col_major,
              uint32_t SM_VAL = SMS>
    constexpr std::size_t required_smem_for_dispatch_gemm() {
        return gemm_smem_size<T, M, N, K, BLOCK_THREADS, LA, LB, LC, SM_VAL>();
    }

    template <typename T, uint32_t M, uint32_t N,
              uint32_t BLOCK_THREADS = 0,
              layout LA = layout::col_major,
              layout LB = layout::col_major,
              layout LC = layout::col_major,
              uint32_t SM_VAL = SMS>
    constexpr std::size_t required_smem_for_dispatch_gemv() {
        return gemv_smem_size<T, M, N, BLOCK_THREADS, LA, LB, LC, SM_VAL>();
    }

    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              uint32_t A_RS = M, uint32_t B_RS = N,
              uint32_t BLOCK_THREADS = 0,
              layout LA = layout::col_major,
              layout LB = layout::col_major,
              layout LC = layout::col_major,
              uint32_t SM_VAL = SMS>
    constexpr std::size_t required_smem_for_dispatch_row_strided_gemm() {
        return row_strided_gemm_smem_size<T, M, N, K, A_RS, B_RS,
                                          BLOCK_THREADS, LA, LB, LC, SM_VAL>();
    }

    template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
              uint32_t BLOCK_THREADS = 0,
              layout LA = layout::col_major,
              layout LB = layout::col_major,
              layout LC = layout::col_major,
              uint32_t SM_VAL = SMS>
    constexpr std::size_t required_smem_for_dispatch_row_strided_gemv() {
        return row_strided_gemv_smem_size<T, M, N, ROW_STRIDE,
                                          BLOCK_THREADS, LA, LB, LC, SM_VAL>();
    }

    template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
              uint32_t BLOCK_THREADS = 0,
              layout LA = layout::col_major,
              layout LB = layout::col_major,
              layout LC = layout::col_major,
              uint32_t SM_VAL = SMS>
    constexpr std::size_t required_smem_for_dispatch_gemm_batched_1d() {
        return gemm_batched_1d_smem_size<T, M, N, K, BATCH,
                                         BLOCK_THREADS, LA, LB, LC, SM_VAL>();
    }
#endif // GLASS_HAVE_CUBLASDX

#if GLASS_HAVE_CUSOLVERDX
    // LAPACK: cuSOLVERDx-backed chol_inplace + trsm
    // (primary templates + DEFINE_NVIDIA_CHOL* / DEFINE_NVIDIA_TRSM* macros)
    #include "./src/nvidia/lapack.cuh"
#endif // GLASS_HAVE_CUSOLVERDX

} // namespace nvidia
} // namespace glass
