#pragma once
#include <cstdint>
#include <cublasdx.hpp>

// glass::nvidia query API — host- and device-callable constexpr helpers that
// answer "what BlockDim does cuBLASDx pick for this (T,M,N,K,SM)?" without
// requiring a DEFINE_NVIDIA_GEMM* macro to have been called first. This lets
// callers (e.g. a code-generator) pick a MAX_PERF_LEVEL_THREADS value at generation
// time and emit the matching DEFINE_NVIDIA_GEMM_BLOCKDIM call.
//
// For dispatch/diagnostic helpers that work without cuBLASDx (e.g.
// should_use_cublasdx, print_dispatch, gemm_batched_1d_block_threads_valid),
// see query_simt.cuh (included unconditionally by glass-nvidia.cuh).
//
// Example:
//   static_assert(glass::nvidia::min_block_threads<float, 6, 6, 6>() > 0);
//   constexpr uint32_t TC = glass::nvidia::min_block_threads<float, 6, 6, 6>();
//   // -> emit DEFINE_NVIDIA_GEMM_BLOCKDIM(6, 6, 6, TC) and launch with TC.

#ifndef SMS
#define SMS 860
#endif

// -- gemm queries -----------------------------------------------------------

// Returns the natural block_dim product cuBLASDx picks for an (M,N,K,SM) GEMM
// with no BlockDim operator. This is the smallest thread count cuBLASDx accepts
// without complaint (since BlockDim<TC,1,1>() with TC < this would either fail
// to compile or fall back to a degraded config).
/**
 * @brief Smallest BlockDim cuBLASDx will accept for a GEMM (host-callable).
 *
 * Constructs the cuBLASDx GEMM type inline and reads its natural `block_dim`
 * product — the minimum thread count to pin via `DEFINE_NVIDIA_GEMM_BLOCKDIM`.
 * No DEFINE macro required. Requires cuBLASDx / MathDx (MATHDX_ROOT). constexpr.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / C.
 * @tparam N      Columns of B / C.
 * @tparam K      Inner dimension.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 * @return Natural block thread count cuBLASDx picks.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t SM_VAL = SMS>
constexpr uint32_t gemm_min_block_threads()
{
    using GEMM = decltype(
        cublasdx::Size<M, N, K>()
        + cublasdx::Precision<float>()
        + cublasdx::Type<cublasdx::type::real>()
        + cublasdx::Function<cublasdx::function::MM>()
        + cublasdx::SM<SM_VAL>()
        + cublasdx::Block());
    return static_cast<uint32_t>(
        GEMM::block_dim.x * GEMM::block_dim.y * GEMM::block_dim.z);
}

// True iff BLOCK_THREADS is at least the natural block_dim cuBLASDx picks.
// (BlockDim<TC,1,1>() with TC >= the natural count is always accepted; smaller
// values are not.)
/**
 * @brief True iff BLOCK_THREADS is enough for a cuBLASDx GEMM (host-callable).
 *
 * Returns whether BLOCK_THREADS >= gemm_min_block_threads<T,M,N,K,SM_VAL>().
 * Use in a static_assert to validate a pinned launch thread count. Requires
 * cuBLASDx / MathDx (MATHDX_ROOT). constexpr.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A / C.
 * @tparam N             Columns of B / C.
 * @tparam K             Inner dimension.
 * @tparam BLOCK_THREADS Candidate launch thread count to validate.
 * @tparam SM_VAL        Target SM architecture (default = SMS).
 * @return true if BLOCK_THREADS meets the minimum.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BLOCK_THREADS, uint32_t SM_VAL = SMS>
constexpr bool gemm_block_threads_valid()
{
    return BLOCK_THREADS >= gemm_min_block_threads<T, M, N, K, SM_VAL>();
}

// -- gemv queries (gemv = GEMM with N=1) ------------------------------------

/**
 * @brief Smallest BlockDim cuBLASDx will accept for a GEMV (host-callable).
 *
 * GEMV is modeled as a `Size<M, 1, N>` GEMM; returns the natural block_dim
 * product. No DEFINE macro required. Requires cuBLASDx / MathDx (MATHDX_ROOT).
 * constexpr.
 *
 * @tparam T      Scalar type.
 * @tparam M      Rows of A / length of y.
 * @tparam N      Columns of A / length of x.
 * @tparam SM_VAL Target SM architecture (default = SMS).
 * @return Natural block thread count cuBLASDx picks.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t SM_VAL = SMS>
constexpr uint32_t gemv_min_block_threads()
{
    using GEMM = decltype(
        cublasdx::Size<M, 1, N>()
        + cublasdx::Precision<float>()
        + cublasdx::Type<cublasdx::type::real>()
        + cublasdx::Function<cublasdx::function::MM>()
        + cublasdx::SM<SM_VAL>()
        + cublasdx::Block());
    return static_cast<uint32_t>(
        GEMM::block_dim.x * GEMM::block_dim.y * GEMM::block_dim.z);
}

/**
 * @brief True iff BLOCK_THREADS is enough for a cuBLASDx GEMV (host-callable).
 *
 * Returns whether BLOCK_THREADS >= gemv_min_block_threads<T,M,N,SM_VAL>().
 * Requires cuBLASDx / MathDx (MATHDX_ROOT). constexpr.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A.
 * @tparam N             Columns of A.
 * @tparam BLOCK_THREADS Candidate launch thread count to validate.
 * @tparam SM_VAL        Target SM architecture (default = SMS).
 * @return true if BLOCK_THREADS meets the minimum.
 */
template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS, uint32_t SM_VAL = SMS>
constexpr bool gemv_block_threads_valid()
{
    return BLOCK_THREADS >= gemv_min_block_threads<T, M, N, SM_VAL>();
}

