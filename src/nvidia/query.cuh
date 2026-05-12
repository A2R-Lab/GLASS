#pragma once
#include <cstdint>
#include <cublasdx.hpp>

// glass::nvidia query API — host- and device-callable constexpr helpers that
// answer "what BlockDim does cuBLASDx pick for this (T,M,N,K,SM)?" without
// requiring a DEFINE_NVIDIA_GEMM* macro to have been called first. This lets
// callers (e.g. a code-generator) pick a SUGGESTED_THREADS value at generation
// time and emit the matching DEFINE_NVIDIA_GEMM_BLOCKDIM call.
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
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BLOCK_THREADS, uint32_t SM_VAL = SMS>
constexpr bool gemm_block_threads_valid()
{
    return BLOCK_THREADS >= gemm_min_block_threads<T, M, N, K, SM_VAL>();
}

// -- gemv queries (gemv = GEMM with N=1) ------------------------------------

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

template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS, uint32_t SM_VAL = SMS>
constexpr bool gemv_block_threads_valid()
{
    return BLOCK_THREADS >= gemv_min_block_threads<T, M, N, SM_VAL>();
}
