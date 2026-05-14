#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cublasdx.hpp>
#include "./tuning_table.cuh"

// glass::nvidia query API — host- and device-callable constexpr helpers.
//
// Two families live here:
//   1. *_min_block_threads / *_block_threads_valid — answer "what BlockDim
//      does cuBLASDx pick for this (T,M,N,K,SM)?" without requiring a
//      DEFINE_NVIDIA_*<size> macro to have been called first. Useful for
//      codegen that wants to pick a SUGGESTED_THREADS value at generation
//      time.
//
//   2. should_use_cublasdx*<> — answer "for this compile-time shape, does
//      cuBLASDx beat the hand-rolled SIMT path?" Drives the auto-dispatch
//      in the gemm<>/gemv<>/row_strided_*/gemm_batched_1d<> primary
//      templates. Decision priority: tuning::kLocalTable hit → kGlobalTable
//      hit → heuristic (compile-time constants below).
//
//   3. print_dispatch* — host-only helpers that report which path a shape
//      would take and (in the _full variant) where the decision came from.

#ifndef SMS
#define SMS 860
#endif

// ---------------------------------------------------------------------------
// gemm queries — natural BlockDim cuBLASDx picks
// ---------------------------------------------------------------------------

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

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BLOCK_THREADS, uint32_t SM_VAL = SMS>
constexpr bool gemm_block_threads_valid()
{
    return BLOCK_THREADS >= gemm_min_block_threads<T, M, N, K, SM_VAL>();
}

// ---------------------------------------------------------------------------
// gemv queries (gemv = GEMM with N=1)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// should_use_cublasdx*<> — compile-time SIMT vs cuBLASDx dispatch decisions.
//
// Each variant consults the tuning table first (local → global) and falls
// back to a per-API heuristic when no entry matches. The heuristic constants
// match what the RFC suggested as reasonable starting points; bench
// measurements update tuning_table.cuh to override them per (shape, SM).
// ---------------------------------------------------------------------------

namespace _dispatch_heuristic {
    constexpr uint32_t kMax3(uint32_t a, uint32_t b, uint32_t c) {
        return (a > b ? a : b) > c ? (a > b ? a : b) : c;
    }
    constexpr uint32_t kMin3(uint32_t a, uint32_t b, uint32_t c) {
        return (a < b ? a : b) < c ? (a < b ? a : b) : c;
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx()
{
    constexpr bool heuristic =
        (_dispatch_heuristic::kMax3(M, N, K) >= 16) &&
        (_dispatch_heuristic::kMin3(M, N, K) >= 4);
    return tuning::lookup<tuning::api::gemm, M, N, K, 1, SM_VAL>(heuristic).use_cublasdx;
}

template <typename T, uint32_t M, uint32_t N, uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_gemv()
{
    // GEMV is less arithmetic-dense than GEMM; SIMT competes farther up.
    constexpr bool heuristic = (M >= 32) || (N >= 32);
    return tuning::lookup<tuning::api::gemv, M, N, 1, 1, SM_VAL>(heuristic).use_cublasdx;
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS, uint32_t B_RS, uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_row_strided_gemm()
{
    // Same shape sensitivity as gemm; packing cost is constant per call so it
    // doesn't move the SIMT-vs-cuBLASDx tipping point much.
    constexpr bool heuristic =
        (_dispatch_heuristic::kMax3(M, N, K) >= 16) &&
        (_dispatch_heuristic::kMin3(M, N, K) >= 4);
    return tuning::lookup<tuning::api::row_strided_gemm, M, N, K, 1, SM_VAL>(heuristic).use_cublasdx;
}

template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE,
          uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_row_strided_gemv()
{
    constexpr bool heuristic = (M >= 32) || (N >= 32);
    return tuning::lookup<tuning::api::row_strided_gemv, M, N, 1, 1, SM_VAL>(heuristic).use_cublasdx;
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t SM_VAL = SMS>
constexpr bool should_use_cublasdx_batched()
{
    // BATCH multiplies compute density. RFC suggested starting point.
    constexpr bool heuristic =
        (BATCH >= 8) &&
        (_dispatch_heuristic::kMax3(M, N, K) >= 8);
    return tuning::lookup<tuning::api::gemm_batched_1d, M, N, K, BATCH, SM_VAL>(heuristic).use_cublasdx;
}

// ---------------------------------------------------------------------------
// print_dispatch<> — host-only debug helpers. Report which path the dispatch
// machinery would take for a given shape.
//
// The _full variants also report the source (local table / global table /
// heuristic), the table index if applicable, and what the heuristic alone
// would have predicted (so you can spot when the table overrides it).
// ---------------------------------------------------------------------------

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t SM = SMS>
inline void print_dispatch_gemm()
{
    std::printf("gemm<float,%u,%u,%u,sm_%u> -> %s\n",
                M, N, K, SM,
                should_use_cublasdx<T, M, N, K, SM>() ? "cuBLASDx" : "SIMT");
}

template <typename T, uint32_t M, uint32_t N, uint32_t SM = SMS>
inline void print_dispatch_gemv()
{
    std::printf("gemv<float,%u,%u,sm_%u> -> %s\n",
                M, N, SM,
                should_use_cublasdx_gemv<T, M, N, SM>() ? "cuBLASDx" : "SIMT");
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = N, uint32_t SM = SMS>
inline void print_dispatch_row_strided_gemm()
{
    std::printf("row_strided_gemm<float,%u,%u,%u,A_RS=%u,B_RS=%u,sm_%u> -> %s\n",
                M, N, K, A_RS, B_RS, SM,
                should_use_cublasdx_row_strided_gemm<T, M, N, K, A_RS, B_RS, SM>()
                    ? "cuBLASDx" : "SIMT");
}

template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          uint32_t SM = SMS>
inline void print_dispatch_row_strided_gemv()
{
    std::printf("row_strided_gemv<float,%u,%u,RS=%u,sm_%u> -> %s\n",
                M, N, ROW_STRIDE, SM,
                should_use_cublasdx_row_strided_gemv<T, M, N, ROW_STRIDE, SM>()
                    ? "cuBLASDx" : "SIMT");
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t SM = SMS>
inline void print_dispatch_batched()
{
    std::printf("gemm_batched_1d<float,%u,%u,%u,BATCH=%u,sm_%u> -> %s\n",
                M, N, K, BATCH, SM,
                should_use_cublasdx_batched<T, M, N, K, BATCH, SM>()
                    ? "cuBLASDx" : "SIMT");
}

// ---------------------------------------------------------------------------
// print_dispatch_full — same as print_dispatch but also shows source.
// ---------------------------------------------------------------------------

inline const char* _dispatch_source_name(tuning::source s) {
    switch (s) {
        case tuning::source::local:     return "local-table";
        case tuning::source::global:    return "global-table";
        case tuning::source::heuristic: return "heuristic";
        default:                        return "none";
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t SM = SMS>
inline void print_dispatch_full_gemm()
{
    constexpr bool heuristic =
        (_dispatch_heuristic::kMax3(M, N, K) >= 16) &&
        (_dispatch_heuristic::kMin3(M, N, K) >= 4);
    constexpr auto d = tuning::lookup<tuning::api::gemm, M, N, K, 1, SM>(heuristic);
    std::printf("gemm<float,%u,%u,%u,sm_%u> -> %s  (source: %s, idx=%d, "
                "heuristic-says=%s)\n",
                M, N, K, SM,
                d.use_cublasdx ? "cuBLASDx" : "SIMT",
                _dispatch_source_name(d.from),
                d.table_index,
                heuristic ? "cuBLASDx" : "SIMT");
}

template <typename T, uint32_t M, uint32_t N, uint32_t SM = SMS>
inline void print_dispatch_full_gemv()
{
    constexpr bool heuristic = (M >= 32) || (N >= 32);
    constexpr auto d = tuning::lookup<tuning::api::gemv, M, N, 1, 1, SM>(heuristic);
    std::printf("gemv<float,%u,%u,sm_%u> -> %s  (source: %s, idx=%d, "
                "heuristic-says=%s)\n",
                M, N, SM,
                d.use_cublasdx ? "cuBLASDx" : "SIMT",
                _dispatch_source_name(d.from),
                d.table_index,
                heuristic ? "cuBLASDx" : "SIMT");
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t SM = SMS>
inline void print_dispatch_full_batched()
{
    constexpr bool heuristic =
        (BATCH >= 8) &&
        (_dispatch_heuristic::kMax3(M, N, K) >= 8);
    constexpr auto d = tuning::lookup<tuning::api::gemm_batched_1d, M, N, K, BATCH, SM>(heuristic);
    std::printf("gemm_batched_1d<float,%u,%u,%u,BATCH=%u,sm_%u> -> %s  "
                "(source: %s, idx=%d, heuristic-says=%s)\n",
                M, N, K, BATCH, SM,
                d.use_cublasdx ? "cuBLASDx" : "SIMT",
                _dispatch_source_name(d.from),
                d.table_index,
                heuristic ? "cuBLASDx" : "SIMT");
}
