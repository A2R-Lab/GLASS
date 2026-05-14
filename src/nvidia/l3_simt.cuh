#pragma once
#include <cstdint>
#include <cassert>
// types.cuh provides the `layout` enum used by the public glass::nvidia API.
// glass.cuh (included by glass-nvidia.cuh) provides ::glass::gemm_impl_ct,
// the SIMT GEMM core that this file delegates to per batch element.
#include "./types.cuh"

// glass::nvidia L3 — SIMT-only batched APIs that work in a 1D launch.
//
// These exist because the cuBLASDx-backed gemm_batched in l3.cuh requires a
// 2D launch dim3(TC, BATCH) — cuBLASDx's Block() collective consumes
// threadIdx.x directly per-GEMM, leaving no clean way to multiplex BATCH
// independent GEMMs across one threadIdx.x range. Callers (notably GRiD-A2R)
// that launch with 1D thread blocks could not use it.
//
// This file is included unconditionally by glass-nvidia.cuh (no cuBLASDx
// dependency), so these APIs are available even in builds that don't link
// cuBLASDx. Best for the small shapes (M,N,K ≲ 8) where cuBLASDx's tile-load
// overhead dominates anyway. Use the cuBLASDx-backed gemm_batched for shapes
// where cuBLASDx wins (see should_use_cublasdx<T,M,N,K,SM> in query.cuh, once
// PR2 lands).

// ---------------------------------------------------------------------------
// gemm_batched_1d (P0-1) — runs BATCH independent (M×N)·(N×K) GEMMs in a
// single CUDA block, with each batch element getting TC threads.
//
// Required launch:  any geometry with blockDim.x*y*z >= TC*BATCH; the
//                   canonical form is <<<grid, dim3(TC*BATCH, 1, 1)>>>.
// Required smem:    none.
//
// A, B, C are arrays of length BATCH; each element points at one matrix.
// Templated on T (no DEFINE macro needed) — the SIMT path has no precompiled
// specialization to instantiate, so any T (float, double, half, ...) works.
// ---------------------------------------------------------------------------
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
__device__ void gemm_batched_1d(T alpha, T* const* A, T* const* B,
                                T beta,  T* const* C)
{
    static_assert(BATCH > 0, "glass::nvidia::gemm_batched_1d: BATCH must be > 0");
    static_assert(TC > 0,    "glass::nvidia::gemm_batched_1d: TC must be > 0");
#ifndef NDEBUG
    assert((blockDim.x * blockDim.y * blockDim.z) >= TC * BATCH &&
           "glass::nvidia::gemm_batched_1d: launched threads < TC*BATCH");
#endif
    constexpr bool RM_A = (LA == layout::row_major);
    constexpr bool RM_B = (LB == layout::row_major);
    constexpr bool RM_C = (LC == layout::row_major);
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t b    = rank / TC;
    if (b >= BATCH) return;
    uint32_t tx   = rank - b * TC;
    // Reuse the well-tested compile-time SIMT gemm core from glass::.
    // Inner loop is fully unrolled; no inter-batch synchronization needed
    // because batches use disjoint thread sets and disjoint output buffers.
    ::glass::gemm_impl_ct<T, M, N, K, /*TRANSPOSE_B=*/false, RM_A, RM_B, RM_C>(
        tx, TC, alpha, A[b], B[b], beta, C[b]);
}

// Smem requirement is zero; provided for API symmetry with gemm_batched.
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
constexpr std::size_t gemm_batched_1d_smem_size() { return 0; }

// Total threads required across the whole 1D block.
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
constexpr uint32_t gemm_batched_1d_threads() { return TC * BATCH; }

// ---------------------------------------------------------------------------
// gemm_strided_batched_1d (P0-2) — 1D-launch batched GEMM with one shared A
// matrix broadcast across BATCH (B,C) pairs.
//
// Common pattern in GRiD's EE-pose-gradient: a single 4×4 transform A is
// applied to BATCH consecutive 4×4 destinations. Avoids constructing 3
// pointer arrays on the caller side; instead pass A directly, plus base
// pointers + element strides for B and C.
//
//   B element address for batch b:  B + b * B_STRIDE
//   C element address for batch b:  C + b * C_STRIDE
//
// Defaults for B_STRIDE / C_STRIDE assume tightly packed batches (B is N×K,
// C is M×K). Override for non-contiguous storage.
//
// Only B and C are strided — A is always shared. Use gemm_batched_1d if a
// per-batch A is needed.
// ---------------------------------------------------------------------------
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          uint32_t B_STRIDE = N * K,
          uint32_t C_STRIDE = M * K,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
__device__ void gemm_strided_batched_1d(T alpha, const T* A_shared, T* B,
                                        T beta,  T* C)
{
    static_assert(BATCH > 0, "glass::nvidia::gemm_strided_batched_1d: BATCH must be > 0");
    static_assert(TC > 0,    "glass::nvidia::gemm_strided_batched_1d: TC must be > 0");
#ifndef NDEBUG
    assert((blockDim.x * blockDim.y * blockDim.z) >= TC * BATCH &&
           "glass::nvidia::gemm_strided_batched_1d: launched threads < TC*BATCH");
#endif
    constexpr bool RM_A = (LA == layout::row_major);
    constexpr bool RM_B = (LB == layout::row_major);
    constexpr bool RM_C = (LC == layout::row_major);
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t b    = rank / TC;
    if (b >= BATCH) return;
    uint32_t tx   = rank - b * TC;
    // const_cast: gemm_impl_ct's signature takes T*, but it only reads A.
    // Safe because every batch element only reads — never writes — A_shared.
    ::glass::gemm_impl_ct<T, M, N, K, /*TRANSPOSE_B=*/false, RM_A, RM_B, RM_C>(
        tx, TC, alpha, const_cast<T*>(A_shared),
                       B + b * B_STRIDE,
        beta,          C + b * C_STRIDE);
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          uint32_t B_STRIDE = N * K, uint32_t C_STRIDE = M * K,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
constexpr std::size_t gemm_strided_batched_1d_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          uint32_t B_STRIDE = N * K, uint32_t C_STRIDE = M * K,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major>
constexpr uint32_t gemm_strided_batched_1d_threads() { return TC * BATCH; }
