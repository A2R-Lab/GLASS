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
//
// TRAILING_SYNC: when true (default), the function emits a __syncthreads()
//                before returning so callers can read any batch's output
//                safely. When false, the caller is responsible for syncing
//                before reading another batch's output. Pass false when
//                fusing with subsequent work that does its own block-wide
//                barrier (e.g. a parallel_loop that begins with a sync), so
//                two back-to-back syncs collapse to one.
// ---------------------------------------------------------------------------
/**
 * @brief Pure-SIMT batched GEMM for a 1D launch (no cuBLASDx, no scratch).
 *
 * Runs BATCH independent `C = alpha*A*B + beta*C` products in one block, giving
 * each batch element TC threads carved out of a 1D launch of `>= TC*BATCH`
 * threads (canonical `dim3(TC*BATCH, 1, 1)`). Reuses the well-tested
 * compile-time SIMT gemm core; no shared memory and no DEFINE macro needed, so
 * any T works. Best for small shapes (max(M,N,K) ≲ 8) where cuBLASDx tile-load
 * overhead dominates. Use the cuBLASDx gemm_batched (l3.cuh) for larger shapes.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A / C.
 * @tparam N             Columns of B / C.
 * @tparam K             Inner dimension.
 * @tparam BATCH         Number of independent GEMMs.
 * @tparam TC            Threads assigned to each batch element.
 * @tparam LA            Memory layout of A.
 * @tparam LB            Memory layout of B.
 * @tparam LC            Memory layout of C.
 * @tparam TRAILING_SYNC Emit a trailing __syncthreads() before return (default true).
 * @param  alpha         Scaling factor for A*B.
 * @param  A             Array of BATCH pointers to the M×K matrices.
 * @param  B             Array of BATCH pointers to the K×N matrices.
 * @param  beta          Scaling factor for the incoming C.
 * @param  C             Array of BATCH pointers to the M×N output matrices.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          bool TRAILING_SYNC = true>
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
    if (b < BATCH) {
        uint32_t tx = rank - b * TC;
        // Reuse the well-tested compile-time SIMT gemm core from glass::.
        // Inner loop is fully unrolled; no inter-batch synchronization needed
        // because batches use disjoint thread sets and disjoint output buffers.
        ::glass::gemm_impl_ct<T, M, N, K, RM_A, RM_B, RM_C>(
            tx, TC, alpha, A[b], B[b], beta, C[b]);
    }
    if constexpr (TRAILING_SYNC) {
        __syncthreads();
    }
}

/**
 * @brief Shared-memory bytes needed by `gemm_batched_1d<...>` (host-callable).
 *
 * Always 0 (the SIMT path needs no scratch); provided for API symmetry with
 * the cuBLASDx gemm_batched. Template parameters match gemm_batched_1d<>.
 * constexpr.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          bool TRAILING_SYNC = true>
constexpr std::size_t gemm_batched_1d_scratch_bytes() { return 0; }

/**
 * @brief Total threads required across the 1D block for `gemm_batched_1d<...>`.
 *
 * Returns `TC * BATCH` — launch with at least this many threads. Host-callable
 * constexpr. Template parameters match gemm_batched_1d<>.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          bool TRAILING_SYNC = true>
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
// Defaults for B_STRIDE / C_STRIDE assume tightly packed batches (B is K×N,
// C is M×N). Override for non-contiguous storage.
//
// Only B and C are strided — A is always shared. Use gemm_batched_1d if a
// per-batch A is needed.
//
// TRAILING_SYNC: see gemm_batched_1d for semantics. Defaults to true so the
// function returns with all threads at a block-wide barrier; pass false when
// fusing with subsequent work that already does its own __syncthreads().
// ---------------------------------------------------------------------------
/**
 * @brief Pure-SIMT batched GEMM (1D launch) with one shared A across BATCH (B,C) pairs.
 *
 * Like gemm_batched_1d, but a single A matrix is broadcast to BATCH strided
 * (B, C) pairs: batch b reads `B + b*B_STRIDE`, writes `C + b*C_STRIDE`. Common
 * in GRiD's EE-pose-gradient (one transform applied to many destinations).
 * Avoids building pointer arrays caller-side. No cuBLASDx, no scratch, no
 * DEFINE macro. Strides default to tightly-packed batches.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A / C.
 * @tparam N             Columns of B / C.
 * @tparam K             Inner dimension.
 * @tparam BATCH         Number of (B,C) pairs sharing A.
 * @tparam TC            Threads assigned to each batch element.
 * @tparam B_STRIDE      Element stride between consecutive B matrices.
 * @tparam C_STRIDE      Element stride between consecutive C matrices.
 * @tparam LA            Memory layout of A.
 * @tparam LB            Memory layout of B.
 * @tparam LC            Memory layout of C.
 * @tparam TRAILING_SYNC Emit a trailing __syncthreads() before return (default true).
 * @param  alpha         Scaling factor for A*B.
 * @param  A_shared      Pointer to the single shared M×K matrix (read-only).
 * @param  B             Base pointer for the BATCH K×N matrices.
 * @param  beta          Scaling factor for the incoming C.
 * @param  C             Base pointer for the BATCH M×N output matrices.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          uint32_t B_STRIDE = N * K,
          uint32_t C_STRIDE = M * N,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          bool TRAILING_SYNC = true>
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
    if (b < BATCH) {
        uint32_t tx = rank - b * TC;
        // const_cast: gemm_impl_ct's signature takes T*, but it only reads A.
        // Safe because every batch element only reads — never writes — A_shared.
        ::glass::gemm_impl_ct<T, M, N, K, RM_A, RM_B, RM_C>(
            tx, TC, alpha, const_cast<T*>(A_shared),
                           B + b * B_STRIDE,
            beta,          C + b * C_STRIDE);
    }
    if constexpr (TRAILING_SYNC) {
        __syncthreads();
    }
}

/**
 * @brief Shared-memory bytes needed by `gemm_strided_batched_1d<...>` (host-callable).
 *
 * Always 0 (SIMT path needs no scratch); provided for API symmetry. Template
 * parameters match gemm_strided_batched_1d<>. constexpr.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          uint32_t B_STRIDE = N * K, uint32_t C_STRIDE = M * N,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          bool TRAILING_SYNC = true>
constexpr std::size_t gemm_strided_batched_1d_scratch_bytes() { return 0; }

/**
 * @brief Total threads required across the 1D block for `gemm_strided_batched_1d<...>`.
 *
 * Returns `TC * BATCH` — launch with at least this many threads. Host-callable
 * constexpr. Template parameters match gemm_strided_batched_1d<>.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BATCH, uint32_t TC,
          uint32_t B_STRIDE = N * K, uint32_t C_STRIDE = M * N,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          bool TRAILING_SYNC = true>
constexpr uint32_t gemm_strided_batched_1d_threads() { return TC * BATCH; }
