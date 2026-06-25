#pragma once
#include <cstdint>

// ─── contraction-dimension-parallel GEMM (the "reduced" engine) ──────────────
//
// The default glass::gemm maps one thread to one OUTPUT element and sums the
// length-N contraction SERIALLY in that thread. When the output count is small
// relative to the block (n_out < blockDim) the spare threads sit idle. The
// `*_reduced` family flips the mapping: one WARP owns one output and its 32
// lanes split the contraction, combining with a single warp-shuffle reduce.
//
// This is a thread-utilization play, NOT a FLOP reduction — total MAC work is
// identical. It wins only when n_out < blockDim (idle threads to soak up) AND
// the contraction K amortizes the ~5-step shuffle tail; it is neutral/slower
// when n_out >= blockDim with a tiny K. See the bench (bench/bench_reduced.cu)
// and concepts/contraction_parallel for the measured crossover; pick with
// glass::suggested_use_reduced<>() rather than guessing.
//
// Thread-count invariance: each output is reduced by the SAME fixed 32-way tree
// regardless of how many warps the block has, so results are bit-identical at
// 32 / 64 / 96 / ... threads. A trailing partial warp (blockDim % 32) idles. A
// block with fewer than 32 threads falls back to a per-thread path that
// reproduces the EXACT shuffle-tree summation order in registers, so the result
// is bit-identical across the 32 boundary too (1 / 7 / 31 == 32 == 256).

// reduced_tree32 (the 32-way register tree that matches glass::warp::reduce's
// lane-0 rounding bit-for-bit) lives in L1/reduce.cuh so every L2/L3 *_reduced
// engine can share it. The sub-warp fallback below uses it for invariance.

/**
 * @brief Should a contraction-parallel `*_reduced` op be preferred over the serial one?
 *
 * Codegen / launch-time picker seeded by the measured crossover sweep
 * (`bench/REDUCED_SWEEP_RESULTS.md`, RTX 5090 / sm_120). On that hardware the
 * `*_reduced` family is **slower than serial in almost every configuration** —
 * it pays a warp-shuffle latency per output and idles most lanes when the
 * contraction is short. It is competitive only in the narrow corner where every
 * output gets its own warp (`n_out <= blockDim/32`, so no serial output loop on
 * top of the shuffle) AND the contraction is long enough to fill a warp and
 * amortize the shuffle tail (`K_contract >= 32`). Returns `false` otherwise —
 * i.e. recommends serial almost always. Not a device function (the choice is a
 * launch/codegen decision); `constexpr` so it folds at compile time.
 *
 * @tparam n_out       Output element count (e.g. M*K for gemm, M for gemv).
 * @tparam K_contract  Length of the contracted dimension.
 * @tparam blockDim    Launch thread count.
 * @return true to use the `*_reduced` variant, false to use the serial op.
 */
template <uint32_t n_out, uint32_t K_contract, uint32_t blockDim>
__host__ __device__ constexpr bool suggested_use_reduced() {
    return (n_out <= blockDim / 32u) && (K_contract >= 32u);
}

// Core: explicit (rank,size), compile-time dims + per-matrix layout flags,
// HAS_BETA selects whether C is read (false ⇒ overwrite, never touches C).
template <typename T, uint32_t M, uint32_t N, uint32_t K, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C, bool HAS_BETA>
__device__ void gemm_reduced_impl_ct(uint32_t rank, uint32_t size,
                                      T alpha, T *A, T *B, T beta, T *C)
{
    constexpr uint32_t C_cols = TRANSPOSE_B ? N : K;
    constexpr uint32_t maxel  = M * C_cols;

    if (size < 32u) {
        // Sub-warp fallback: each thread owns whole outputs (strided by size).
        // Reproduce the 32-lane tree in registers so the rounding matches the
        // full-warp path exactly — invariant across the 32-thread boundary.
        for (uint32_t el = rank; el < maxel; el += size) {
            const uint32_t row = el % M, col = el / M;
            T p[32];
            #pragma unroll
            for (uint32_t v = 0; v < 32u; ++v) {
                T acc = static_cast<T>(0);
                for (uint32_t ind = v; ind < N; ind += 32u) {
                    T a = ROW_MAJOR_A ? A[row*N + ind] : A[ind*M + row];
                    T b;
                    if (TRANSPOSE_B) b = ROW_MAJOR_B ? B[col*N + ind] : B[ind*N + col];
                    else             b = ROW_MAJOR_B ? B[ind*K + col] : B[col*N + ind];
                    acc += a * b;
                }
                p[v] = acc;
            }
            T res = reduced_tree32<T>(p);
            const uint32_t cidx = ROW_MAJOR_C ? (row*C_cols + col) : (col*M + row);
            C[cidx] = HAS_BETA ? (alpha*res + beta*C[cidx]) : (alpha*res);
        }
        return;
    }

    // Full-warp path: G = size>>5 warp-groups, group `warp` owns outputs strided
    // by G; the 32 lanes split the contraction and combine via warp::reduce.
    const uint32_t n_warps = size >> 5;       // full warps only
    const uint32_t warp    = rank >> 5;
    const uint32_t lane    = rank & 31u;
    if (warp < n_warps) {
        for (uint32_t el = warp; el < maxel; el += n_warps) {
            const uint32_t row = el % M, col = el / M;
            T partial = static_cast<T>(0);
            for (uint32_t ind = lane; ind < N; ind += 32u) {
                T a = ROW_MAJOR_A ? A[row*N + ind] : A[ind*M + row];
                T b;
                if (TRANSPOSE_B) b = ROW_MAJOR_B ? B[col*N + ind] : B[ind*N + col];
                else             b = ROW_MAJOR_B ? B[ind*K + col] : B[col*N + ind];
                partial += a * b;
            }
            T res = warp::reduce<T>(partial);   // full mask: warp is full
            if (lane == 0) {
                const uint32_t cidx = ROW_MAJOR_C ? (row*C_cols + col) : (col*M + row);
                C[cidx] = HAS_BETA ? (alpha*res + beta*C[cidx]) : (alpha*res);
            }
        }
    }
    // trailing partial-warp threads (warp >= n_warps) idle
}

/**
 * @brief Contraction-parallel GEMM: `C = alpha * A * op(B) + beta * C`.
 *
 * Same math and layout as the compile-time `glass::gemm`, but parallelizes the
 * length-N contraction: one warp owns each output element and its 32 lanes
 * split the inner sum (combined with a single warp-shuffle reduce) instead of
 * one thread summing serially. A utilization win when the output count is
 * smaller than the block — see :doc:`../../user_guide/concepts/contraction_parallel`
 * and `glass::suggested_use_reduced`. Total MAC work is unchanged.
 *
 * Thread-count invariant: bit-identical at any block size (a trailing partial
 * warp idles; below 32 threads a register path reproduces the same rounding).
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (requires B square: N == K).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true) so callers can read C safely.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemm_reduced(T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR, true>(
        rank, size, alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Contraction-parallel GEMM with implicit `beta = 0`: `C = alpha * A * op(B)`.
 *
 * Overwrites C (the existing C is not read), avoiding the `beta * C` term.
 * Otherwise identical to the beta overload above.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (requires B square: N == K).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemm_reduced(T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR, false>(
        rank, size, alpha, A, B, static_cast<T>(0), C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-warp contraction-parallel GEMM ───────────────────────────────────
namespace warp {
    /**
     * @brief Single-warp contraction-parallel GEMM: `C = alpha * A * op(B) + beta * C`.
     *
     * One 32-lane warp computes the full product, parallelizing the contraction
     * across its lanes (warp-shuffle reduce per output). The warp-per-problem
     * analogue of the block `glass::gemm_reduced`; the caller must run a full
     * 32-lane warp. `C` must not alias `A`/`B`.
     *
     * @tparam T  Scalar type.
     * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
     * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (requires B square: N == K).
     * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true) so lanes can read C safely.
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices.
     * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
     * @param C      In/out result matrix.
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              bool TRANSPOSE_B = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
    __device__ void gemm_reduced(T alpha, T *A, T *B, T beta, T *C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR, true>(
            lane, 32u, alpha, A, B, beta, C);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }

    /**
     * @brief Single-warp contraction-parallel GEMM with implicit `beta = 0`: `C = alpha * A * op(B)`.
     *
     * Overwrites C (the existing C is not read). Otherwise identical to the beta
     * overload above; the caller must run a full 32-lane warp.
     *
     * @tparam T  Scalar type.
     * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
     * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (requires B square: N == K).
     * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices.
     * @param C      Output result matrix (overwritten).
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              bool TRANSPOSE_B = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
    __device__ void gemm_reduced(T alpha, T *A, T *B, T *C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        gemm_reduced_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR, false>(
            lane, 32u, alpha, A, B, static_cast<T>(0), C);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}
