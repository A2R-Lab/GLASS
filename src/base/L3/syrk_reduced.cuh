#pragma once
#include <cstdint>

// ─── contraction-dimension-parallel symmetric rank-k update ──────────────────
//
// C = alpha*A*op(A) + beta*C (C symmetric), with the contracted dimension split
// across a warp's lanes (one warp owns each output) and the lower triangle
// computed then mirrored — the SYRK sibling of glass::gemm_reduced. Total MAC
// work is unchanged; the win is thread utilization when the output count is
// small. Column-major A. Requires L1/reduce.cuh (reduced_tree32 + warp::reduce),
// included first by glass.cuh.
//
// Within a surface the result is thread-count invariant (bit-identical across
// block sizes). Across surfaces (block/warp/cgrps) it agrees to floating-point
// tolerance — the compiler may fuse the `a*a` accumulation's FMA differently per
// instantiation (both correct).

namespace detail {
    // C = alpha * A op(A) (+ beta*C), A is ROWS×COLS column-major.
    // TRANSPOSE=false (A·Aᵀ): C is ROWS×ROWS, contract c over COLS, A[i + c*ROWS].
    // TRANSPOSE=true  (Aᵀ·A): C is COLS×COLS, contract c over ROWS, A[c + i*ROWS].
    // Lower triangle computed and mirrored (output is full symmetric).
    template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANSPOSE, bool HAS_BETA>
    __device__ void syrk_reduced_impl(uint32_t rank, uint32_t size, T alpha, const T* A, T beta, T* C)
    {
        constexpr uint32_t OUT  = TRANSPOSE ? COLS : ROWS;
        constexpr uint32_t CDIM = TRANSPOSE ? ROWS : COLS;
        constexpr uint32_t maxel = OUT * OUT;

        if (size < 32u) {
            for (uint32_t el = rank; el < maxel; el += size) {
                const uint32_t i = el % OUT, j = el / OUT;
                if (i < j) continue;                 // lower triangle owns the mirror
                T p[32];
                #pragma unroll
                for (uint32_t vlane = 0; vlane < 32u; ++vlane) {
                    T acc = static_cast<T>(0);
                    for (uint32_t c = vlane; c < CDIM; c += 32u) {
                        const T ai = TRANSPOSE ? A[c + i*ROWS] : A[i + c*ROWS];
                        const T aj = TRANSPOSE ? A[c + j*ROWS] : A[j + c*ROWS];
                        acc += ai * aj;
                    }
                    p[vlane] = acc;
                }
                const T res = reduced_tree32<T>(p);
                const uint32_t idx = i + j*OUT;
                C[idx] = HAS_BETA ? (alpha*res + beta*C[idx]) : (alpha*res);
                if (i != j) {
                    const uint32_t m = j + i*OUT;
                    C[m] = HAS_BETA ? (alpha*res + beta*C[m]) : (alpha*res);
                }
            }
            return;
        }
        const uint32_t n_warps = size >> 5;
        const uint32_t warp = rank >> 5, lane = rank & 31u;
        if (warp < n_warps) {
            for (uint32_t el = warp; el < maxel; el += n_warps) {
                const uint32_t i = el % OUT, j = el / OUT;
                if (i < j) continue;
                T partial = static_cast<T>(0);
                for (uint32_t c = lane; c < CDIM; c += 32u) {
                    const T ai = TRANSPOSE ? A[c + i*ROWS] : A[i + c*ROWS];
                    const T aj = TRANSPOSE ? A[c + j*ROWS] : A[j + c*ROWS];
                    partial += ai * aj;
                }
                const T res = warp::reduce<T>(partial);
                if (lane == 0) {
                    const uint32_t idx = i + j*OUT;
                    C[idx] = HAS_BETA ? (alpha*res + beta*C[idx]) : (alpha*res);
                    if (i != j) {
                        const uint32_t m = j + i*OUT;
                        C[m] = HAS_BETA ? (alpha*res + beta*C[m]) : (alpha*res);
                    }
                }
            }
        }
    }
} // namespace detail

/**
 * @brief Contraction-parallel symmetric rank-k update: `C = alpha * A·op(A) + beta * C`.
 *
 * Compile-time-size SYRK that parallelizes the contracted dimension across a
 * warp's lanes (one warp per output) and computes the lower triangle, mirroring
 * it to produce a full symmetric `C` — the SYRK analogue of `glass::gemm_reduced`.
 * Column-major. Thread-count invariant at any block size.
 *
 * @tparam T  Scalar type.
 * @tparam ROWS,COLS  A is ROWS x COLS (column-major).
 * @tparam TRANSPOSE  If true, `C = AᵀA` (COLS x COLS); else `C = AAᵀ` (ROWS x ROWS).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar on the product.
 * @param A      Input matrix (ROWS x COLS, column-major).
 * @param beta   Scalar on the existing C (read; caller must initialize it).
 * @param C      In/out symmetric result (full storage; OUT x OUT, OUT = TRANSPOSE?COLS:ROWS).
 */
template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void syrk_reduced(T alpha, const T* A, T beta, T* C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::syrk_reduced_impl<T, ROWS, COLS, TRANSPOSE, true>(rank, size, alpha, A, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Contraction-parallel SYRK with implicit `beta = 0`: `C = alpha * A·op(A)`.
 *
 * Overwrites C (not read). Otherwise identical to the beta overload.
 *
 * @tparam T,ROWS,COLS,TRANSPOSE,TRAILING_SYNC  See the beta overload.
 * @param alpha,A  See the beta overload.
 * @param C  Output (overwritten; full symmetric).
 */
template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void syrk_reduced(T alpha, const T* A, T* C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::syrk_reduced_impl<T, ROWS, COLS, TRANSPOSE, false>(rank, size, alpha, A, static_cast<T>(0), C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace warp {
    /**
     * @brief Single-warp contraction-parallel SYRK: `C = alpha * A·op(A) + beta * C`.
     *
     * Warp-per-problem analogue of `glass::syrk_reduced`; one full 32-lane warp.
     *
     * @tparam T,ROWS,COLS,TRANSPOSE  See glass::syrk_reduced.
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param alpha,A,beta,C  See glass::syrk_reduced.
     */
    template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
    __device__ void syrk_reduced(T alpha, const T* A, T beta, T* C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::syrk_reduced_impl<T, ROWS, COLS, TRANSPOSE, true>(lane, 32u, alpha, A, beta, C);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }

    /**
     * @brief Single-warp contraction-parallel SYRK, implicit `beta = 0`: `C = alpha * A·op(A)`.
     *
     * @tparam T,ROWS,COLS,TRANSPOSE,TRAILING_SYNC  See the beta overload.
     * @param alpha,A,C  See the beta overload.
     */
    template <typename T, uint32_t ROWS, uint32_t COLS, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
    __device__ void syrk_reduced(T alpha, const T* A, T* C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::syrk_reduced_impl<T, ROWS, COLS, TRANSPOSE, false>(lane, 32u, alpha, A, static_cast<T>(0), C);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}
