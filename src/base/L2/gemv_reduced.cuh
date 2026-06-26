#pragma once
#include <cstdint>

// ─── contraction-dimension-parallel GEMV ─────────────────────────────────────
//
// y = alpha*op(A)*x + beta*y, with the contraction parallelized across a warp's
// lanes (one warp owns each output element) instead of one thread summing it
// serially — the L2 sibling of glass::gemm_reduced. A utilization win when the
// output count is small relative to the block; total MAC work is unchanged. See
// gemm_reduced.cuh for the engine + the honest win-condition. Column-major A.
//
// Requires gemm_reduced.cuh (reduced_tree32 + glass::warp::reduce), included
// first by glass.cuh.

namespace detail {
    // y[out] = alpha * sum_c A(.)·x[c] + (HAS_BETA ? beta*y[out] : 0), compile-time
    // M×N A. TRANSPOSE=false: out=row i in [0,M), contract c over N, A[i + c*M].
    //        TRANSPOSE=true : out=col j in [0,N), contract c over M, A[c + j*M].
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE, bool HAS_BETA>
    __device__ void gemv_reduced_impl(uint32_t rank, uint32_t size,
                                      T alpha, const T* A, const T* x, T beta, T* y)
    {
        constexpr uint32_t n_out = TRANSPOSE ? N : M;
        constexpr uint32_t CDIM  = TRANSPOSE ? M : N;
        if (size < 32u) {
            for (uint32_t o = rank; o < n_out; o += size) {
                T p[32];
                #pragma unroll
                for (uint32_t vlane = 0; vlane < 32u; ++vlane) {
                    T acc = static_cast<T>(0);
                    for (uint32_t c = vlane; c < CDIM; c += 32u)
                        acc += (TRANSPOSE ? A[c + o*M] : A[o + c*M]) * x[c];
                    p[vlane] = acc;
                }
                const T res = reduced_tree32<T>(p);
                y[o] = HAS_BETA ? (alpha*res + beta*y[o]) : (alpha*res);
            }
            return;
        }
        const uint32_t n_warps = size >> 5;
        const uint32_t warp = rank >> 5, lane = rank & 31u;
        if (warp < n_warps) {
            for (uint32_t o = warp; o < n_out; o += n_warps) {
                T partial = static_cast<T>(0);
                for (uint32_t c = lane; c < CDIM; c += 32u)
                    partial += (TRANSPOSE ? A[c + o*M] : A[o + c*M]) * x[c];
                const T res = warp::reduce<T>(partial);
                if (lane == 0) y[o] = HAS_BETA ? (alpha*res + beta*y[o]) : (alpha*res);
            }
        }
    }
} // namespace detail

/**
 * @brief Contraction-parallel GEMV: `y = alpha * op(A) * x + beta * y`.
 *
 * Compile-time-size matrix-vector product that parallelizes the contraction
 * across a warp's lanes (one warp per output element) rather than one thread
 * summing serially — the L2 analogue of `glass::gemm_reduced`. Column-major `A`.
 * Thread-count invariant at any block size.
 *
 * @tparam T  Scalar type.
 * @tparam M,N  A is M x N (column-major).
 * @tparam TRANSPOSE  If true, computes `Aᵀ x` (output length N, contract over M); else `A x` (length M, contract over N).
 * @tparam TRAILING_SYNC  Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar on the product.
 * @param A      Input matrix (M x N, column-major).
 * @param x      Input vector (length N if !TRANSPOSE else M).
 * @param beta   Scalar on the existing y (read; caller must initialize it).
 * @param y      In/out result (length M if !TRANSPOSE else N).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void gemv_reduced(T alpha, const T* A, const T* x, T beta, T* y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::gemv_reduced_impl<T, M, N, TRANSPOSE, true>(rank, size, alpha, A, x, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Contraction-parallel GEMV with implicit `beta = 0`: `y = alpha * op(A) * x`.
 *
 * Overwrites y (not read). Otherwise identical to the beta overload.
 *
 * @tparam T,M,N,TRANSPOSE,TRAILING_SYNC  See the beta overload.
 * @param alpha,A,x  See the beta overload.
 * @param y  Output (overwritten).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
__device__ void gemv_reduced(T alpha, const T* A, const T* x, T* y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    detail::gemv_reduced_impl<T, M, N, TRANSPOSE, false>(rank, size, alpha, A, x, static_cast<T>(0), y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace warp {
    /**
     * @brief Single-warp contraction-parallel GEMV: `y = alpha * op(A) * x + beta * y`.
     *
     * Warp-per-problem analogue of `glass::gemv_reduced`; one full 32-lane warp.
     *
     * @tparam T,M,N,TRANSPOSE  See glass::gemv_reduced.
     * @tparam TRAILING_SYNC  Emit a trailing `__syncwarp()` (default true).
     * @param alpha,A,x,beta,y  See glass::gemv_reduced.
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
    __device__ void gemv_reduced(T alpha, const T* A, const T* x, T beta, T* y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::gemv_reduced_impl<T, M, N, TRANSPOSE, true>(lane, 32u, alpha, A, x, beta, y);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }

    /**
     * @brief Single-warp contraction-parallel GEMV, implicit `beta = 0`: `y = alpha * op(A) * x`.
     *
     * @tparam T,M,N,TRANSPOSE,TRAILING_SYNC  See the beta overload.
     * @param alpha,A,x,y  See the beta overload.
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool TRAILING_SYNC = true>
    __device__ void gemv_reduced(T alpha, const T* A, const T* x, T* y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31u;
        detail::gemv_reduced_impl<T, M, N, TRANSPOSE, false>(lane, 32u, alpha, A, x, static_cast<T>(0), y);
        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}
