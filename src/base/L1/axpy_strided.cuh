#pragma once
#include <cstdint>

/**
 * @file axpy_strided.cuh
 * @brief Row-strided AXPY: `Y[r + c*Y_RS] += alpha * X[r + c*X_RS]` over an M×N block.
 *
 * The AXPY analogue of `gemv_strided` — adds an `M×N` (column-major) block of
 * `X` into a same-shaped block of `Y` when the two live inside wider buffers with
 * different leading dimensions (`X_RS`, `Y_RS`). PDDP packs a 14×14 update into a
 * 21-lead buffer top-left (`Y_RS=21`, `X_RS=14`). Each `(r,c)` element is written
 * by exactly one thread/lane, so there is no race and the op is trivially
 * thread-count invariant. Block + `warp::`.
 */

/**
 * @brief Row-strided AXPY `Y[r + c*Y_RS] += alpha * X[r + c*X_RS]` over an M×N block.
 *
 * Column-major; `X` is addressed at leading dimension `X_RS` (default `M`), `Y` at
 * `Y_RS`. When `X_RS == M` and `Y_RS == M` this is a plain `glass::axpy` over the
 * `M*N` contiguous elements. NumPy: `Y[:M,:N] += alpha * X[:M,:N]` (col-major lds).
 *
 * @tparam T             Scalar type (e.g. `float`, `double`).
 * @tparam M             Rows of the block.
 * @tparam N             Columns of the block.
 * @tparam Y_RS          Column-major leading dimension of `Y`.
 * @tparam X_RS          Column-major leading dimension of `X` (default `M`).
 * @tparam TRAILING_SYNC Emit a trailing `__syncthreads()` (default true).
 * @param alpha  Scalar multiplier on `X`.
 * @param X      Input block, addressed at `X[r + c*X_RS]` (read-only).
 * @param Y      In/out block, addressed at `Y[r + c*Y_RS]`.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t Y_RS, uint32_t X_RS = M, bool TRAILING_SYNC = true>
__device__ void axpy_strided(T alpha, const T* X, T* Y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t k = rank; k < M * N; k += size) {
        uint32_t r = k % M, c = k / M;
        Y[r + c * Y_RS] += alpha * X[r + c * X_RS];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace warp {
    /**
     * @brief Row-strided AXPY within one warp: `Y[r + c*Y_RS] += alpha * X[r + c*X_RS]`.
     *
     * Single-warp form of `axpy_strided`: one 32-lane warp strides over the
     * `M*N` block elements. Each element written once; no inter-lane comms, no
     * shared scratch. `TRAILING_SYNC` gates a closing `__syncwarp()`. Full 32 lanes
     * required; independent warps may run distinct problems concurrently.
     *
     * @tparam T             Scalar type.
     * @tparam M,N           Block shape.
     * @tparam Y_RS          Leading dimension of `Y`.
     * @tparam X_RS          Leading dimension of `X` (default `M`).
     * @tparam TRAILING_SYNC Emit a trailing `__syncwarp()` (default true).
     * @param alpha  Scalar multiplier on `X`.
     * @param X      Input block, addressed at `X[r + c*X_RS]` (read-only).
     * @param Y      In/out block, addressed at `Y[r + c*Y_RS]`.
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t Y_RS, uint32_t X_RS = M, bool TRAILING_SYNC = true>
    __device__ void axpy_strided(T alpha, const T* X, T* Y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        for (uint32_t k = lane; k < M * N; k += 32) {
            uint32_t r = k % M, c = k / M;
            Y[r + c * Y_RS] += alpha * X[r + c * X_RS];
        }
        if constexpr (TRAILING_SYNC) __syncwarp();
    }
}
