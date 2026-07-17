#pragma once
#include <cstdint>

// core impl: explicit rank/size + layout flags
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n,
                           T alpha, const T *A, const T *x, T beta, T *y)
{
    if (TRANSPOSE) {
        for (uint32_t row = rank; row < n; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < m; col++) {
                T a = ROW_MAJOR_A ? A[col*n + row] : A[col + row*m];
                res += a * x[col];
            }
            y[row] = beta_blend(alpha*res, beta, y[row]);
        }
    } else {
        for (uint32_t row = rank; row < m; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < n; col++) {
                T a = ROW_MAJOR_A ? A[row*n + col] : A[row + col*m];
                res += a * x[col];
            }
            y[row] = beta_blend(alpha*res, beta, y[row]);
        }
    }
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n,
                           T alpha, const T *A, const T *x, T *y)
{
    if (TRANSPOSE) {
        for (uint32_t row = rank; row < n; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < m; col++) {
                T a = ROW_MAJOR_A ? A[col*n + row] : A[col + row*m];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    } else {
        for (uint32_t row = rank; row < m; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < n; col++) {
                T a = ROW_MAJOR_A ? A[row*n + col] : A[row + col*m];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    }
}

// ─── runtime variants ─────────────────────────────────────────────────────────

/**
 * @brief Matrix-vector product: `y = alpha * A * x + beta * y` (GEMV).
 *
 * Threads are distributed over the output rows of the `m×n` matrix `A`. Set
 * `TRANSPOSE=true` to compute `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`
 * (`A` is column-major by default). NumPy equivalent: `y = alpha*A@x + beta*y`
 * (or `alpha*A.T@x + beta*y` when transposed).
 *
 * Unlike `gemm` — where a row-major operand is just a transpose, so the only
 * layout flag is `ROW_MAJOR_C` — GEMV keeps a per-matrix `ROW_MAJOR` flag:
 * `TRANSPOSE` already selects the mathematical operation (`A·x` vs `Aᵀ·x`), so it
 * cannot also stand in for the storage order. `TRANSPOSE` and `ROW_MAJOR` are
 * therefore independent. (This flag fully subsumes the former `gemv_ex`, which
 * was just `gemv` with the defaults removed and has been deleted.)
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param m      Number of rows of `A`.
 * @param n      Number of columns of `A`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `m*n` elements.
 * @param x      Input vector (length `n`, or `m` when transposed).
 * @param beta   Scalar multiplier on the prior `y`.
 * @param y      In/out vector (length `m`, or `n` when transposed).
 */
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, const T *A, const T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Matrix-vector product: `y = alpha * A * x` (GEMV), no-beta overload.
 *
 * Same as the full GEMV but overwrites `y` (no `beta * y` term). Set
 * `TRANSPOSE=true` for `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`. NumPy
 * equivalent: `y = alpha*A@x` (or `alpha*A.T@x` when transposed).
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param m      Number of rows of `A`.
 * @param n      Number of columns of `A`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `m*n` elements.
 * @param x      Input vector (length `n`, or `m` when transposed).
 * @param y      Output vector (length `m`, or `n` when transposed).
 */
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, const T *A, const T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// compile-time impl: M, N as template params so inner col-loop is fully unrolled
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *A, const T *x, T beta, T *y)
{
    if (TRANSPOSE) {
        for (uint32_t row = rank; row < N; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < M; col++) {
                T a = ROW_MAJOR_A ? A[col*N + row] : A[col + row*M];
                res += a * x[col];
            }
            y[row] = beta_blend(alpha*res, beta, y[row]);
        }
    } else {
        for (uint32_t row = rank; row < M; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < N; col++) {
                T a = ROW_MAJOR_A ? A[row*N + col] : A[row + col*M];
                res += a * x[col];
            }
            y[row] = beta_blend(alpha*res, beta, y[row]);
        }
    }
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *A, const T *x, T *y)
{
    if (TRANSPOSE) {
        for (uint32_t row = rank; row < N; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < M; col++) {
                T a = ROW_MAJOR_A ? A[col*N + row] : A[col + row*M];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    } else {
        for (uint32_t row = rank; row < M; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < N; col++) {
                T a = ROW_MAJOR_A ? A[row*N + col] : A[row + col*M];
                res += a * x[col];
            }
            y[row] = alpha*res;
        }
    }
}

// ─── compile-time size variants ───────────────────────────────────────────────

/**
 * @brief Matrix-vector product: `y = alpha * A * x + beta * y` (GEMV), compile-time size.
 *
 * Compile-time-`M`,`N` overload; the inner column loop is fully unrolled. Set
 * `TRANSPOSE=true` for `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`. NumPy
 * equivalent: `y = alpha*A@x + beta*y` (or `alpha*A.T@x + beta*y`).
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam M          Number of rows of `A` (compile-time constant).
 * @tparam N          Number of columns of `A` (compile-time constant).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `M*N` elements.
 * @param x      Input vector (length `N`, or `M` when transposed).
 * @param beta   Scalar multiplier on the prior `y`.
 * @param y      In/out vector (length `M`, or `N` when transposed).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(T alpha, const T *A, const T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, x, beta, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Matrix-vector product: `y = alpha * A * x` (GEMV), compile-time size, no-beta overload.
 *
 * Compile-time-`M`,`N` overload that overwrites `y` (no `beta * y` term). NumPy
 * equivalent: `y = alpha*A@x` (or `alpha*A.T@x` when transposed).
 *
 * @tparam T          Scalar type (e.g. `float`, `double`).
 * @tparam M          Number of rows of `A` (compile-time constant).
 * @tparam N          Number of columns of `A` (compile-time constant).
 * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
 * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `M*N` elements.
 * @param x      Input vector (length `N`, or `M` when transposed).
 * @param y      Output vector (length `M`, or `N` when transposed).
 */
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false, bool TRAILING_SYNC = true>
__device__ void gemv(T alpha, const T *A, const T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, x, y);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

namespace thread {
    // Single-thread GEMV: one THREAD computes the whole matvec, walking the output
    // rows serially. Reuses the block impl `gemv_impl_ct(0u, 1u, …)` exactly as
    // `warp::gemv` reuses it with `(lane, 32u)`. No shared scratch, no barriers.
    // For thread-per-problem kernels packing 32 independent low-DOF matvecs into
    // one warp.
    //
    // LAYOUT NOTE: `ROW_MAJOR` is a correctness flag, not a free choice. The
    // block/warp tiers map lane rank onto the ROW index, so column-major
    // (`A[row + col*M]`, row fastest-varying) makes lane-adjacent reads
    // address-adjacent. A single thread has no lane axis at all, so within-matrix
    // layout costs nothing here — but if `A` is in GLOBAL memory laid out
    // per-problem-contiguous, then it is the PROBLEM index that strides across
    // lanes, and neither flag helps: consecutive threads land N*N elements apart
    // and every access serializes. Keep `A` thread-local (register-resident) —
    // CUDA local memory is hardware-interleaved across lanes, so it coalesces for
    // free — or interleave the batch yourself so the problem index is
    // fastest-varying.

    /**
     * @brief Matrix-vector product on one thread: `y = alpha * A * x + beta * y` (GEMV), compile-time size.
     *
     * One thread computes the matvec, walking the output rows of the `M×N` matrix
     * `A` serially (each row an independent inner product). Set `TRANSPOSE=true`
     * for `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`. No shared scratch, no
     * barriers, no `threadIdx` read; operands may be thread-local register arrays.
     * `y` is read only when `beta != 0` (BLAS semantics: `beta == 0` treats `y` as
     * write-only). NumPy equivalent: `y = alpha*A@x + beta*y`.
     *
     * @tparam T          Scalar type (e.g. `float`, `double`).
     * @tparam M          Number of rows of `A` (compile-time constant).
     * @tparam N          Number of columns of `A` (compile-time constant).
     * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
     * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A      Input matrix of `M*N` elements.
     * @param x      Input vector (length `N`, or `M` when transposed).
     * @param beta   Scalar multiplier on the prior `y`.
     * @param y      In/out vector (length `M`, or `N` when transposed).
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void gemv(T alpha, const T *A, const T *x, T beta, T *y)
    {
        gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(0u, 1u, alpha, A, x, beta, y);
    }

    /**
     * @brief Matrix-vector product on one thread: `y = alpha * A * x` (GEMV), compile-time size, no-beta overload.
     *
     * Overwrites `y` (no `beta * y` term). NumPy equivalent: `y = alpha*A@x`
     * (or `alpha*A.T@x` when transposed).
     *
     * @tparam T          Scalar type (e.g. `float`, `double`).
     * @tparam M          Number of rows of `A` (compile-time constant).
     * @tparam N          Number of columns of `A` (compile-time constant).
     * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
     * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A      Input matrix of `M*N` elements.
     * @param x      Input vector (length `N`, or `M` when transposed).
     * @param y      Output vector (length `M`, or `N` when transposed).
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void gemv(T alpha, const T *A, const T *x, T *y)
    {
        gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(0u, 1u, alpha, A, x, y);
    }
}

namespace warp {
    // Single-warp GEMV: one 32-lane warp computes the matvec, lanes striding over
    // the output rows (lane i owns output rows i, i+32, …). Each lane's row is an
    // independent inner product — no cross-lane communication, no shared scratch,
    // no `__syncthreads`. Reuses the block impl `gemv_impl_ct(lane, 32u, …)` exactly
    // as `warp::gemm` reuses `gemm_impl_ct`. For warp-per-problem kernels packing
    // many small matvecs into one block via independent warps. Full 32 lanes
    // required.

    /**
     * @brief Matrix-vector product within one warp: `y = alpha * A * x + beta * y` (GEMV), single-warp, compile-time size.
     *
     * One 32-lane warp computes the matvec with lanes striding over the output rows
     * of the `M×N` matrix `A` (each row an independent inner product). Set
     * `TRANSPOSE=true` for `Aᵀ * x` and `ROW_MAJOR=true` for row-major `A`. No shared
     * scratch, no `__syncthreads`; independent warps may run distinct problems
     * concurrently. Full 32 lanes required. `y` is read only when `beta != 0`
     * (BLAS semantics: `beta == 0` treats `y` as write-only). NumPy
     * equivalent: `y = alpha*A@x + beta*y` (or `alpha*A.T@x + beta*y` when transposed).
     *
     * @tparam T          Scalar type (e.g. `float`, `double`).
     * @tparam M          Number of rows of `A` (compile-time constant).
     * @tparam N          Number of columns of `A` (compile-time constant).
     * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
     * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A      Input matrix of `M*N` elements.
     * @param x      Input vector (length `N`, or `M` when transposed).
     * @param beta   Scalar multiplier on the prior `y`.
     * @param y      In/out vector (length `M`, or `N` when transposed).
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void gemv(T alpha, const T *A, const T *x, T beta, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(lane, 32u, alpha, A, x, beta, y);
        __syncwarp();
    }

    /**
     * @brief Matrix-vector product within one warp: `y = alpha * A * x` (GEMV), single-warp, compile-time size, implicit beta = 0.
     *
     * Overwrites `y` (no `beta * y` term — `y` is never read, so it is safe to write
     * into cold/uninitialized scratch). Otherwise identical to the beta overload
     * above. No shared scratch, no `__syncthreads`. Full 32 lanes required. NumPy
     * equivalent: `y = alpha*A@x` (or `alpha*A.T@x` when transposed).
     *
     * @tparam T          Scalar type (e.g. `float`, `double`).
     * @tparam M          Number of rows of `A` (compile-time constant).
     * @tparam N          Number of columns of `A` (compile-time constant).
     * @tparam TRANSPOSE  When true, multiply by `Aᵀ` instead of `A` (default false).
     * @tparam ROW_MAJOR  When true, `A` is stored row-major (default false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A      Input matrix of `M*N` elements.
     * @param x      Input vector (length `N`, or `M` when transposed).
     * @param y      Output vector (length `M`, or `N` when transposed; overwritten).
     */
    template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__ void gemv(T alpha, const T *A, const T *x, T *y)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(lane, 32u, alpha, A, x, y);
        __syncwarp();
    }
}
