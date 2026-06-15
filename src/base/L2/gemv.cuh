#pragma once
#include <cstdint>

// core impl: explicit rank/size + layout flags
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n,
                           T alpha, T *A, T *x, T beta, T *y)
{
    if (TRANSPOSE) {
        for (uint32_t row = rank; row < n; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < m; col++) {
                T a = ROW_MAJOR_A ? A[col*n + row] : A[col + row*m];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    } else {
        for (uint32_t row = rank; row < m; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < n; col++) {
                T a = ROW_MAJOR_A ? A[row*n + col] : A[row + col*m];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    }
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n,
                           T alpha, T *A, T *x, T *y)
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
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, beta, y);
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
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, y);
}

/**
 * @brief Matrix-vector product with explicit layout control: `y = alpha * A * x + beta * y` (GEMV).
 *
 * Like `gemv` but exposes the per-matrix layout flag explicitly (no defaults),
 * for callers that want full control over `TRANSPOSE` and the storage order of
 * `A`. NumPy equivalent: `y = alpha*A@x + beta*y` (or `alpha*A.T@x + beta*y`).
 *
 * @tparam T           Scalar type (e.g. `float`, `double`).
 * @tparam TRANSPOSE   When true, multiply by `Aᵀ` instead of `A`.
 * @tparam ROW_MAJOR_A When true, `A` is stored row-major.
 * @param m      Number of rows of `A`.
 * @param n      Number of columns of `A`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `m*n` elements.
 * @param x      Input vector (length `n`, or `m` when transposed).
 * @param beta   Scalar multiplier on the prior `y`.
 * @param y      In/out vector (length `m`, or `n` when transposed).
 */
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(rank, size, m, n, alpha, A, x, beta, y);
}

/**
 * @brief Matrix-vector product with explicit layout control: `y = alpha * A * x` (GEMV), no-beta overload.
 *
 * Like the full `gemv_ex` but overwrites `y` (no `beta * y` term). NumPy
 * equivalent: `y = alpha*A@x` (or `alpha*A.T@x` when transposed).
 *
 * @tparam T           Scalar type (e.g. `float`, `double`).
 * @tparam TRANSPOSE   When true, multiply by `Aᵀ` instead of `A`.
 * @tparam ROW_MAJOR_A When true, `A` is stored row-major.
 * @param m      Number of rows of `A`.
 * @param n      Number of columns of `A`.
 * @param alpha  Scalar multiplier on the product.
 * @param A      Input matrix of `m*n` elements.
 * @param x      Input vector (length `n`, or `m` when transposed).
 * @param y      Output vector (length `m`, or `n` when transposed).
 */
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(rank, size, m, n, alpha, A, x, y);
}

// compile-time impl: M, N as template params so inner col-loop is fully unrolled
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, T *A, T *x, T beta, T *y)
{
    if (TRANSPOSE) {
        for (uint32_t row = rank; row < N; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < M; col++) {
                T a = ROW_MAJOR_A ? A[col*N + row] : A[col + row*M];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    } else {
        for (uint32_t row = rank; row < M; row += size) {
            T res = static_cast<T>(0);
            for (uint32_t col = 0; col < N; col++) {
                T a = ROW_MAJOR_A ? A[row*N + col] : A[row + col*M];
                res += a * x[col];
            }
            y[row] = alpha*res + beta*y[row];
        }
    }
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, T *A, T *x, T *y)
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
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, x, beta, y);
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
template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl_ct<T, M, N, TRANSPOSE, ROW_MAJOR>(rank, size, alpha, A, x, y);
}
