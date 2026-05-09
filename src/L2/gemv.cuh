#pragma once

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// ─── gemv_impl: core implementation with per-matrix storage-order flag ─────────
// TRANSPOSE=false: y = alpha * A(m×n) * x(n) + beta * y(m)
// TRANSPOSE=true:  y = alpha * A(m×n)^T * x(m) + beta * y(n)
// ROW_MAJOR_A=false: A stored column-major (A[row + col*m])  [default, cuBLAS style]
// ROW_MAJOR_A=true:  A stored row-major   (A[row*n + col])   [C array style]
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__
void gemv_impl(std::uint32_t m,
               std::uint32_t n,
               T alpha,
               T *A,
               T *x,
               T beta,
               T *y,
               cgrps::thread_group g = cgrps::this_thread_block())
{
    if (TRANSPOSE) {
        // result y has length n; sum over m columns of A^T
        for (std::uint32_t row = g.thread_rank(); row < n; row += g.size()) {
            T res = static_cast<T>(0);
            for (std::uint32_t col = 0; col < m; col++) {
                // A^T[row][col] = A[col][row]
                T a = ROW_MAJOR_A ? A[col * n + row] : A[col + row * m];
                res += a * x[col];
            }
            y[row] = alpha * res + beta * y[row];
        }
    } else {
        // result y has length m; sum over n columns
        for (std::uint32_t row = g.thread_rank(); row < m; row += g.size()) {
            T res = static_cast<T>(0);
            for (std::uint32_t col = 0; col < n; col++) {
                T a = ROW_MAJOR_A ? A[row * n + col] : A[row + col * m];
                res += a * x[col];
            }
            y[row] = alpha * res + beta * y[row];
        }
    }
}

// no-beta overload
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__
void gemv_impl(std::uint32_t m,
               std::uint32_t n,
               T alpha,
               T *A,
               T *x,
               T *y,
               cgrps::thread_group g = cgrps::this_thread_block())
{
    if (TRANSPOSE) {
        for (std::uint32_t row = g.thread_rank(); row < n; row += g.size()) {
            T res = static_cast<T>(0);
            for (std::uint32_t col = 0; col < m; col++) {
                T a = ROW_MAJOR_A ? A[col * n + row] : A[col + row * m];
                res += a * x[col];
            }
            y[row] = alpha * res;
        }
    } else {
        for (std::uint32_t row = g.thread_rank(); row < m; row += g.size()) {
            T res = static_cast<T>(0);
            for (std::uint32_t col = 0; col < n; col++) {
                T a = ROW_MAJOR_A ? A[row * n + col] : A[row + col * m];
                res += a * x[col];
            }
            y[row] = alpha * res;
        }
    }
}

// ─── gemv: convenience single-flag wrapper (ROW_MAJOR applies to A) ──────────
// Backward-compatible default: TRANSPOSE=false, ROW_MAJOR=false (column-major)
template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__
void gemv(std::uint32_t m,
          std::uint32_t n,
          T alpha,
          T *A,
          T *x,
          T beta,
          T *y,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(m, n, alpha, A, x, beta, y, g);
}

template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__
void gemv(std::uint32_t m,
          std::uint32_t n,
          T alpha,
          T *A,
          T *x,
          T *y,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(m, n, alpha, A, x, y, g);
}

// ─── gemv_ex: explicit per-matrix storage-order control ─────────────────────
// For gemv there is only one matrix A, so ROW_MAJOR_A is the only relevant flag.
// gemv_ex is provided for API symmetry with gemm_ex.
template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__
void gemv_ex(std::uint32_t m,
             std::uint32_t n,
             T alpha,
             T *A,
             T *x,
             T beta,
             T *y,
             cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(m, n, alpha, A, x, beta, y, g);
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__
void gemv_ex(std::uint32_t m,
             std::uint32_t n,
             T alpha,
             T *A,
             T *x,
             T *y,
             cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(m, n, alpha, A, x, y, g);
}

// === glass::simple variants ===
namespace simple {

    // ─── gemv_impl ────────────────────────────────────────────────────────────
    template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
    __device__
    void gemv_impl(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        if (TRANSPOSE) {
            for (uint32_t row = rank; row < n; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < m; col++) {
                    T a = ROW_MAJOR_A ? A[col * n + row] : A[col + row * m];
                    res += a * x[col];
                }
                y[row] = alpha * res + beta * y[row];
            }
        } else {
            for (uint32_t row = rank; row < m; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < n; col++) {
                    T a = ROW_MAJOR_A ? A[row * n + col] : A[row + col * m];
                    res += a * x[col];
                }
                y[row] = alpha * res + beta * y[row];
            }
        }
    }

    template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
    __device__
    void gemv_impl(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        if (TRANSPOSE) {
            for (uint32_t row = rank; row < n; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < m; col++) {
                    T a = ROW_MAJOR_A ? A[col * n + row] : A[col + row * m];
                    res += a * x[col];
                }
                y[row] = alpha * res;
            }
        } else {
            for (uint32_t row = rank; row < m; row += size) {
                T res = static_cast<T>(0);
                for (uint32_t col = 0; col < n; col++) {
                    T a = ROW_MAJOR_A ? A[row * n + col] : A[row + col * m];
                    res += a * x[col];
                }
                y[row] = alpha * res;
            }
        }
    }

    // ─── gemv: single-flag convenience (ROW_MAJOR applies to A) ──────────────
    template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__
    void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
    {
        gemv_impl<T, TRANSPOSE, ROW_MAJOR>(m, n, alpha, A, x, beta, y);
    }

    template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
    __device__
    void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
    {
        gemv_impl<T, TRANSPOSE, ROW_MAJOR>(m, n, alpha, A, x, y);
    }

    // ─── gemv_ex: explicit per-matrix control ────────────────────────────────
    template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
    __device__
    void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
    {
        gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(m, n, alpha, A, x, beta, y);
    }

    template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
    __device__
    void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
    {
        gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(m, n, alpha, A, x, y);
    }
}
// ===
