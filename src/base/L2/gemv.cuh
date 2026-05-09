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

template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, beta, y);
}

template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, m, n, alpha, A, x, y);
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(rank, size, m, n, alpha, A, x, beta, y);
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(rank, size, m, n, alpha, A, x, y);
}

// ─── compile-time size variants ───────────────────────────────────────────────

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T beta, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, M, N, alpha, A, x, beta, y);
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T *y)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(rank, size, M, N, alpha, A, x, y);
}
