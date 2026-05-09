#pragma once
#include <cstdint>
#include <cooperative_groups.h>
#include "../base/L2/gemv.cuh"
namespace cgrps = cooperative_groups;

// glass::cgrps::gemv — delegates to shared gemv_impl with g.thread_rank()/g.size()

template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), m, n, alpha, A, x, beta, y);
}

template <typename T, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), m, n, alpha, A, x, y);
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T beta, T *y,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(g.thread_rank(), g.size(), m, n, alpha, A, x, beta, y);
}

template <typename T, bool TRANSPOSE, bool ROW_MAJOR_A>
__device__ void gemv_ex(uint32_t m, uint32_t n, T alpha, T *A, T *x, T *y,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR_A>(g.thread_rank(), g.size(), m, n, alpha, A, x, y);
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T beta, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), M, N, alpha, A, x, beta, y);
}

template <typename T, uint32_t M, uint32_t N, bool TRANSPOSE = false, bool ROW_MAJOR = false>
__device__ void gemv(T alpha, T *A, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemv_impl<T, TRANSPOSE, ROW_MAJOR>(g.thread_rank(), g.size(), M, N, alpha, A, x, y);
}

// ger: A += alpha * x * y^T (column-major)
template <typename T>
__device__ void ger(uint32_t m, uint32_t n, T alpha, T *x, T *y, T *A,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t col = 0; col < n; col++) {
        T ay = alpha * y[col];
        for (uint32_t row = g.thread_rank(); row < m; row += g.size())
            A[row + col*m] += ay * x[row];
    }
}

template <typename T, uint32_t M, uint32_t N>
__device__ void ger(T alpha, T *x, T *y, T *A,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t col = 0; col < N; col++) {
        T ay = alpha * y[col];
        for (uint32_t row = g.thread_rank(); row < M; row += g.size())
            A[row + col*M] += ay * x[row];
    }
}
