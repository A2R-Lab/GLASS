#pragma once
#include <cstdint>
#include <math.h>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

/**
 * @brief Perform a Cholesky decomposition in place.
 *
 * Performs a Cholesky decomposition on the square matrix @p s_A, storing the result in the
 * lower triangular portion of @p s_A.
 *
 * @param T* s_A: a square symmetric positive definite matrix, column-major order.
 * @param  int n: number of cols/rows in a square matrix s_A (n*n)
 *
 */

template <typename T>
__device__
void cholDecomp_InPlace(std::uint32_t n,
                        T *s_A,
                        cgrps::thread_group g = cgrps::this_thread_block())
{
    for (std::uint32_t row = 0; row < n; row++) {
        if (g.thread_rank() == 0) {
            T sum = static_cast<T>(0);
            T val = s_A[n * row + row];
            for (std::int32_t row_l = 0; row_l < static_cast<std::int32_t>(row); row_l++) {
                sum += s_A[row_l * n + row] * s_A[row_l * n + row];
            }
            s_A[row * n + row] = sqrt(val - sum);
        }
        g.sync();

        for (std::uint32_t col = g.thread_rank() + row + 1; col < n; col += g.size()) {
            T sum = static_cast<T>(0);
            for (std::uint32_t k = 0; k < row; k++) {
                sum += s_A[k * n + col] * s_A[k * n + row];
            }
            s_A[row * n + col] = (static_cast<T>(1) / s_A[row * n + row]) * (s_A[row * n + col] - sum);
        }
        g.sync();
    }
}

// === glass::simple variants ===
namespace simple {
    template <typename T>
    __device__
    void cholDecomp_InPlace(uint32_t n, T *s_A)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t row = 0; row < n; row++) {
            if (rank == 0) {
                T sum = static_cast<T>(0);
                T val = s_A[n * row + row];
                for (int32_t row_l = 0; row_l < static_cast<int32_t>(row); row_l++) {
                    sum += s_A[row_l * n + row] * s_A[row_l * n + row];
                }
                s_A[row * n + row] = sqrt(val - sum);
            }
            __syncthreads();
            for (uint32_t col = rank + row + 1; col < n; col += size) {
                T sum = static_cast<T>(0);
                for (uint32_t k = 0; k < row; k++) {
                    sum += s_A[k * n + col] * s_A[k * n + row];
                }
                s_A[row * n + col] = (static_cast<T>(1) / s_A[row * n + row]) * (s_A[row * n + col] - sum);
            }
            __syncthreads();
        }
    }
}
// ===
