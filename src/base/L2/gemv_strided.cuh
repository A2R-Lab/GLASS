#pragma once
#include <cstdint>

// Compile-time-size GEMV with an explicit column-major leading dimension.
// A[i][j] = A_ptr[i + j*ROW_STRIDE].
// When ROW_STRIDE == M this is identical to glass::gemv<T,M,N>.
// Useful for spatial 6x6 matrices embedded in larger arrays (e.g., GRiD).
//
// Uses threadIdx-based parallelism (same as gemv_impl_ct): threads are
// distributed over rows, and the inner column loop is fully unrolled by the
// compiler since N and ROW_STRIDE are compile-time constants.

template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M>
__device__ void row_strided_gemv(const T* A, const T* x, T* y, T alpha, T beta)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t row = rank; row < M; row += size) {
        T res = static_cast<T>(0);
        for (uint32_t col = 0; col < N; col++)
            res += A[row + col * ROW_STRIDE] * x[col];
        y[row] = alpha * res + beta * y[row];
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M>
__device__ void row_strided_gemv(const T* A, const T* x, T* y, T alpha)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t row = rank; row < M; row += size) {
        T res = static_cast<T>(0);
        for (uint32_t col = 0; col < N; col++)
            res += A[row + col * ROW_STRIDE] * x[col];
        y[row] = alpha * res;
    }
}
