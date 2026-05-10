#pragma once
#include <cstdint>

// Compile-time-size GEMM with explicit column-major leading dimensions for A and B.
// A[i][j] = A_ptr[i + j*A_RS]; B[j][l] = B_ptr[j + l*B_RS].
// Output C is standard column-major with LDC=M.
// When A_RS==M and B_RS==N this is identical to glass::gemm<T,M,N,K>.
//
// Uses flat-element parallelism (same as gemm_impl_ct): each thread owns one
// output element el in [rank..M*K) step size.  row=el%M and col=el/M use
// compiler magic-multiply since M is compile-time.  The inner N-loop is fully
// unrolled since N, A_RS, B_RS are all compile-time constants.

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = N>
__device__ void row_strided_gemm(const T* A, const T* B, T* C, T alpha, T beta)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < M * K; el += size) {
        uint32_t row = el % M, col = el / M;
        T res = static_cast<T>(0);
        for (uint32_t ind = 0; ind < N; ind++)
            res += A[row + ind * A_RS] * B[ind + col * B_RS];
        C[el] = alpha * res + beta * C[el];
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = N>
__device__ void row_strided_gemm(const T* A, const T* B, T* C, T alpha)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < M * K; el += size) {
        uint32_t row = el % M, col = el / M;
        T res = static_cast<T>(0);
        for (uint32_t ind = 0; ind < N; ind++)
            res += A[row + ind * A_RS] * B[ind + col * B_RS];
        C[el] = alpha * res;
    }
}
