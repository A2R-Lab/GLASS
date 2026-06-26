#pragma once
#include <cstdint>

// Compile-time-size GEMM (standard BLAS convention) with explicit column-major
// leading dimensions for A and B.
//   C is M×N, contraction K.
//   A is M×K column-major with leading dim A_RS:  A[m][k] = A[m + k*A_RS].
//   B is K×N column-major with leading dim B_RS:  B[k][n] = B[k + n*B_RS].
//   Output C is standard column-major with LDC = M.
// When A_RS==M and B_RS==K this is identical to glass::gemm<T,M,N,K>.
//
// Uses flat-element parallelism (same as gemm_impl_ct): each thread owns one
// output element el in [rank..M*N) step size.  row=el%M and col=el/M use the
// compiler magic-multiply since M is compile-time.  The inner K-loop is fully
// unrolled since K, A_RS, B_RS are all compile-time constants.

/**
 * @brief Strided compile-time GEMM: `C = alpha * A * B + beta * C` with custom leading dims.
 *
 * Column-major GEMM (standard convention: C is M×N, contraction K) where A and B
 * carry explicit leading dimensions (column strides): `A[m][k] = A[m + k*A_RS]`,
 * `B[k][n] = B[k + n*B_RS]`. Output C is standard column-major with LDC = M. When
 * `A_RS == M` and `B_RS == K` this is identical to `glass::gemm<T,M,N,K>`.
 * Single-block, flat-element parallelism; the inner K-loop is fully unrolled.
 * NumPy: `C = alpha * A @ B + beta * C` on the strided sub-views.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M×K, B is K×N, C is M×N (contraction K).
 * @tparam A_RS  Column stride (leading dimension) of A (default M).
 * @tparam B_RS  Column stride (leading dimension) of B (default K).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices (column-major, strided).
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix (column-major, LDC = M).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = K>
__device__ void gemm_strided(T alpha, const T* A, const T* B, T beta, T* C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < M * N; el += size) {
        uint32_t m = el % M, n = el / M;
        T res = static_cast<T>(0);
        for (uint32_t k = 0; k < K; k++)
            res += A[m + k * A_RS] * B[k + n * B_RS];
        C[m + n * M] = alpha * res + beta * C[m + n * M];
    }
}

/**
 * @brief Strided compile-time GEMM with implicit `beta = 0`: `C = alpha * A * B`.
 *
 * Same as the beta overload but overwrites C (the existing C is not read).
 * Column-major with explicit leading dims `A_RS` / `B_RS`; LDC = M. NumPy:
 * `C = alpha * A @ B`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M×K, B is K×N, C is M×N (contraction K).
 * @tparam A_RS  Column stride (leading dimension) of A (default M).
 * @tparam B_RS  Column stride (leading dimension) of B (default K).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices (column-major, strided).
 * @param C      Output result matrix (overwritten; column-major, LDC = M).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = K>
__device__ void gemm_strided(T alpha, const T* A, const T* B, T* C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < M * N; el += size) {
        uint32_t m = el % M, n = el / M;
        T res = static_cast<T>(0);
        for (uint32_t k = 0; k < K; k++)
            res += A[m + k * A_RS] * B[k + n * B_RS];
        C[m + n * M] = alpha * res;
    }
}
