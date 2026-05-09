#pragma once
#include <cstdint>
#include <math.h>
#include <cooperative_groups.h>
#include "../base/L3/gemm.cuh"
namespace cgrps = cooperative_groups;

// glass::cgrps::gemm — delegates to shared gemm_impl

template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T beta, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, beta, C);
}

template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, C);
}

template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_ex(uint32_t m, uint32_t n, uint32_t k,
                         T alpha, T *A, T *B, T beta, T *C,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(
        g.thread_rank(), g.size(), m, n, k, alpha, A, B, beta, C);
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T beta, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), M, N, K, alpha, A, B, beta, C);
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T *C,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(
        g.thread_rank(), g.size(), M, N, K, alpha, A, B, C);
}

// invertMatrix — no cgrps variant needed (already uses __syncthreads internally)
// Exposed here for API completeness via glass::cgrps::invertMatrix
template <typename T>
__device__ void invertMatrix(uint32_t dimA, T *A, T *s_temp,
                              cgrps::thread_group g = cgrps::this_thread_block())
{
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivOff = pivRC * dimA;
        T pvInv = static_cast<T>(1) / A[pivRC + pivOff];
        for (unsigned ind = g.thread_rank(); ind < 2*dimA+1; ind++) {
            unsigned AInd = (ind < dimA) ? (ind + pivOff) : (pivRC + pivOff + (ind-dimA)*dimA);
            s_temp[ind] = A[AInd];
        }
        g.sync();
        for (unsigned ind = g.thread_rank(); ind < dimA*(dimA+1); ind += g.size()) {
            unsigned row = ind % dimA, col = ind / dimA, coff = ind - row;
            if (row == pivRC) A[row + pivOff + coff] *= pvInv;
            else A[row + pivOff + coff] -= s_temp[row]*pvInv*s_temp[dimA+col];
        }
        g.sync();
    }
}

// cholDecomp_InPlace
template <typename T>
__device__ void cholDecomp_InPlace(uint32_t n, T *s_A,
                                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t row = 0; row < n; row++) {
        if (g.thread_rank() == 0) {
            T sum = static_cast<T>(0), val = s_A[n*row + row];
            for (int32_t rl = 0; rl < (int32_t)row; rl++) sum += s_A[rl*n+row]*s_A[rl*n+row];
            s_A[row*n + row] = sqrtf(val - sum);
        }
        g.sync();
        for (uint32_t col = g.thread_rank() + row + 1; col < n; col += g.size()) {
            T sum = static_cast<T>(0);
            for (uint32_t kk = 0; kk < row; kk++) sum += s_A[kk*n+col]*s_A[kk*n+row];
            s_A[row*n+col] = (static_cast<T>(1)/s_A[row*n+row])*(s_A[row*n+col] - sum);
        }
        g.sync();
    }
}

// trsm
template <typename T>
__device__ void trsm(uint32_t n, T *L, T *b,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t col = 0; col < n; col++) {
        if (g.thread_rank() == 0) b[col] /= L[col*n + col];
        g.sync();
        T factor = b[col];
        for (uint32_t row = g.thread_rank() + col + 1; row < n; row += g.size())
            b[row] -= L[col*n + row] * factor;
        g.sync();
    }
}
