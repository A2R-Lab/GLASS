#pragma once
#include <cstdint>

// core impl: explicit rank/size + layout flags
template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n, uint32_t k,
                           T alpha, T *A, T *B, T beta, T *C)
{
    const uint32_t C_cols = TRANSPOSE_B ? n : k;
    const uint32_t maxel  = m * C_cols;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % m, col = el / m;
        T res = static_cast<T>(0);
        if (TRANSPOSE_B) {
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind] : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[col*n + ind] : B[ind*n + col];
                res += a * b;
            }
        } else {
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind] : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[ind*k + col] : B[col*n + ind];
                res += a * b;
            }
        }
        uint32_t cidx = ROW_MAJOR_C ? (row*C_cols + col) : (col*m + row);
        C[cidx] = alpha*res + beta*C[cidx];
    }
}

template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_impl(uint32_t rank, uint32_t size,
                           uint32_t m, uint32_t n, uint32_t k,
                           T alpha, T *A, T *B, T *C)
{
    const uint32_t C_cols = TRANSPOSE_B ? n : k;
    const uint32_t maxel  = m * C_cols;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % m, col = el / m;
        T res = static_cast<T>(0);
        if (TRANSPOSE_B) {
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind] : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[col*n + ind] : B[ind*n + col];
                res += a * b;
            }
        } else {
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind] : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[ind*k + col] : B[col*n + ind];
                res += a * b;
            }
        }
        uint32_t cidx = ROW_MAJOR_C ? (row*C_cols + col) : (col*m + row);
        C[cidx] = alpha*res;
    }
}

// ─── runtime variants ─────────────────────────────────────────────────────────

template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, m, n, k, alpha, A, B, beta, C);
}

template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, m, n, k, alpha, A, B, C);
}

template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_ex(uint32_t m, uint32_t n, uint32_t k,
                        T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(rank, size, m, n, k, alpha, A, B, beta, C);
}

template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_ex(uint32_t m, uint32_t n, uint32_t k,
                        T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(rank, size, m, n, k, alpha, A, B, C);
}

// ─── compile-time size variants ───────────────────────────────────────────────

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, M, N, K, alpha, A, B, beta, C);
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, M, N, K, alpha, A, B, C);
}

// ─── tiled GEMM (shared-memory staging, column-major only) ───────────────────
template <typename T, int TILE = 8>
__device__ void gemm_tiled(uint32_t m, uint32_t n, uint32_t k,
                            T alpha, T *A, T *B, T beta, T *C,
                            T *s_A, T *s_B)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t mk   = m * k;
    bool valid    = (rank < mk);
    uint32_t crow = valid ? (rank % m) : 0;
    uint32_t ccol = valid ? (rank / m) : 0;
    T acc = static_cast<T>(0);

    for (uint32_t t = 0; t < n; t += TILE) {
        uint32_t tile_end = (t + TILE < n) ? (t + TILE) : n;
        uint32_t tile_w   = tile_end - t;
        for (uint32_t i = rank; i < m * tile_w; i += size) {
            uint32_t ar = i % m, ac = i / m;
            s_A[ar + ac*m] = A[ar + (t+ac)*m];
        }
        for (uint32_t i = rank; i < tile_w * k; i += size) {
            uint32_t br = i % tile_w, bc = i / tile_w;
            s_B[br + bc*tile_w] = B[(t+br) + bc*n];
        }
        __syncthreads();
        if (valid) {
            for (uint32_t i = 0; i < tile_w; i++)
                acc += s_A[crow + i*m] * s_B[i + ccol*tile_w];
        }
        __syncthreads();
    }
    if (valid) C[crow + ccol*m] = alpha*acc + beta*C[crow + ccol*m];
}

// ─── auto-dispatch: tiled when scratch provided and m*k <= blockDim ──────────
template <typename T, int TILE = 8>
__device__ void gemm_dispatch(uint32_t m, uint32_t n, uint32_t k,
                               T alpha, T *A, T *B, T beta, T *C,
                               T *s_A = nullptr, T *s_B = nullptr)
{
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    if (s_A != nullptr && m*k <= size)
        gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
    else
        gemm<T, false>(m, n, k, alpha, A, B, beta, C);
}

namespace high_speed {
    template <typename T, int TILE = 8>
    __device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                         T alpha, T *A, T *B, T beta, T *C,
                         T *s_A, T *s_B)
    {
        gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
    }
}
