#pragma once
#include <cstdint>

// SIMT batched GEMM with a 1D-launch convention. Distinct from
// glass::nvidia::gemm_batched (the cuBLASDx-backed wrapper in src/nvidia/l3.cuh)
// which requires a 2D launch dim3(TC, BATCH).
//
// The "_1d" variants below process BATCH independent (M×N)·(N×K) GEMMs using
// a flat 1D thread grid: every thread strides across BATCH*M*(K or N) output
// elements and derives its (batch, row, col) indices from the linear index.
// Suits codegen patterns where the surrounding kernel already has a 1D launch
// and the batched data lives behind either an array of pointers or a single
// contiguous stride.
//
// Both forms exist:
//   gemm_batched_1d         — A,B,C are arrays of pointers (length BATCH).
//   gemm_strided_batched_1d — A,B,C are base pointers, each batch lives at
//                             base + batch * STRIDE_{A,B,C} (default tight).

// --- Internal: compile-time-size SIMT kernel for one (batch, row, col) tuple.

// Matches the col-major indexing of ::glass::gemm_impl_ct in
// src/base/L3/gemm.cuh:
//   TRANSPOSE_B=false → B is N×K, b = B[col*N + ind]   (B[ind, col])
//   TRANSPOSE_B=true  → B is N×N, b = B[ind*N + col]   (B[col, ind])
// The TRANSPOSE_B=true branch requires B to be square (the same restriction
// the non-batched SIMT gemm has — see README "Notes").
template <typename T, uint32_t M, uint32_t N, uint32_t K, bool TRANSPOSE_B>
__device__ inline T _gemm_batched_1d_dot(const T* A, const T* B,
                                         uint32_t row, uint32_t col)
{
    T res = static_cast<T>(0);
    if (TRANSPOSE_B) {
        for (uint32_t ind = 0; ind < N; ind++)
            res += A[ind * M + row] * B[ind * N + col];
    } else {
        for (uint32_t ind = 0; ind < N; ind++)
            res += A[ind * M + row] * B[col * N + ind];
    }
    return res;
}

// ─── pointer-array variant ────────────────────────────────────────────────

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          bool TRANSPOSE_B = false>
__device__ void gemm_batched_1d(T alpha,
                                T* const* A, T* const* B,
                                T beta,
                                T* const* C)
{
    constexpr uint32_t C_cols    = TRANSPOSE_B ? N : K;
    constexpr uint32_t per_batch = M * C_cols;
    constexpr uint32_t total     = BATCH * per_batch;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < total; el += size) {
        uint32_t b   = el / per_batch;
        uint32_t loc = el - b * per_batch;
        uint32_t row = loc % M;
        uint32_t col = loc / M;
        T res = _gemm_batched_1d_dot<T, M, N, K, TRANSPOSE_B>(A[b], B[b], row, col);
        uint32_t cidx = col * M + row;
        C[b][cidx] = alpha * res + beta * C[b][cidx];
    }
}

// "Pure" output variant: C := alpha * A*B (no read of existing C).
template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          bool TRANSPOSE_B = false>
__device__ void gemm_batched_1d(T alpha,
                                T* const* A, T* const* B,
                                T* const* C)
{
    constexpr uint32_t C_cols    = TRANSPOSE_B ? N : K;
    constexpr uint32_t per_batch = M * C_cols;
    constexpr uint32_t total     = BATCH * per_batch;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < total; el += size) {
        uint32_t b   = el / per_batch;
        uint32_t loc = el - b * per_batch;
        uint32_t row = loc % M;
        uint32_t col = loc / M;
        T res = _gemm_batched_1d_dot<T, M, N, K, TRANSPOSE_B>(A[b], B[b], row, col);
        uint32_t cidx = col * M + row;
        C[b][cidx] = alpha * res;
    }
}

// ─── strided variant (single base pointer per matrix) ─────────────────────

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t A_STRIDE = M * N,
          uint32_t B_STRIDE = N * K,
          uint32_t C_STRIDE = M * K,
          bool TRANSPOSE_B = false>
__device__ void gemm_strided_batched_1d(T alpha,
                                        T* A, T* B,
                                        T beta,
                                        T* C)
{
    constexpr uint32_t C_cols    = TRANSPOSE_B ? N : K;
    constexpr uint32_t per_batch = M * C_cols;
    constexpr uint32_t total     = BATCH * per_batch;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < total; el += size) {
        uint32_t b   = el / per_batch;
        uint32_t loc = el - b * per_batch;
        uint32_t row = loc % M;
        uint32_t col = loc / M;
        const T* Ab = A + b * A_STRIDE;
        const T* Bb = B + b * B_STRIDE;
        T*       Cb = C + b * C_STRIDE;
        T res = _gemm_batched_1d_dot<T, M, N, K, TRANSPOSE_B>(Ab, Bb, row, col);
        uint32_t cidx = col * M + row;
        Cb[cidx] = alpha * res + beta * Cb[cidx];
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t A_STRIDE = M * N,
          uint32_t B_STRIDE = N * K,
          uint32_t C_STRIDE = M * K,
          bool TRANSPOSE_B = false>
__device__ void gemm_strided_batched_1d(T alpha,
                                        T* A, T* B,
                                        T* C)
{
    constexpr uint32_t C_cols    = TRANSPOSE_B ? N : K;
    constexpr uint32_t per_batch = M * C_cols;
    constexpr uint32_t total     = BATCH * per_batch;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t el = rank; el < total; el += size) {
        uint32_t b   = el / per_batch;
        uint32_t loc = el - b * per_batch;
        uint32_t row = loc % M;
        uint32_t col = loc / M;
        const T* Ab = A + b * A_STRIDE;
        const T* Bb = B + b * B_STRIDE;
        T*       Cb = C + b * C_STRIDE;
        T res = _gemm_batched_1d_dot<T, M, N, K, TRANSPOSE_B>(Ab, Bb, row, col);
        uint32_t cidx = col * M + row;
        Cb[cidx] = alpha * res;
    }
}
