#pragma once

#ifndef GEMM_H
#define GEMM_H

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// ─── gemm_impl: core with per-matrix storage-order flags ──────────────────────
// C(m×C_cols) = alpha * op(A) * op(B) + beta * C
//   TRANSPOSE_B=false: A(m×n) * B(n×k) → C(m×k); inner dim = n
//   TRANSPOSE_B=true:  A(m×n) * B(n×n)^T → C(m×n); B must be square n×n
//
// ROW_MAJOR_A/B/C=false: column-major storage A[row + col*ld]  [default, cuBLAS]
// ROW_MAJOR_A/B/C=true:  row-major storage    A[row*cols + col] [C array style]
//
// All bools are compile-time constants — the compiler eliminates dead branches.
template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__
void gemm_impl(std::uint32_t m,
               std::uint32_t n,
               std::uint32_t k,
               T alpha,
               T *A,
               T *B,
               T beta,
               T *C,
               cgrps::thread_group g = cgrps::this_thread_block())
{
    // C is m×n when TRANSPOSE_B, else m×k
    const uint32_t C_cols = TRANSPOSE_B ? n : k;
    const uint32_t max = m * C_cols;

    for (uint32_t element = g.thread_rank(); element < max; element += g.size()) {
        T res = static_cast<T>(0);
        uint32_t row = element % m;
        uint32_t col = element / m;

        if (TRANSPOSE_B) {
            // inner product of A[row,:] and B[col,:] (B row because B is transposed)
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind]   : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[col*n + ind]   : B[ind*n + col]; // B^T[ind][col]=B[col][ind]
                res += a * b;
            }
        } else {
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind]   : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[ind*k + col]   : B[col*n + ind];
                res += a * b;
            }
        }

        uint32_t c_idx = ROW_MAJOR_C ? (row * C_cols + col) : (col * m + row);
        C[c_idx] = alpha * res + beta * C[c_idx];
    }
}

// no-beta overload (C = alpha * A * B)
template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__
void gemm_impl(std::uint32_t m,
               std::uint32_t n,
               std::uint32_t k,
               T alpha,
               T *A,
               T *B,
               T *C,
               cgrps::thread_group g = cgrps::this_thread_block())
{
    const uint32_t C_cols = TRANSPOSE_B ? n : k;
    const uint32_t max = m * C_cols;

    for (uint32_t element = g.thread_rank(); element < max; element += g.size()) {
        T res = static_cast<T>(0);
        uint32_t row = element % m;
        uint32_t col = element / m;

        if (TRANSPOSE_B) {
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind]   : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[col*n + ind]   : B[ind*n + col];
                res += a * b;
            }
        } else {
            for (uint32_t ind = 0; ind < n; ind++) {
                T a = ROW_MAJOR_A ? A[row*n + ind]   : A[ind*m + row];
                T b = ROW_MAJOR_B ? B[ind*k + col]   : B[col*n + ind];
                res += a * b;
            }
        }

        uint32_t c_idx = ROW_MAJOR_C ? (row * C_cols + col) : (col * m + row);
        C[c_idx] = alpha * res;
    }
}

// ─── gemm: single-flag convenience wrapper ────────────────────────────────────
// ROW_MAJOR applies to all three matrices. Default false = column-major (backward-compatible).
// With-beta overload requires explicit thread_group (legacy GLASS behavior).
template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__
void gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha,
          T *A,
          T *B,
          T beta,
          T *C,
          cgrps::thread_group g)
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(m, n, k, alpha, A, B, beta, C, g);
}

// No-beta overload (C = alpha * A * B) with default thread_group.
template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__
void gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha,
          T *A,
          T *B,
          T *C,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(m, n, k, alpha, A, B, C, g);
}

// ─── gemm_ex: per-matrix storage-order control ───────────────────────────────
template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__
void gemm_ex(std::uint32_t m,
             std::uint32_t n,
             std::uint32_t k,
             T alpha,
             T *A,
             T *B,
             T beta,
             T *C,
             cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(m, n, k, alpha, A, B, beta, C, g);
}

template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__
void gemm_ex(std::uint32_t m,
             std::uint32_t n,
             std::uint32_t k,
             T alpha,
             T *A,
             T *B,
             T *C,
             cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(m, n, k, alpha, A, B, C, g);
}


// ─── Legacy helpers (kept for internal use / compatibility) ───────────────────

/*
    dot product of two vectors
    s_temp is temporary shared memory to incrementally store the result
    this method is intended to be run by a single thread, as part of a larger matmul operation
    x and y are input vectors
    store the result in out
    n is the length of the vectors
    x_stride is the stride of x
    y_stride is the stride of y
    g is the thread group
*/
template <typename T>
__device__
void dot_single(T * out,
         T * s_temp,
         uint32_t n,
         T *x,
         int x_stride,
         T *y,
         int y_stride,
         cgrps::thread_group g = cgrps::this_thread_block()) {
    s_temp[threadIdx.x] = 0;
    for(uint32_t i = 0; i < n; i++){
        s_temp[threadIdx.x] += x[i * x_stride] * y[i * y_stride];
    }

    *out = s_temp[threadIdx.x];
}

/*
    Function for a single dot product computation as part of a larger matrix multiply
    A and B are pointers to the start of the input matrices, stored in column major format
    ld_A and ld_B are the number of rows in A and B, since we are using column major storage
    A_vec_ind and B_vec_ind are row/column indices of the components of A and B which will be multiplied
    If TRANSPOSE_A and TRANSPOSE_B are each false, we then want the dot product of the A_vec_ind-th row of A and the B_vec_ind-th column of B
    If TRANSPOSE_A is true, we want the dot product of the A_vec_ind-th column of A and the B_vec_ind-th column of B
    If TRANSPOSE_B is true, we want the dot product of the A_vec_ind-th row of A and the B_vec_ind-th row of B

    store the result in out
*/
template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B>
__device__
void dotMM(T * out,
           T * s_temp,
           uint32_t n,
           T * A,
           int ld_A,
           int A_vec_ind,
           T * B,
           int ld_B,
           int B_vec_ind,
           cgrps::thread_group g = cgrps::this_thread_block()) {

    int a_start_ind;
    int a_stride;
    if (TRANSPOSE_A) {
        a_start_ind = ld_A * A_vec_ind;
        a_stride = 1;
    } else {
        a_start_ind = A_vec_ind;
        a_stride = ld_A;
    }

    int b_start_ind;
    int b_stride;
    if (TRANSPOSE_B){
        b_start_ind = B_vec_ind;
        b_stride = ld_B;
    } else {
        b_start_ind = ld_B * B_vec_ind;
        b_stride = 1;
    }

    dot_single(out, s_temp, n, &A[a_start_ind], a_stride, &B[b_start_ind], b_stride, g);
}

template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B>
__device__
void gemm_v2(std::uint32_t A_rows,
          std::uint32_t A_cols,
          std::uint32_t B_rows,
          std::uint32_t B_cols,
          T *A,
          int ld_A,
          T *B,
          int ld_B,
          T *C,
          T * s_temp,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    int C_size;
    int vector_length;
    if (TRANSPOSE_A) {
        C_size = A_cols * B_cols;
        vector_length = A_rows;
        for (uint32_t i = g.thread_rank(); i < C_size; i += g.size()){
            int a_vec_ind = i % A_cols;
            int b_vec_ind = i / A_cols;
            dotMM<T, true, false>(&C[i], s_temp, vector_length, A, ld_A, a_vec_ind, B, ld_B, b_vec_ind, g);
        }
    } else if (TRANSPOSE_B)
    {
        C_size = A_rows * B_rows;
        vector_length = A_cols;
        for (uint32_t i = g.thread_rank(); i < C_size; i += g.size()){
            int a_vec_ind = i % A_rows;
            int b_vec_ind = i / A_rows;
            dotMM<T, false, true>(&C[i], s_temp, vector_length, A, ld_A, a_vec_ind, B, ld_B, b_vec_ind, g);
        }
    } else {
        C_size = A_rows * B_cols;
        vector_length = A_cols;
        for (uint32_t i = g.thread_rank(); i < C_size; i += g.size()){
            int a_vec_ind = i % A_rows;
            int b_vec_ind = i / A_rows;
            dotMM<T, false, false>(&C[i], s_temp, vector_length, A, ld_A, a_vec_ind, B, ld_B, b_vec_ind, g);
        }
    }
}

template <typename T>
__device__ void simple_submatrix_gemm(T *s_C, const T *s_A, const T *s_B, int subA_rows, int subA_cols, int subB_cols,
                                      int ld_A, int ld_B, int ld_C)
{
    int row = threadIdx.x / subB_cols;
    int col = threadIdx.x % subB_cols;

    if (row < subA_rows && col < subB_cols)
    {
        T sum = 0;
        for (int k = 0; k < subA_cols; ++k)
        {
            sum += s_A[row + k * ld_A] * s_B[k * ld_B + col];
        }

        s_C[row + col * ld_C] = sum;
    }
}

/*for C=αA+B
more convenient than blas routine sometimes*/
template <typename T>
__device__
void matrixAlphaAdd(T alpha,
                    T *A,
                    T *B,
                    T *C,
                    std::uint32_t rows,
                    std::uint32_t cols,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    std::uint32_t n = rows * cols;

    for (std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()) {
        C[ind] = alpha * A[ind] + B[ind];
    }
}

// === glass::simple variants ===
namespace simple {

    // ─── simple::gemm_impl: per-matrix storage-order flags, no coop groups ───
    template <typename T, bool TRANSPOSE_B,
              bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
    __device__
    void gemm_impl(uint32_t m, uint32_t n, uint32_t k,
                   T alpha, T *A, T *B, T beta, T *C)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        const uint32_t C_cols = TRANSPOSE_B ? n : k;
        const uint32_t max = m * C_cols;

        for (uint32_t el = rank; el < max; el += size) {
            uint32_t row = el % m;
            uint32_t col = el / m;
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

            uint32_t c_idx = ROW_MAJOR_C ? (row * C_cols + col) : (col * m + row);
            C[c_idx] = alpha * res + beta * C[c_idx];
        }
    }

    // no-beta overload
    template <typename T, bool TRANSPOSE_B,
              bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
    __device__
    void gemm_impl(uint32_t m, uint32_t n, uint32_t k,
                   T alpha, T *A, T *B, T *C)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        const uint32_t C_cols = TRANSPOSE_B ? n : k;
        const uint32_t max = m * C_cols;

        for (uint32_t el = rank; el < max; el += size) {
            uint32_t row = el % m;
            uint32_t col = el / m;
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

            uint32_t c_idx = ROW_MAJOR_C ? (row * C_cols + col) : (col * m + row);
            C[c_idx] = alpha * res;
        }
    }

    // ─── simple::gemm: single-flag convenience ────────────────────────────────
    template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
    __device__
    void gemm(uint32_t m, uint32_t n, uint32_t k, T alpha, T *A, T *B, T beta, T *C)
    {
        gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(m, n, k, alpha, A, B, beta, C);
    }

    template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
    __device__
    void gemm(uint32_t m, uint32_t n, uint32_t k, T alpha, T *A, T *B, T *C)
    {
        gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(m, n, k, alpha, A, B, C);
    }

    // ─── simple::gemm_ex: per-matrix control ─────────────────────────────────
    template <typename T, bool TRANSPOSE_B,
              bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
    __device__
    void gemm_ex(uint32_t m, uint32_t n, uint32_t k, T alpha, T *A, T *B, T beta, T *C)
    {
        gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(m, n, k, alpha, A, B, beta, C);
    }

    template <typename T, bool TRANSPOSE_B,
              bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
    __device__
    void gemm_ex(uint32_t m, uint32_t n, uint32_t k, T alpha, T *A, T *B, T *C)
    {
        gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(m, n, k, alpha, A, B, C);
    }

    // ─── simple::gemm_tiled: shared-memory tiled GEMM (column-major only) ────
    //
    // C(m×k) = alpha * A(m×n) * B(n×k) + beta * C(m×k), all column-major.
    // Uses shared-memory staging to coalesce global B loads when m < 32.
    //
    // Requires m*k <= blockDim (one C element per thread). Threads with rank >= m*k
    // participate only in cooperative tile loading. For m*k > blockDim, use plain gemm.
    //
    // TILE: compile-time tile width (default 8).
    // s_A: caller scratch, size (m * TILE) * sizeof(T)
    // s_B: caller scratch, size (TILE * k) * sizeof(T)
    template <typename T, int TILE = 8>
    __device__
    void gemm_tiled(uint32_t m, uint32_t n, uint32_t k,
                    T alpha, T *A, T *B, T beta, T *C,
                    T *s_A, T *s_B)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        uint32_t mk   = m * k;

        bool valid  = (rank < mk);
        uint32_t crow = valid ? (rank % m) : 0;
        uint32_t ccol = valid ? (rank / m) : 0;
        T acc = static_cast<T>(0);

        // Tile loop outermost so ALL threads hit every __syncthreads().
        for (uint32_t t = 0; t < n; t += TILE) {
            uint32_t tile_end = (t + TILE < n) ? (t + TILE) : n;
            uint32_t tile_w   = tile_end - t;

            // ALL threads cooperate on tile loading.
            for (uint32_t i = rank; i < m * tile_w; i += size) {
                uint32_t ar = i % m;
                uint32_t ac = i / m;
                s_A[ar + ac * m] = A[ar + (t + ac) * m];
            }
            for (uint32_t i = rank; i < tile_w * k; i += size) {
                uint32_t br = i % tile_w;
                uint32_t bc = i / tile_w;
                s_B[br + bc * tile_w] = B[(t + br) + bc * n];
            }
            __syncthreads();

            if (valid) {
                for (uint32_t i = 0; i < tile_w; i++) {
                    acc += s_A[crow + i * m] * s_B[i + ccol * tile_w];
                }
            }
            __syncthreads();
        }

        if (valid) {
            C[crow + ccol * m] = alpha * acc + beta * C[crow + ccol * m];
        }
    }

    // ─── simple::gemm_dispatch: auto-select tiled or plain based on size ─────
    // If s_A/s_B are non-null and m*k <= blockDim: use tiled (better for m < 32).
    // Otherwise: use plain gemm. Column-major only.
    // Call glass_gemm_dispatch_smem() (host helper in glass.cuh) for scratch size.
    template <typename T, int TILE = 8>
    __device__
    void gemm_dispatch(uint32_t m, uint32_t n, uint32_t k,
                       T alpha, T *A, T *B, T beta, T *C,
                       T *s_A = nullptr, T *s_B = nullptr)
    {
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        if (s_A != nullptr && m * k <= size) {
            gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
        } else {
            gemm<T, false>(m, n, k, alpha, A, B, beta, C);
        }
    }

    namespace high_speed {
        // Alias for gemm_tiled — matches high_speed naming convention.
        template <typename T, int TILE = 8>
        __device__
        void gemm(uint32_t m, uint32_t n, uint32_t k,
                  T alpha, T *A, T *B, T beta, T *C,
                  T *s_A, T *s_B)
        {
            simple::gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
        }
    }
}
// ===


#endif
