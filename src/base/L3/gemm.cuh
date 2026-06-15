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

// compile-time impl: M, N, K as template params so el%M and el/M use
// cheap compiler-generated magic-number multiply instead of MUFU.RCP
template <typename T, uint32_t M, uint32_t N, uint32_t K, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, T *A, T *B, T beta, T *C)
{
    constexpr uint32_t C_cols = TRANSPOSE_B ? N : K;
    constexpr uint32_t maxel  = M * C_cols;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % M, col = el / M;
        T res = static_cast<T>(0);
        if (TRANSPOSE_B) {
            for (uint32_t ind = 0; ind < N; ind++) {
                T a = ROW_MAJOR_A ? A[row*N + ind] : A[ind*M + row];
                T b = ROW_MAJOR_B ? B[col*N + ind] : B[ind*N + col];
                res += a * b;
            }
        } else {
            for (uint32_t ind = 0; ind < N; ind++) {
                T a = ROW_MAJOR_A ? A[row*N + ind] : A[ind*M + row];
                T b = ROW_MAJOR_B ? B[ind*K + col] : B[col*N + ind];
                res += a * b;
            }
        }
        uint32_t cidx = ROW_MAJOR_C ? (row*C_cols + col) : (col*M + row);
        C[cidx] = alpha*res + beta*C[cidx];
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, T *A, T *B, T *C)
{
    constexpr uint32_t C_cols = TRANSPOSE_B ? N : K;
    constexpr uint32_t maxel  = M * C_cols;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % M, col = el / M;
        T res = static_cast<T>(0);
        if (TRANSPOSE_B) {
            for (uint32_t ind = 0; ind < N; ind++) {
                T a = ROW_MAJOR_A ? A[row*N + ind] : A[ind*M + row];
                T b = ROW_MAJOR_B ? B[col*N + ind] : B[ind*N + col];
                res += a * b;
            }
        } else {
            for (uint32_t ind = 0; ind < N; ind++) {
                T a = ROW_MAJOR_A ? A[row*N + ind] : A[ind*M + row];
                T b = ROW_MAJOR_B ? B[ind*K + col] : B[col*N + ind];
                res += a * b;
            }
        }
        uint32_t cidx = ROW_MAJOR_C ? (row*C_cols + col) : (col*M + row);
        C[cidx] = alpha*res;
    }
}

// ─── runtime variants ─────────────────────────────────────────────────────────

/**
 * @brief General matrix-matrix multiply: `C = alpha * A * op(B) + beta * C` (GEMM).
 *
 * Runtime-size, single-block, flat-element parallelism: each thread owns output
 * elements strided over the block. Storage order is uniform across A, B, C
 * (`ROW_MAJOR` flag; false = column-major). NumPy equivalent:
 * `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (pure-SIMT path requires B square).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param m,n,k  Dimensions: A is m x n, B is n x k (or k x n when TRANSPOSE_B), C is m x (TRANSPOSE_B ? n : k).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 */
template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, m, n, k, alpha, A, B, beta, C);
}

/**
 * @brief General matrix-matrix multiply with implicit `beta = 0`: `C = alpha * A * op(B)` (GEMM).
 *
 * Runtime-size overload that overwrites C (the existing C is not read), avoiding
 * the `beta * C` term. Single-block, flat-element parallelism; uniform storage
 * order. NumPy equivalent: `C = alpha * A @ B`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T` (pure-SIMT path requires B square).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param m,n,k  Dimensions: A is m x n, B is n x k (or k x n when TRANSPOSE_B), C is m x (TRANSPOSE_B ? n : k).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 */
template <typename T, bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, m, n, k, alpha, A, B, C);
}

/**
 * @brief GEMM with per-matrix layout control: `C = alpha * A * op(B) + beta * C`.
 *
 * Like `gemm` but the storage order of A, B and C is set independently rather
 * than by a single uniform flag. Runtime-size, single-block. NumPy equivalent:
 * `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T`.
 * @tparam ROW_MAJOR_A/ROW_MAJOR_B/ROW_MAJOR_C  Storage order per matrix (false = column-major).
 * @param m,n,k  Dimensions: A is m x n, B is n x k (or k x n when TRANSPOSE_B), C is m x (TRANSPOSE_B ? n : k).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 */
template <typename T, bool TRANSPOSE_B,
          bool ROW_MAJOR_A, bool ROW_MAJOR_B, bool ROW_MAJOR_C>
__device__ void gemm_ex(uint32_t m, uint32_t n, uint32_t k,
                        T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_B, ROW_MAJOR_A, ROW_MAJOR_B, ROW_MAJOR_C>(rank, size, m, n, k, alpha, A, B, beta, C);
}

/**
 * @brief GEMM with per-matrix layout control and implicit `beta = 0`: `C = alpha * A * op(B)`.
 *
 * Per-matrix-layout overload that overwrites C (the existing C is not read).
 * Runtime-size, single-block. NumPy equivalent: `C = alpha * A @ B`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T`.
 * @tparam ROW_MAJOR_A/ROW_MAJOR_B/ROW_MAJOR_C  Storage order per matrix (false = column-major).
 * @param m,n,k  Dimensions: A is m x n, B is n x k (or k x n when TRANSPOSE_B), C is m x (TRANSPOSE_B ? n : k).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 */
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

/**
 * @brief Compile-time-size GEMM: `C = alpha * A * op(B) + beta * C` (GEMM).
 *
 * Dimensions are template parameters so the compiler unrolls the inner loops and
 * replaces the `el % M` / `el / M` index math with magic-number multiplies.
 * Single-block, flat-element parallelism, uniform storage order. NumPy
 * equivalent: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T`.
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, alpha, A, B, beta, C);
}

/**
 * @brief Compile-time-size GEMM with implicit `beta = 0`: `C = alpha * A * op(B)`.
 *
 * Compile-time-size overload that overwrites C (the existing C is not read).
 * Single-block, flat-element parallelism, uniform storage order. NumPy
 * equivalent: `C = alpha * A @ B`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  Compile-time dimensions: A is M x N, B is N x K (or K x N when TRANSPOSE_B), C is M x (TRANSPOSE_B ? N : K).
 * @tparam TRANSPOSE_B  If true, computes `A @ B^T`.
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_B = false, bool ROW_MAJOR = false>
__device__ void gemm(T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl_ct<T, M, N, K, TRANSPOSE_B, ROW_MAJOR, ROW_MAJOR, ROW_MAJOR>(rank, size, alpha, A, B, C);
}

// ─── tiled GEMM (shared-memory staging, column-major only) ───────────────────
/**
 * @brief Tiled GEMM with shared-memory staging: `C = alpha * A * B + beta * C`.
 *
 * Column-major only. Stages `TILE`-wide column blocks of A and the matching rows
 * of B into the caller-provided shared-memory scratch, accumulating the product
 * across tiles. Single-block; best when A/B columns can be reused from shared
 * memory. NumPy equivalent: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TILE  Column-block width staged per pass.
 * @param m,n,k  Dimensions: A is m x n, B is n x k, C is m x k.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices (column-major).
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param s_A    Shared scratch of `m * TILE` elements for the A tile.
 * @param s_B    Shared scratch of `TILE * k` elements for the B tile.
 */
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
/**
 * @brief Auto-dispatching GEMM: `C = alpha * A * B + beta * C` (column-major).
 *
 * Selects `gemm_tiled` when shared-memory scratch is provided and one output
 * element fits per thread (`m * k <= blockDim`); otherwise falls back to the
 * plain `gemm`. Single-block. NumPy equivalent: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TILE  Tile width passed through to `gemm_tiled`.
 * @param m,n,k  Dimensions: A is m x n, B is n x k, C is m x k.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices (column-major).
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param s_A    Optional shared scratch for the A tile (nullptr selects the plain path).
 * @param s_B    Optional shared scratch for the B tile.
 */
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
    /**
     * @brief High-speed (tiled) GEMM: `C = alpha * A * B + beta * C` (column-major).
     *
     * `glass::high_speed::gemm` — a thin alias that always takes the shared-memory
     * tiled path (`gemm_tiled`). Requires the caller to supply scratch. NumPy
     * equivalent: `C = alpha * A @ B + beta * C`.
     *
     * @tparam T  Scalar type.
     * @tparam TILE  Tile width passed through to `gemm_tiled`.
     * @param m,n,k  Dimensions: A is m x n, B is n x k, C is m x k.
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices (column-major).
     * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
     * @param C      In/out result matrix.
     * @param s_A    Shared scratch of `m * TILE` elements for the A tile.
     * @param s_B    Shared scratch of `TILE * k` elements for the B tile.
     */
    template <typename T, int TILE = 8>
    __device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                         T alpha, T *A, T *B, T beta, T *C,
                         T *s_A, T *s_B)
    {
        gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
    }
}
