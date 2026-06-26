#pragma once
#include <cstdint>

// ─── GEMM: standard-BLAS convention ──────────────────────────────────────────
//
//   C = alpha * op(A) * op(B) + beta * C
//
//   C is M×N, the contraction dimension is K (this matches BLAS / cuBLASDx /
//   NumPy / Eigen — `C[m,n] = alpha·Σ_k op(A)[m,k]·op(B)[k,n] + beta·C[m,n]`).
//
//   op(A) is M×K:  TRANSPOSE_A=false ⇒ A is M×K (A[m + k*M]);
//                  TRANSPOSE_A=true  ⇒ A is K×M (A[k + m*K], op(A)=Aᵀ).
//   op(B) is K×N:  TRANSPOSE_B=false ⇒ B is K×N (B[k + n*K]);
//                  TRANSPOSE_B=true  ⇒ B is N×K (B[n + k*N], op(B)=Bᵀ).
//   C:             ROW_MAJOR_C=false ⇒ column-major (C[m + n*M], LDC=M);
//                  ROW_MAJOR_C=true  ⇒ row-major   (C[m*N + n]).
//
// All four transpose combinations work at any M,N,K (no squareness assumption).
// Each C[m,n] is written by exactly one thread/lane (serial-K inner loop, no
// reduction) ⇒ trivially thread-count invariant. Column-major operands by default.
//
// Row-major operands need no separate path: a row-major M×K matrix is exactly a
// column-major K×M matrix, so pass it with the matching TRANSPOSE flag (see
// examples/11_rowmajor_is_transpose.cu). That is why there is a single ROW_MAJOR_C
// output flag and no per-operand row-major flags.
//
// NumPy: C = alpha*opA(A) @ opB(B) + beta*C ;
// Eigen: C.noalias() = alpha*(opA(A)*opB(B)) + beta*C;  (column-major default)

// ─── core impls: explicit rank/size + (TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C) ──

template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B, bool ROW_MAJOR_C>
__device__ void gemm_impl(uint32_t rank, uint32_t size,
                          uint32_t m_, uint32_t n_, uint32_t k_,
                          T alpha, const T *A, const T *B, T beta, T *C)
{
    const uint32_t maxel = m_ * n_;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t m = el % m_, n = el / m_;
        T res = static_cast<T>(0);
        for (uint32_t k = 0; k < k_; k++) {
            T a = TRANSPOSE_A ? A[k + m*k_] : A[m + k*m_];
            T b = TRANSPOSE_B ? B[n + k*n_] : B[k + n*k_];
            res += a * b;
        }
        uint32_t cidx = ROW_MAJOR_C ? (m*n_ + n) : (m + n*m_);
        C[cidx] = alpha*res + beta*C[cidx];
    }
}

template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B, bool ROW_MAJOR_C>
__device__ void gemm_impl(uint32_t rank, uint32_t size,
                          uint32_t m_, uint32_t n_, uint32_t k_,
                          T alpha, const T *A, const T *B, T *C)
{
    const uint32_t maxel = m_ * n_;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t m = el % m_, n = el / m_;
        T res = static_cast<T>(0);
        for (uint32_t k = 0; k < k_; k++) {
            T a = TRANSPOSE_A ? A[k + m*k_] : A[m + k*m_];
            T b = TRANSPOSE_B ? B[n + k*n_] : B[k + n*k_];
            res += a * b;
        }
        uint32_t cidx = ROW_MAJOR_C ? (m*n_ + n) : (m + n*m_);
        C[cidx] = alpha*res;
    }
}

// compile-time impl: M, N, K as template params so el%M and el/M use cheap
// compiler-generated magic-number multiply instead of MUFU.RCP, and the inner
// K-loop is fully unrolled.
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A, bool TRANSPOSE_B, bool ROW_MAJOR_C>
__device__ void gemm_impl_ct(uint32_t rank, uint32_t size,
                             T alpha, const T *A, const T *B, T beta, T *C)
{
    constexpr uint32_t maxel = M * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t m = el % M, n = el / M;
        T res = static_cast<T>(0);
        for (uint32_t k = 0; k < K; k++) {
            T a = TRANSPOSE_A ? A[k + m*K] : A[m + k*M];
            T b = TRANSPOSE_B ? B[n + k*N] : B[k + n*K];
            res += a * b;
        }
        uint32_t cidx = ROW_MAJOR_C ? (m*N + n) : (m + n*M);
        C[cidx] = alpha*res + beta*C[cidx];
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A, bool TRANSPOSE_B, bool ROW_MAJOR_C>
__device__ void gemm_impl_ct(uint32_t rank, uint32_t size,
                             T alpha, const T *A, const T *B, T *C)
{
    constexpr uint32_t maxel = M * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t m = el % M, n = el / M;
        T res = static_cast<T>(0);
        for (uint32_t k = 0; k < K; k++) {
            T a = TRANSPOSE_A ? A[k + m*K] : A[m + k*M];
            T b = TRANSPOSE_B ? B[n + k*N] : B[k + n*K];
            res += a * b;
        }
        uint32_t cidx = ROW_MAJOR_C ? (m*N + n) : (m + n*M);
        C[cidx] = alpha*res;
    }
}

// ─── runtime variants ─────────────────────────────────────────────────────────

/**
 * @brief General matrix-matrix multiply: `C = alpha * op(A) * op(B) + beta * C` (GEMM).
 *
 * Standard BLAS convention: `C` is `m×n`, contraction `k`. Runtime-size,
 * single-block, flat-element parallelism: each thread owns output elements
 * strided over the block. NumPy: `C = alpha * opA(A) @ opB(B) + beta * C`;
 * Eigen: `C.noalias() = alpha*(opA(A)*opB(B)) + beta*C;`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_A  If true, `A` is `k×m` and `op(A)=Aᵀ` (else `A` is `m×k`).
 * @tparam TRANSPOSE_B  If true, `B` is `n×k` and `op(B)=Bᵀ` (else `B` is `k×n`).
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major / Fortran, LDC=m).
 * @param m,n,k  Dimensions: `C` is `m×n`, contraction `k`.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices (column-major; shapes per the transpose flags).
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 */
template <typename T, bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, const T *A, const T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(rank, size, m, n, k, alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief GEMM with implicit `beta = 0`: `C = alpha * op(A) * op(B)` (overwrite).
 *
 * Runtime-size overload that overwrites C (the existing C is not read), avoiding
 * the `beta * C` term. NumPy: `C = alpha * opA(A) @ opB(B)`.
 *
 * @tparam T  Scalar type.
 * @tparam TRANSPOSE_A  If true, `A` is `k×m` and `op(A)=Aᵀ`.
 * @tparam TRANSPOSE_B  If true, `B` is `n×k` and `op(B)=Bᵀ`.
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @param m,n,k  Dimensions: `C` is `m×n`, contraction `k`.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 */
template <typename T, bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, const T *A, const T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl<T, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(rank, size, m, n, k, alpha, A, B, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── compile-time size variants ───────────────────────────────────────────────

/**
 * @brief Compile-time-size GEMM: `C = alpha * op(A) * op(B) + beta * C` (GEMM).
 *
 * Dimensions are template parameters so the compiler unrolls the inner loop and
 * replaces the `el % M` / `el / M` index math with magic-number multiplies.
 * Standard BLAS convention: `C` is `M×N`, contraction `K`. NumPy:
 * `C = alpha * opA(A) @ opB(B) + beta * C`; Eigen:
 * `C.noalias() = alpha*(opA(A)*opB(B)) + beta*C;`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  `C` is `M×N`, contraction `K`.
 * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ` (else `A` is `M×K`).
 * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ` (else `B` is `K×N`).
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major / Fortran, LDC=M).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(T alpha, const T *A, const T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(rank, size, alpha, A, B, beta, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Compile-time-size GEMM with implicit `beta = 0`: `C = alpha * op(A) * op(B)`.
 *
 * Compile-time-size overload that overwrites C (the existing C is not read).
 * NumPy: `C = alpha * opA(A) @ opB(B)`.
 *
 * @tparam T  Scalar type.
 * @tparam M,N,K  `C` is `M×N`, contraction `K`.
 * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
 * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
 * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices.
 * @param C      Output result matrix (overwritten).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(T alpha, const T *A, const T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(rank, size, alpha, A, B, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-warp compile-time GEMM ───────────────────────────────────────────
namespace warp {
    /**
     * @brief Single-warp compile-time-size GEMM: `C = alpha * op(A) * op(B) + beta * C`.
     *
     * One 32-lane warp computes the product with flat per-element parallelism
     * (lanes stride over the `M*N` outputs, serial-K inner loop) — same semantics
     * as the block-scoped compile-time `gemm`, but scoped to a single warp for
     * warp-per-problem kernels (e.g. 4×4 homogeneous-transform multiplies). No
     * inter-lane communication, no sync. `C` must not alias `A`/`B`.
     *
     * @tparam T  Scalar type.
     * @tparam M,N,K  `C` is `M×N`, contraction `K`.
     * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
     * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
     * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices.
     * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
     * @param C      In/out result matrix.
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false>
    __device__ void gemm(T alpha, const T *A, const T *B, T beta, T *C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(lane, 32u, alpha, A, B, beta, C);
    }

    /**
     * @brief Single-warp compile-time-size GEMM with implicit `beta = 0`: `C = alpha * op(A) * op(B)`.
     *
     * Overwrites C (the existing C is not read). Otherwise identical to the
     * beta overload above.
     *
     * @tparam T  Scalar type.
     * @tparam M,N,K  `C` is `M×N`, contraction `K`.
     * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
     * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
     * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices.
     * @param C      Output result matrix (overwritten).
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false>
    __device__ void gemm(T alpha, const T *A, const T *B, T *C)
    {
        uint32_t lane = (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y) & 31;
        gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(lane, 32u, alpha, A, B, C);
    }
}

// ─── tiled GEMM (shared-memory staging, column-major, no transpose) ──────────
/**
 * @brief Tiled GEMM with shared-memory staging: `C = alpha * A * B + beta * C`.
 *
 * Standard convention, column-major, no transpose. `C` is `m×n`, contraction
 * `k`. Stages `TILE`-wide column blocks of A (`m×TILE`) and the matching row
 * blocks of B (`TILE×n`) into the caller-provided shared scratch, accumulating
 * across tiles. Single-block; best when A/B values can be reused from shared
 * memory. NumPy: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TILE  Column-block width staged per pass.
 * @param m,n,k  Dimensions: A is m×k, B is k×n, C is m×n.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices (column-major).
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param s_A    Shared scratch of `m * TILE` elements for the A tile.
 * @param s_B    Shared scratch of `TILE * n` elements for the B tile.
 */
template <typename T, int TILE = 8>
__device__ void gemm_tiled(uint32_t m, uint32_t n, uint32_t k,
                            T alpha, const T *A, const T *B, T beta, T *C,
                            T *s_A, T *s_B)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    uint32_t mn   = m * n;
    bool valid    = (rank < mn);
    uint32_t crow = valid ? (rank % m) : 0;
    uint32_t ccol = valid ? (rank / m) : 0;
    T acc = static_cast<T>(0);

    for (uint32_t t = 0; t < k; t += TILE) {
        uint32_t tile_end = (t + TILE < k) ? (t + TILE) : k;
        uint32_t tile_w   = tile_end - t;
        // stage A columns [t, t+tile_w):  A is m×k col-major (LDA=m)
        for (uint32_t i = rank; i < m * tile_w; i += size) {
            uint32_t ar = i % m, ac = i / m;
            s_A[ar + ac*m] = A[ar + (t+ac)*m];
        }
        // stage B rows [t, t+tile_w):  B is k×n col-major (LDB=k)
        for (uint32_t i = rank; i < tile_w * n; i += size) {
            uint32_t br = i % tile_w, bc = i / tile_w;
            s_B[br + bc*tile_w] = B[(t+br) + bc*k];
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

// ─── auto-dispatch: tiled when scratch provided and m*n <= blockDim ──────────
/**
 * @brief Auto-dispatching GEMM: `C = alpha * A * B + beta * C` (column-major).
 *
 * Selects `gemm_tiled` when shared-memory scratch is provided and one output
 * element fits per thread (`m * n <= blockDim`); otherwise falls back to the
 * plain `gemm`. Standard convention: C is m×n, contraction k. Single-block.
 * NumPy: `C = alpha * A @ B + beta * C`.
 *
 * @tparam T  Scalar type.
 * @tparam TILE  Tile width passed through to `gemm_tiled`.
 * @param m,n,k  Dimensions: A is m×k, B is k×n, C is m×n.
 * @param alpha  Scalar multiplier on the product.
 * @param A,B    Input matrices (column-major).
 * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C      In/out result matrix.
 * @param s_A    Optional shared scratch for the A tile (nullptr selects the plain path).
 * @param s_B    Optional shared scratch for the B tile.
 */
template <typename T, int TILE = 8>
__device__ void gemm_dispatch(uint32_t m, uint32_t n, uint32_t k,
                               T alpha, const T *A, const T *B, T beta, T *C,
                               T *s_A = nullptr, T *s_B = nullptr)
{
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    if (s_A != nullptr && m*n <= size)
        gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
    else
        gemm<T>(m, n, k, alpha, A, B, beta, C);
}

namespace high_speed {
    /**
     * @brief High-speed (tiled) GEMM: `C = alpha * A * B + beta * C` (column-major).
     *
     * `glass::high_speed::gemm` — a thin alias that always takes the shared-memory
     * tiled path (`gemm_tiled`). Requires the caller to supply scratch. Standard
     * convention: C is m×n, contraction k. NumPy: `C = alpha * A @ B + beta * C`.
     *
     * @tparam T  Scalar type.
     * @tparam TILE  Tile width passed through to `gemm_tiled`.
     * @param m,n,k  Dimensions: A is m×k, B is k×n, C is m×n.
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices (column-major).
     * @param beta   Scalar multiplier on the existing C (C is read; caller must initialize it).
     * @param C      In/out result matrix.
     * @param s_A    Shared scratch of `m * TILE` elements for the A tile.
     * @param s_B    Shared scratch of `TILE * n` elements for the B tile.
     */
    template <typename T, int TILE = 8>
    __device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                         T alpha, const T *A, const T *B, T beta, T *C,
                         T *s_A, T *s_B)
    {
        gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
    }
}
