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
// Fast path: when TRANSPOSE_A=false and ROW_MAJOR_C=false (any TRANSPOSE_B),
// threads own 4-row register tiles of a C column (B value reused across 4 FMAs,
// contiguous A rows vector-loaded when aligned) — same per-element ascending-k
// accumulation order, so results stay bit-identical across thread counts.
// A, B, C must not alias each other (C is written while A/B are read; declared
// __restrict__). A and B may point to the SAME data (both are read-only).
//
// Row-major operands need no separate path: a row-major M×K matrix is exactly a
// column-major K×M matrix, so pass it with the matching TRANSPOSE flag (see
// examples/11_rowmajor_is_transpose.cu). That is why there is a single ROW_MAJOR_C
// output flag and no per-operand row-major flags.
//
// NumPy: C = alpha*opA(A) @ opB(B) + beta*C ;
// Eigen: C.noalias() = alpha*(opA(A)*opB(B)) + beta*C;  (column-major default)

// ─── shared 4-row register-tile helpers ──────────────────────────────────────
// Identical guarded copy lives in L2/gemv.cuh, L3/gemm.cuh, and L3/syrk.cuh:
// these base headers are #include'd inside `namespace glass { }` (with #pragma
// once), so whichever the umbrella pulls in first defines the block and the
// guard skips the twins — no dependency on include order.
#ifndef GLASS_TILE4_HELPERS_DEFINED
#define GLASS_TILE4_HELPERS_DEFINED

// Scalar types with a 16-byte vector load (float4 / 2×double2).
template <typename T> struct tile4_has_vec { static constexpr bool value = false; };
template <> struct tile4_has_vec<float>  { static constexpr bool value = true; };
template <> struct tile4_has_vec<double> { static constexpr bool value = true; };

// Load 4 consecutive elements p[0..3] into registers. The vector overloads
// require `p` 16-byte aligned. Loads only change HOW values reach registers —
// never any output's accumulation order — so vector vs scalar is bit-identical.
__device__ __forceinline__ void tile4_load(const float *p, float &a0, float &a1, float &a2, float &a3)
{
    float4 v = *reinterpret_cast<const float4 *>(p);
    a0 = v.x; a1 = v.y; a2 = v.z; a3 = v.w;
}
__device__ __forceinline__ void tile4_load(const double *p, double &a0, double &a1, double &a2, double &a3)
{
    double2 v0 = *reinterpret_cast<const double2 *>(p);
    double2 v1 = *reinterpret_cast<const double2 *>(p + 2);
    a0 = v0.x; a1 = v0.y; a2 = v1.x; a3 = v1.y;
}
template <typename T>
__device__ __forceinline__ void tile4_load(const T *p, T &a0, T &a1, T &a2, T &a3)
{
    a0 = p[0]; a1 = p[1]; a2 = p[2]; a3 = p[3];
}

// Base-pointer half of the vector-load precondition: with a leading dimension
// that is a multiple of 4 (float; 2 suffices for double but we require 4
// uniformly) and 4-row tiles starting at row r ≡ 0 (mod 4), every tile start
// A + r + k*ld stays 16-byte aligned iff the base pointer is.
template <typename T>
__device__ __forceinline__ bool tile4_aligned(const T *p)
{
    return (reinterpret_cast<uintptr_t>(p) & 0xFu) == 0u;
}
// 4-row tiles pay off only when every C column decomposes into WHOLE tiles
// and M is big enough for the B-register reuse to beat the 4x loss in
// parallel work units: an M%4 tail runs its rows serially inside one work
// unit and bottlenecks the whole block (measured sm_120: non-multiple-of-4
// M loses at every size probed — 5..81), and M=8 leaves too few tiles to
// feed more than one warp (loses at 64+ threads). Multiples of 4 from 12 up
// win at every probed thread count.
__host__ __device__ constexpr bool tile4_profitable(uint32_t m)
{
    return (m % 4u == 0u) && (m >= 12u);
}
#endif  // GLASS_TILE4_HELPERS_DEFINED

// ─── 4-row register-tiled gemm cores (TRANSPOSE_A=false, ROW_MAJOR_C=false) ──
// Each thread owns TILE=4 CONSECUTIVE ROWS of one C column: the flat M*N output
// element space becomes ceil(M/4)*N row-tiles, strided over the block exactly
// like gemm_impl strides elements (thread t handles tiles t, t+size, …). Per k,
// B[k,n] is loaded ONCE into a register and reused by 4 FMAs against the
// contiguous A[r..r+3, k] (one float4 / two double2 when M%4==0 and A is
// 16-byte aligned; scalar loads otherwise and for the M%4 tail rows). Each C
// element accumulates in its own register with the SAME serial ascending-k
// chain as the untiled loop and is written by exactly one thread, so
// thread-count invariance stays bit-identical. HAS_BETA folds the beta /
// implicit-beta=0 overloads into one body (beta unused and C never read when
// false). No interior sync needed — disjoint outputs, exactly like gemm_impl.

template <typename T, bool TRANSPOSE_B, bool HAS_BETA, bool VEC>
__device__ __forceinline__ void gemm_tile4_loop(uint32_t rank, uint32_t size,
                                                uint32_t m_, uint32_t n_, uint32_t k_,
                                                T alpha, const T *__restrict__ A,
                                                const T *__restrict__ B,
                                                T beta, T *__restrict__ C)
{
    const uint32_t tpc    = (m_ + 3u) / 4u;   // 4-row tiles per column of C
    const uint32_t ntiles = tpc * n_;
    for (uint32_t t = rank; t < ntiles; t += size) {
        const uint32_t n = t / tpc;
        const uint32_t r = (t % tpc) * 4u;
        if (r + 4u <= m_) {                    // full 4-row tile
            T acc0 = static_cast<T>(0), acc1 = static_cast<T>(0);
            T acc2 = static_cast<T>(0), acc3 = static_cast<T>(0);
            for (uint32_t k = 0; k < k_; k++) {
                const T b = TRANSPOSE_B ? B[n + k*n_] : B[k + n*k_];
                T a0, a1, a2, a3;
                if constexpr (VEC) {
                    tile4_load(A + (r + k*m_), a0, a1, a2, a3);
                } else {
                    a0 = A[r      + k*m_]; a1 = A[r + 1u + k*m_];
                    a2 = A[r + 2u + k*m_]; a3 = A[r + 3u + k*m_];
                }
                acc0 += a0 * b; acc1 += a1 * b; acc2 += a2 * b; acc3 += a3 * b;
            }
            T *__restrict__ c = C + (r + n*m_);
            if constexpr (HAS_BETA) {
                c[0] = beta_blend(alpha*acc0, beta, c[0]);
                c[1] = beta_blend(alpha*acc1, beta, c[1]);
                c[2] = beta_blend(alpha*acc2, beta, c[2]);
                c[3] = beta_blend(alpha*acc3, beta, c[3]);
            } else {
                c[0] = alpha*acc0; c[1] = alpha*acc1;
                c[2] = alpha*acc2; c[3] = alpha*acc3;
            }
        } else {                               // m_%4 tail rows: scalar per row
            for (uint32_t m = r; m < m_; m++) {
                T res = static_cast<T>(0);
                for (uint32_t k = 0; k < k_; k++) {
                    const T b = TRANSPOSE_B ? B[n + k*n_] : B[k + n*k_];
                    res += A[m + k*m_] * b;
                }
                if constexpr (HAS_BETA) C[m + n*m_] = beta_blend(alpha*res, beta, C[m + n*m_]);
                else                    C[m + n*m_] = alpha*res;
            }
        }
    }
}

template <typename T, bool TRANSPOSE_B, bool HAS_BETA>
__device__ void gemm_tile4(uint32_t rank, uint32_t size,
                           uint32_t m_, uint32_t n_, uint32_t k_,
                           T alpha, const T *__restrict__ A, const T *__restrict__ B,
                           T beta, T *__restrict__ C)
{
    if constexpr (tile4_has_vec<T>::value) {
        if ((m_ % 4u == 0u) && tile4_aligned(A)) {
            gemm_tile4_loop<T, TRANSPOSE_B, HAS_BETA, true>(rank, size, m_, n_, k_, alpha, A, B, beta, C);
            return;
        }
    }
    gemm_tile4_loop<T, TRANSPOSE_B, HAS_BETA, false>(rank, size, m_, n_, k_, alpha, A, B, beta, C);
}

// compile-time twin: M, N, K as template params (magic-number div/mod, fully
// unrolled k loop; M%4 tail statically absent when M is a multiple of 4).
template <typename T, uint32_t M, uint32_t N, uint32_t K, bool TRANSPOSE_B, bool HAS_BETA, bool VEC>
__device__ __forceinline__ void gemm_tile4_loop_ct(uint32_t rank, uint32_t size,
                                                   T alpha, const T *__restrict__ A,
                                                   const T *__restrict__ B,
                                                   T beta, T *__restrict__ C)
{
    constexpr uint32_t TPC    = (M + 3u) / 4u;   // 4-row tiles per column of C
    constexpr uint32_t NTILES = TPC * N;
    for (uint32_t t = rank; t < NTILES; t += size) {
        const uint32_t n = t / TPC;
        const uint32_t r = (t % TPC) * 4u;
        if (r + 4u <= M) {                     // full 4-row tile
            T acc0 = static_cast<T>(0), acc1 = static_cast<T>(0);
            T acc2 = static_cast<T>(0), acc3 = static_cast<T>(0);
            for (uint32_t k = 0; k < K; k++) {
                const T b = TRANSPOSE_B ? B[n + k*N] : B[k + n*K];
                T a0, a1, a2, a3;
                if constexpr (VEC) {
                    tile4_load(A + (r + k*M), a0, a1, a2, a3);
                } else {
                    a0 = A[r      + k*M]; a1 = A[r + 1u + k*M];
                    a2 = A[r + 2u + k*M]; a3 = A[r + 3u + k*M];
                }
                acc0 += a0 * b; acc1 += a1 * b; acc2 += a2 * b; acc3 += a3 * b;
            }
            T *__restrict__ c = C + (r + n*M);
            if constexpr (HAS_BETA) {
                c[0] = beta_blend(alpha*acc0, beta, c[0]);
                c[1] = beta_blend(alpha*acc1, beta, c[1]);
                c[2] = beta_blend(alpha*acc2, beta, c[2]);
                c[3] = beta_blend(alpha*acc3, beta, c[3]);
            } else {
                c[0] = alpha*acc0; c[1] = alpha*acc1;
                c[2] = alpha*acc2; c[3] = alpha*acc3;
            }
        } else {                               // M%4 tail rows: scalar per row
            for (uint32_t m = r; m < M; m++) {
                T res = static_cast<T>(0);
                for (uint32_t k = 0; k < K; k++) {
                    const T b = TRANSPOSE_B ? B[n + k*N] : B[k + n*K];
                    res += A[m + k*M] * b;
                }
                if constexpr (HAS_BETA) C[m + n*M] = beta_blend(alpha*res, beta, C[m + n*M]);
                else                    C[m + n*M] = alpha*res;
            }
        }
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, bool TRANSPOSE_B, bool HAS_BETA>
__device__ void gemm_tile4_ct(uint32_t rank, uint32_t size,
                              T alpha, const T *__restrict__ A, const T *__restrict__ B,
                              T beta, T *__restrict__ C)
{
    if constexpr (tile4_has_vec<T>::value && (M % 4u == 0u)) {
        if (tile4_aligned(A)) {
            gemm_tile4_loop_ct<T, M, N, K, TRANSPOSE_B, HAS_BETA, true>(rank, size, alpha, A, B, beta, C);
            return;
        }
    }
    gemm_tile4_loop_ct<T, M, N, K, TRANSPOSE_B, HAS_BETA, false>(rank, size, alpha, A, B, beta, C);
}

// ─── core impls: explicit rank/size + (TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C) ──
// No-transpose-A, column-major-C combos route to the 4-row register-tiled core
// when tile4_profitable(m) says the tiles win (M a multiple of 4, M >= 8 —
// measured sm_120 crossover); every other case takes the one-output-per-thread
// flat loop, byte-identical to before. Tiled and flat accumulate each C element
// with the same serial ascending-k chain, so the two paths agree bit-for-bit
// and the runtime gate cannot break thread-count invariance.

template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B, bool ROW_MAJOR_C>
__device__ void gemm_impl(uint32_t rank, uint32_t size,
                          uint32_t m_, uint32_t n_, uint32_t k_,
                          T alpha, const T *__restrict__ A, const T *__restrict__ B,
                          T beta, T *__restrict__ C)
{
    if constexpr (!TRANSPOSE_A && !ROW_MAJOR_C) {
        if (tile4_profitable(m_)) {
            gemm_tile4<T, TRANSPOSE_B, true>(rank, size, m_, n_, k_, alpha, A, B, beta, C);
            return;
        }
    }
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
        C[cidx] = beta_blend(alpha*res, beta, C[cidx]);
    }
}

template <typename T, bool TRANSPOSE_A, bool TRANSPOSE_B, bool ROW_MAJOR_C>
__device__ void gemm_impl(uint32_t rank, uint32_t size,
                          uint32_t m_, uint32_t n_, uint32_t k_,
                          T alpha, const T *__restrict__ A, const T *__restrict__ B,
                          T *__restrict__ C)
{
    if constexpr (!TRANSPOSE_A && !ROW_MAJOR_C) {
        if (tile4_profitable(m_)) {
            gemm_tile4<T, TRANSPOSE_B, false>(rank, size, m_, n_, k_, alpha, A, B, static_cast<T>(0), C);
            return;
        }
    }
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
                             T alpha, const T *__restrict__ A, const T *__restrict__ B,
                             T beta, T *__restrict__ C)
{
    if constexpr (!TRANSPOSE_A && !ROW_MAJOR_C && tile4_profitable(M)) {
        gemm_tile4_ct<T, M, N, K, TRANSPOSE_B, true>(rank, size, alpha, A, B, beta, C);
    } else {
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
            C[cidx] = beta_blend(alpha*res, beta, C[cidx]);
        }
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A, bool TRANSPOSE_B, bool ROW_MAJOR_C>
__device__ void gemm_impl_ct(uint32_t rank, uint32_t size,
                             T alpha, const T *__restrict__ A, const T *__restrict__ B,
                             T *__restrict__ C)
{
    if constexpr (!TRANSPOSE_A && !ROW_MAJOR_C && tile4_profitable(M)) {
        gemm_tile4_ct<T, M, N, K, TRANSPOSE_B, false>(rank, size, alpha, A, B, static_cast<T>(0), C);
    } else {
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
 * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
 * @param C      In/out result matrix.
 */
template <typename T, bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(uint32_t m, uint32_t n, uint32_t k,
                     T alpha, const T *__restrict__ A, const T *__restrict__ B,
                     T beta, T *__restrict__ C)
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
                     T alpha, const T *__restrict__ A, const T *__restrict__ B,
                     T *__restrict__ C)
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
 * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
 * @param C      In/out result matrix.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false, bool TRAILING_SYNC = true>
__device__ void gemm(T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
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
__device__ void gemm(T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(rank, size, alpha, A, B, C);
    if constexpr (TRAILING_SYNC) __syncthreads();
}

// ─── single-thread compile-time GEMM ─────────────────────────────────────────
namespace thread {
    /**
     * @brief Single-thread compile-time-size GEMM: `C = alpha * op(A) * op(B) + beta * C`.
     *
     * ONE thread computes the whole product, walking the `M*N` outputs serially
     * (serial-K inner loop) — same semantics as the block/warp compile-time `gemm`,
     * reusing the same `gemm_impl_ct` body with `(rank=0, size=1)`. 
     * Generally performs worse. 
     *
     * @warning The shared body switches to the 4-row **tile4** path when
     * `tile4_profitable(M)` (`M % 4 == 0 && M >= 12`), which issues `float4` /
     * `double2` vector loads through a `reinterpret_cast`. Unlikely to happen 
     * in the DOF where the single-thread is valuable
     *
     * @tparam T  Scalar type.
     * @tparam M,N,K  `C` is `M×N`, contraction `K`.
     * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
     * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
     * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices.
     * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
     * @param C      In/out result matrix.
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false>
    __device__ void gemm(T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
    {
        gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(0u, 1u, alpha, A, B, beta, C);
    }

    /**
     * @brief Single-thread compile-time-size GEMM with implicit `beta = 0`: `C = alpha * op(A) * op(B)`.
     *
     * Overwrites C (the existing C is not read). Otherwise identical to the beta
     * overload above, including the tile4 caveat.
     *
     * @tparam T  Scalar type.
     * @tparam M,N,K  `C` is `M×N`, contraction `K`.
     * @tparam TRANSPOSE_A  If true, `A` is `K×M` and `op(A)=Aᵀ`.
     * @tparam TRANSPOSE_B  If true, `B` is `N×K` and `op(B)=Bᵀ`.
     * @tparam ROW_MAJOR_C  Output storage order (false = column-major).
     * @param alpha  Scalar multiplier on the product.
     * @param A,B    Input matrices.
     * @param C      Output result matrix.
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false>
    __device__ void gemm(T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
    {
        gemm_impl_ct<T, M, N, K, TRANSPOSE_A, TRANSPOSE_B, ROW_MAJOR_C>(0u, 1u, alpha, A, B, C);
    }
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
     * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
     * @param C      In/out result matrix.
     */
    template <typename T, uint32_t M, uint32_t N, uint32_t K,
              bool TRANSPOSE_A = false, bool TRANSPOSE_B = false, bool ROW_MAJOR_C = false>
    __device__ void gemm(T alpha, const T *__restrict__ A, const T *__restrict__ B, T beta, T *__restrict__ C)
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
    __device__ void gemm(T alpha, const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ C)
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
 * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
 * @param C      In/out result matrix.
 * @param s_A    Shared scratch of `m * TILE` elements for the A tile.
 * @param s_B    Shared scratch of `TILE * n` elements for the B tile.
 */
template <typename T, int TILE = 8>
__device__ void gemm_tiled(uint32_t m, uint32_t n, uint32_t k,
                            T alpha, const T *__restrict__ A, const T *__restrict__ B,
                            T beta, T *__restrict__ C,
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
    if (valid) C[crow + ccol*m] = beta_blend(alpha*acc, beta, C[crow + ccol*m]);
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
 * @param beta   Scalar multiplier on the existing C (read only when `beta != 0`).
 * @param C      In/out result matrix.
 * @param s_A    Optional shared scratch for the A tile (nullptr selects the plain path).
 * @param s_B    Optional shared scratch for the B tile.
 */
template <typename T, int TILE = 8>
__device__ void gemm_dispatch(uint32_t m, uint32_t n, uint32_t k,
                               T alpha, const T *__restrict__ A, const T *__restrict__ B,
                               T beta, T *__restrict__ C,
                               T *s_A = nullptr, T *s_B = nullptr)
{
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    if (s_A != nullptr && m*n <= size)
        gemm_tiled<T, TILE>(m, n, k, alpha, A, B, beta, C, s_A, s_B);
    else
        gemm<T>(m, n, k, alpha, A, B, beta, C);
}
