#pragma once
#include <cstdint>

// FillMode selects which triangle of the symmetric result C is written:
//   Lower — only cells with row >= col
//   Upper — only cells with row <= col
//   Full  — both triangles (C is materialized as a full symmetric matrix)
// (file-scope so it lands in `namespace glass` via the glass.cuh include trick).
enum class FillMode : uint32_t { Lower = 0, Upper = 1, Full = 2 };

// ─── helpers: is this (row,col) cell in the canonical (computed) triangle? ────
// Lower/Full compute the lower triangle (row>=col); Upper computes row<=col.
// For Full we ALSO materialize the mirror, but only the lower-owning thread
// writes it — see the write phase below.
__device__ __forceinline__ bool syrk_in_canonical(FillMode fill, uint32_t row, uint32_t col)
{
    return (fill == FillMode::Upper) ? (row <= col) : (row >= col);
}

// ─── syrk core impl: explicit rank/size + layout flags ───────────────────────
// C = alpha * op(A) * op(A)^T + beta * C, C is n x n symmetric.
//   TRANS == false: op(A) = A   (A is n x k) → C = alpha*A*A^T + beta*C
//   TRANS == true : op(A) = A^T (A is k x n) → C = alpha*A^T*A + beta*C
// Flat-element parallelism over the n*n grid exactly like gemm_impl: each thread
// owns disjoint output cells, so NO interior barrier is needed (guide §1a
// counter-note). The k-loop runs only in the canonical triangle (the symmetry
// win); off-triangle cells are filled by the mirror write of their transpose.
template <typename T, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syrk_impl(uint32_t rank, uint32_t size,
                          uint32_t n, uint32_t k,
                          T alpha, T *A, T beta, T *C)
{
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;  // mirror handles it
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            // TRANS=false: A is n x k → A[row,i], A[col,i].
            // TRANS=true : A is k x n → A[i,row], A[i,col].
            T ar, ac;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

// beta = 0 form: never reads C.
template <typename T, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syrk_impl(uint32_t rank, uint32_t size,
                          uint32_t n, uint32_t k,
                          T alpha, T *A, T *C)
{
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            T ar, ac;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res;
        }
    }
}

// compile-time impl: N, K as template params so el%N / el/N use magic-number
// multiply instead of MUFU.RCP (mirrors gemm_impl_ct).
template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syrk_impl_ct(uint32_t rank, uint32_t size,
                             T alpha, T *A, T beta, T *C)
{
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syrk_impl_ct(uint32_t rank, uint32_t size,
                             T alpha, T *A, T *C)
{
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
            }
            res += ar * ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res;
        }
    }
}

// ─── syr2k core impl: explicit rank/size + layout flags ──────────────────────
// C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C, C n x n symmetric.
// Symmetric by construction: cell (row,col) is the symmetric dot
//   Σ_i ( a(row,i)*b(col,i) + b(row,i)*a(col,i) )  [TRANS=false reading semantics]
// which equals cell (col,row), so the same mirror-write trick applies.
template <typename T, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syr2k_impl(uint32_t rank, uint32_t size,
                           uint32_t n, uint32_t k,
                           T alpha, T *A, T *B, T beta, T *C)
{
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            T ar, ac, br, bc;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
                br = ROW_MAJOR ? B[i*n + row] : B[row*k + i];
                bc = ROW_MAJOR ? B[i*n + col] : B[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
                br = ROW_MAJOR ? B[row*k + i] : B[i*n + row];
                bc = ROW_MAJOR ? B[col*k + i] : B[i*n + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

template <typename T, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syr2k_impl(uint32_t rank, uint32_t size,
                           uint32_t n, uint32_t k,
                           T alpha, T *A, T *B, T *C)
{
    const uint32_t maxel = n * n;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % n, col = el / n;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < k; i++) {
            T ar, ac, br, bc;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*n + row] : A[row*k + i];
                ac = ROW_MAJOR ? A[i*n + col] : A[col*k + i];
                br = ROW_MAJOR ? B[i*n + row] : B[row*k + i];
                bc = ROW_MAJOR ? B[i*n + col] : B[col*k + i];
            } else {
                ar = ROW_MAJOR ? A[row*k + i] : A[i*n + row];
                ac = ROW_MAJOR ? A[col*k + i] : A[i*n + col];
                br = ROW_MAJOR ? B[row*k + i] : B[i*n + row];
                bc = ROW_MAJOR ? B[col*k + i] : B[i*n + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*n + col) : (col*n + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*n + row) : (row*n + col);
            C[midx] = alpha*res;
        }
    }
}

template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syr2k_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, T *A, T *B, T beta, T *C)
{
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac, br, bc;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
                br = ROW_MAJOR ? B[i*N + row] : B[row*K + i];
                bc = ROW_MAJOR ? B[i*N + col] : B[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
                br = ROW_MAJOR ? B[row*K + i] : B[i*N + row];
                bc = ROW_MAJOR ? B[col*K + i] : B[i*N + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res + beta*C[cidx];
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res + beta*C[midx];
        }
    }
}

template <typename T, uint32_t N, uint32_t K, FillMode FILL, bool TRANS, bool ROW_MAJOR>
__device__ void syr2k_impl_ct(uint32_t rank, uint32_t size,
                              T alpha, T *A, T *B, T *C)
{
    constexpr uint32_t maxel = N * N;
    for (uint32_t el = rank; el < maxel; el += size) {
        uint32_t row = el % N, col = el / N;
        if (!syrk_in_canonical(FILL, row, col)) continue;
        T res = static_cast<T>(0);
        for (uint32_t i = 0; i < K; i++) {
            T ar, ac, br, bc;
            if (TRANS) {
                ar = ROW_MAJOR ? A[i*N + row] : A[row*K + i];
                ac = ROW_MAJOR ? A[i*N + col] : A[col*K + i];
                br = ROW_MAJOR ? B[i*N + row] : B[row*K + i];
                bc = ROW_MAJOR ? B[i*N + col] : B[col*K + i];
            } else {
                ar = ROW_MAJOR ? A[row*K + i] : A[i*N + row];
                ac = ROW_MAJOR ? A[col*K + i] : A[i*N + col];
                br = ROW_MAJOR ? B[row*K + i] : B[i*N + row];
                bc = ROW_MAJOR ? B[col*K + i] : B[i*N + col];
            }
            res += ar*bc + br*ac;
        }
        uint32_t cidx = ROW_MAJOR ? (row*N + col) : (col*N + row);
        C[cidx] = alpha*res;
        if (FILL == FillMode::Full && row != col) {
            uint32_t midx = ROW_MAJOR ? (col*N + row) : (row*N + col);
            C[midx] = alpha*res;
        }
    }
}

// ─── syrk runtime variants ───────────────────────────────────────────────────

/**
 * @brief Symmetric rank-k update: `C = alpha * op(A) * op(A)^T + beta * C` (SYRK).
 *
 * Runtime-size, single-block, flat-element parallelism: each thread owns output
 * cells of the n x n symmetric `C` strided over the block. The length-k dot is
 * computed ONLY in the canonical triangle (the symmetry win, ~half the FLOPs of
 * a GEMM); for `Full` the lower-cell-owning thread also writes the mirror
 * `C[col,row]` (diagonal written once) so each cell is written exactly once and
 * NO interior barrier is needed. `Lower`/`Upper` write only the named triangle
 * and leave the other untouched.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op(A)=A (A is n x k); if true, op(A)=A^T (A is k x n).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (n x k if TRANS=false, else k x n).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out n x n symmetric result matrix.
 *
 * NumPy equivalent: TRANS=false → `alpha * A @ A.T + beta * C`;
 * TRANS=true → `alpha * A.T @ A + beta * C`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syrk(uint32_t n, uint32_t k, T alpha, T *A, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl<T, FILL, TRANS, ROW_MAJOR>(rank, size, n, k, alpha, A, beta, C);
}

/**
 * @brief SYRK with implicit `beta = 0`: `C = alpha * op(A) * op(A)^T` (SYRK).
 *
 * Runtime-size overload that overwrites C (C is overwritten, not read), avoiding
 * the `beta * C` term — safe to write into uninitialized scratch. For `Full`,
 * the full symmetric matrix is written; for `Lower`/`Upper`, only the named
 * triangle is written and the other is left untouched. Single-block,
 * flat-element parallelism; no interior barrier.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op(A)=A (A is n x k); if true, op(A)=A^T (A is k x n).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (n x k if TRANS=false, else k x n).
 * @param C  Output n x n symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANS=false → `alpha * A @ A.T`; TRANS=true → `alpha * A.T @ A`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syrk(uint32_t n, uint32_t k, T alpha, T *A, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl<T, FILL, TRANS, ROW_MAJOR>(rank, size, n, k, alpha, A, C);
}

// ─── syrk compile-time size variants ─────────────────────────────────────────

/**
 * @brief Compile-time-size SYRK: `C = alpha * op(A) * op(A)^T + beta * C` (SYRK).
 *
 * Dimensions are template parameters so the compiler unrolls the inner loop and
 * replaces the `el % N` / `el / N` index math with magic-number multiplies.
 * Single-block, flat-element parallelism, symmetry-exploiting (canonical
 * triangle + mirror write), no interior barrier. C is read; caller must
 * initialize it.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op(A)=A (A is N x K); if true, op(A)=A^T (A is K x N).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (N x K if TRANS=false, else K x N).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out N x N symmetric result matrix.
 *
 * NumPy equivalent: TRANS=false → `alpha * A @ A.T + beta * C`;
 * TRANS=true → `alpha * A.T @ A + beta * C`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syrk(T alpha, T *A, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl_ct<T, N, K, FILL, TRANS, ROW_MAJOR>(rank, size, alpha, A, beta, C);
}

/**
 * @brief Compile-time-size SYRK with implicit `beta = 0`: `C = alpha * op(A) * op(A)^T`.
 *
 * Compile-time-size overload that overwrites C (C is overwritten, not read).
 * Single-block, flat-element parallelism, symmetry-exploiting; no interior
 * barrier. Safe to write into uninitialized scratch.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op(A)=A (A is N x K); if true, op(A)=A^T (A is K x N).
 * @tparam ROW_MAJOR  Storage order for A and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the product.
 * @param A  Input matrix (N x K if TRANS=false, else K x N).
 * @param C  Output N x N symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANS=false → `alpha * A @ A.T`; TRANS=true → `alpha * A.T @ A`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syrk(T alpha, T *A, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syrk_impl_ct<T, N, K, FILL, TRANS, ROW_MAJOR>(rank, size, alpha, A, C);
}

// ─── syr2k runtime variants ──────────────────────────────────────────────────

/**
 * @brief Symmetric rank-2k update: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C` (SYR2K).
 *
 * Runtime-size, single-block, flat-element parallelism. The result is symmetric
 * by construction; the length-k dot is computed only in the canonical triangle
 * and (for `Full`) mirrored, so each cell is written once and NO interior
 * barrier is needed. `Lower`/`Upper` write only the named triangle.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op = identity (A,B are n x k); if true, op = transpose (A,B are k x n).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (n x k if TRANS=false, else k x n).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out n x n symmetric result matrix.
 *
 * NumPy equivalent: TRANS=false → `alpha*(A@B.T + B@A.T) + beta*C`;
 * TRANS=true → `alpha*(A.T@B + B.T@A) + beta*C`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syr2k(uint32_t n, uint32_t k, T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl<T, FILL, TRANS, ROW_MAJOR>(rank, size, n, k, alpha, A, B, beta, C);
}

/**
 * @brief SYR2K with implicit `beta = 0`: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T)`.
 *
 * Runtime-size overload that overwrites C (C is overwritten, not read). Safe to
 * write into uninitialized scratch. Single-block, flat-element parallelism,
 * symmetry-exploiting; no interior barrier.
 *
 * @tparam T  Scalar type.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op = identity (A,B are n x k); if true, op = transpose (A,B are k x n).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param n  Dimension of the symmetric result C (n x n).
 * @param k  Contraction length.
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (n x k if TRANS=false, else k x n).
 * @param C  Output n x n symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANS=false → `alpha*(A@B.T + B@A.T)`;
 * TRANS=true → `alpha*(A.T@B + B.T@A)`.
 */
template <typename T, FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syr2k(uint32_t n, uint32_t k, T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl<T, FILL, TRANS, ROW_MAJOR>(rank, size, n, k, alpha, A, B, C);
}

// ─── syr2k compile-time size variants ────────────────────────────────────────

/**
 * @brief Compile-time-size SYR2K: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C`.
 *
 * Dimensions are template parameters so the inner loop unrolls and `el % N` /
 * `el / N` become magic-number multiplies. Single-block, flat-element
 * parallelism, symmetry-exploiting; no interior barrier. C is read; caller must
 * initialize it.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op = identity (A,B are N x K); if true, op = transpose (A,B are K x N).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (N x K if TRANS=false, else K x N).
 * @param beta  Scalar multiplier on the existing C (C is read; caller must initialize it).
 * @param C  In/out N x N symmetric result matrix.
 *
 * NumPy equivalent: TRANS=false → `alpha*(A@B.T + B@A.T) + beta*C`;
 * TRANS=true → `alpha*(A.T@B + B.T@A) + beta*C`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syr2k(T alpha, T *A, T *B, T beta, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl_ct<T, N, K, FILL, TRANS, ROW_MAJOR>(rank, size, alpha, A, B, beta, C);
}

/**
 * @brief Compile-time-size SYR2K with implicit `beta = 0`: `C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T)`.
 *
 * Compile-time-size overload that overwrites C (C is overwritten, not read).
 * Safe to write into uninitialized scratch. Single-block, flat-element
 * parallelism, symmetry-exploiting; no interior barrier.
 *
 * @tparam T  Scalar type.
 * @tparam N  Compile-time dimension of the symmetric result C (N x N).
 * @tparam K  Compile-time contraction length.
 * @tparam FILL  Which triangle of C to write (Lower / Upper / Full).
 * @tparam TRANS  If false, op = identity (A,B are N x K); if true, op = transpose (A,B are K x N).
 * @tparam ROW_MAJOR  Storage order for A, B and C (false = column-major / Fortran).
 * @param alpha  Scalar multiplier on the symmetrized product.
 * @param A,B  Input matrices (N x K if TRANS=false, else K x N).
 * @param C  Output N x N symmetric result matrix (overwritten, not read).
 *
 * NumPy equivalent: TRANS=false → `alpha*(A@B.T + B@A.T)`;
 * TRANS=true → `alpha*(A.T@B + B.T@A)`.
 */
template <typename T, uint32_t N, uint32_t K,
          FillMode FILL = FillMode::Full, bool TRANS = false, bool ROW_MAJOR = false>
__device__ void syr2k(T alpha, T *A, T *B, T *C)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    syr2k_impl_ct<T, N, K, FILL, TRANS, ROW_MAJOR>(rank, size, alpha, A, B, C);
}
