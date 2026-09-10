#pragma once

// dimm — diagonal-matrix × dense-matrix multiply (cuBLAS `dgmm` analogue).
// Originally contributed by Seyoung Yang (@syangav) on the 2024 `yang` branch;
// modernized here to the single-block conventions (column-major, flat-strided,
// thread-count invariant, TRAILING_SYNC).

/**
 * @brief Diagonal-matrix scale of a dense matrix: `C = alpha * diag(d) * B`
 *        (RIGHT=false, scales ROWS) or `C = alpha * B * diag(d)` (RIGHT=true,
 *        scales COLUMNS). cuBLAS analogue: `cublasXdgmm`.
 *
 * `B` and `C` are `m×n` column-major; `d` holds the diagonal only — length `m`
 * when RIGHT=false, length `n` when RIGHT=true. Pure elementwise: each output
 * is owned by exactly one thread, so `C` may alias `B` (in-place scale).
 * NumPy: `C = alpha * np.diag(d) @ B` / `C = alpha * B @ np.diag(d)`.
 *
 * @tparam T      Scalar type.
 * @tparam RIGHT  false: rows scaled by `d[row]` (diag on the left, default);
 *                true: columns scaled by `d[col]` (diag on the right).
 * @tparam TRAILING_SYNC  End on a barrier so `C` is valid for every thread on
 *                return (default true); callers owning the next barrier pass
 *                false to elide it.
 * @param m,n    Dimensions (`B`/`C` are `m×n`).
 * @param alpha  Scalar multiplier.
 * @param d      Diagonal entries (length `m` or `n` per RIGHT; read-only).
 * @param B      Input matrix (column-major; read-only, may alias `C`).
 * @param C      Output matrix (column-major).
 */
template <typename T, bool RIGHT = false, bool TRAILING_SYNC = true>
__device__ void dimm(uint32_t m, uint32_t n, T alpha, const T *d, const T *B, T *C)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    for (uint32_t idx = rank; idx < m * n; idx += size) {
        uint32_t row = idx % m, col = idx / m;
        C[idx] = alpha * d[RIGHT ? col : row] * B[idx];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}

/**
 * @brief Compile-time-size diagonal-matrix scale (see the runtime `dimm`).
 *
 * @tparam T      Scalar type.
 * @tparam M,N    Dimensions (`B`/`C` are `M×N`).
 * @tparam RIGHT  false: `C = alpha*diag(d)*B`; true: `C = alpha*B*diag(d)`.
 * @tparam TRAILING_SYNC  End on a barrier (default true).
 * @param alpha  Scalar multiplier.
 * @param d      Diagonal entries (length `M` or `N` per RIGHT; read-only).
 * @param B      Input matrix (column-major; read-only, may alias `C`).
 * @param C      Output matrix (column-major).
 */
template <typename T, uint32_t M, uint32_t N, bool RIGHT = false, bool TRAILING_SYNC = true>
__device__ void dimm(T alpha, const T *d, const T *B, T *C)
{
    uint32_t rank = flat_rank();
    uint32_t size = flat_size();
    #pragma unroll
    for (uint32_t idx = rank; idx < M * N; idx += size) {
        uint32_t row = idx % M, col = idx / M;
        C[idx] = alpha * d[RIGHT ? col : row] * B[idx];
    }
    if constexpr (TRAILING_SYNC) __syncthreads();
}
