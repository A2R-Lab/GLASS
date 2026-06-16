#pragma once
#include <cstdint>

/**
 * @file bdmv.cuh
 * @brief Block-tridiagonal matrix-vector product (`glass::banded::bdmv`).
 *
 * Single-block, thread-count-invariant matvec for a block-tridiagonal matrix
 * stored as `NumBlockRows` contiguous strips. Each block-row strip is a
 * `BlockSize x (3*BlockSize)` **row-major** tile laid out `[L | D | R]`
 * (left/diagonal/right `BlockSize x BlockSize` blocks). The block-row at index
 * `br` starts at `s_matrix + br * (3*BlockSize) * BlockSize`.
 *
 * The input/output vectors use the **padded** layout `(NumBlockRows + 2) *
 * BlockSize`: one `BlockSize` pad block on each end. Block-row `br` reads the
 * window `s_vector[br*BlockSize : br*BlockSize + 3*BlockSize)` (= previous /
 * current / next state blocks) and writes its result at
 * `s_output[(br+1)*BlockSize + row]`. Edge block-rows (0 and N-1) work with no
 * special case because their absent `L` / `R` multiply the zero pad — the
 * caller must pre-zero the leading/trailing pad blocks of `s_vector`.
 *
 * No trailing `__syncthreads()` — the caller barriers before reusing the output
 * (matches the rest of the GLASS surface).
 */

namespace banded {

/**
 * @brief Block-tridiagonal matvec: `s_output = A_bd * s_vector`.
 *
 * @tparam T            Scalar type (e.g. `float`, `double`).
 * @tparam NumBlockRows Number of block-rows.
 * @tparam BlockSize    Block dimension (rows/cols of each `L`/`D`/`R` block).
 * @param s_output  Padded output, length `(NumBlockRows+2)*BlockSize`; result
 *                  for block-row `br` written at `(br+1)*BlockSize + row`.
 * @param s_matrix  Block-tridiagonal strips, `[L|D|R]` row-major per block-row.
 * @param s_vector  Padded input, length `(NumBlockRows+2)*BlockSize`.
 */
template <typename T, uint32_t NumBlockRows, uint32_t BlockSize>
__device__ void bdmv(T *s_output, const T *s_matrix, const T *s_vector)
{
    constexpr uint32_t BRL = 3 * BlockSize;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t idx = rank; idx < NumBlockRows * BlockSize; idx += size) {
        uint32_t br  = idx / BlockSize;
        uint32_t row = idx % BlockSize;
        const T *blk = s_matrix + br * BRL * BlockSize;
        const T *vec = s_vector + br * BlockSize;
        T sum = static_cast<T>(0);
        for (uint32_t col = 0; col < BRL; col++)
            sum += blk[row * BRL + col] * vec[col];
        s_output[(br + 1) * BlockSize + row] = sum;
    }
}

/**
 * @brief Block-tridiagonal matvec writing the result to two buffers at once.
 *
 * Computes `A_bd * s_vector` and stores the (identical) result into both
 * `s_output_1` and `s_output_2` in a single pass — e.g. the PCG initialization
 * `z = p = Pinv * r`.
 *
 * @tparam T            Scalar type (e.g. `float`, `double`).
 * @tparam NumBlockRows Number of block-rows.
 * @tparam BlockSize    Block dimension.
 * @param s_output_1  First padded output, length `(NumBlockRows+2)*BlockSize`.
 * @param s_output_2  Second padded output, length `(NumBlockRows+2)*BlockSize`.
 * @param s_matrix    Block-tridiagonal strips, `[L|D|R]` row-major per block-row.
 * @param s_vector    Padded input, length `(NumBlockRows+2)*BlockSize`.
 */
template <typename T, uint32_t NumBlockRows, uint32_t BlockSize>
__device__ void bdmv(T *s_output_1, T *s_output_2, const T *s_matrix, const T *s_vector)
{
    constexpr uint32_t BRL = 3 * BlockSize;
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t idx = rank; idx < NumBlockRows * BlockSize; idx += size) {
        uint32_t br  = idx / BlockSize;
        uint32_t row = idx % BlockSize;
        const T *blk = s_matrix + br * BRL * BlockSize;
        const T *vec = s_vector + br * BlockSize;
        T sum = static_cast<T>(0);
        for (uint32_t col = 0; col < BRL; col++)
            sum += blk[row * BRL + col] * vec[col];
        uint32_t o = (br + 1) * BlockSize + row;
        s_output_1[o] = sum;
        s_output_2[o] = sum;
    }
}

} // namespace banded
