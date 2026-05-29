#pragma once
#include <cstdint>

// Gauss-Jordan inversion of an augmented dimA×(2*dimA) matrix in-place.
// Expected layout: column-major [A | I]; on return columns dimA..2*dimA-1 hold A^-1.
// s_temp: (2*dimA+1)*sizeof(T) bytes.
template <typename T>
__device__ void invertMatrix(uint32_t dimA, T *A, T *s_temp)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivOff = pivRC * dimA;
        T pvInv = static_cast<T>(1) / A[pivRC + pivOff];
        for (unsigned ind = rank; ind < 2*dimA+1; ind++) {
            unsigned AInd = (ind < dimA) ? (ind + pivOff) : (pivRC + pivOff + (ind-dimA)*dimA);
            s_temp[ind] = A[AInd];
        }
        __syncthreads();
        for (unsigned ind = rank; ind < dimA*(dimA+1); ind += size) {
            unsigned row = ind % dimA, col = ind / dimA, coff = ind - row;
            if (row == pivRC) A[row + pivOff + coff] *= pvInv;
            else A[row + pivOff + coff] -= s_temp[row]*pvInv*s_temp[dimA+col];
        }
        __syncthreads();
    }
}

template <typename T, uint32_t N>
__device__ void invertMatrix(T *A, T *s_temp)
{
    invertMatrix<T>(N, A, s_temp);
}


// Block-cooperative Gauss-Jordan inversion of a dimA×dimA matrix.
//   A:    in/out, column-major dimA×dimA.  On return: A := A^{-1}.
//   Ainv: workspace, column-major dimA×dimA (overwritten).  On return: A^{-1}.
//          (Some callers want A unchanged + a separate inverse — they get both.)
//   s_temp: shared scratch of size (3*dimA)*sizeof(T).
// Pivot loop is serial (pivot-to-pivot data dependency); within each pivot, the
// dimA save and the dimA*dimA cell update are parallelized across the block.
//
// Algorithm: same dual-update (A, Ainv) Gauss-Jordan as a classic [A | I] →
// [I | A^{-1}] reduction but without materializing the augmented layout. A and
// Ainv are tracked in separate buffers so callers that need A^{-1} alongside
// the original A get it without extra copies.
template <typename T>
__device__ void invertMatrix_dense(uint32_t dimA, T *A, T *Ainv, T *s_temp)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    // Seed Ainv = I (parallel over dimA*dimA cells).
    for (uint32_t ind = rank; ind < dimA*dimA; ind += size) {
        uint32_t row = ind % dimA, col = ind / dimA;
        Ainv[ind] = (row == col) ? static_cast<T>(1) : static_cast<T>(0);
    }
    __syncthreads();
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++) {
        unsigned pivColOffset = pivRC * dimA;
        // Block-cooperative save: pivot row of A, pivot row of Ainv, pivot column of A.
        // 3*dimA entries → block-strided thread distribution.
        for (uint32_t ind = rank; ind < dimA; ind += size) {
            s_temp[ind]            = A[pivRC + dimA * ind];          // pivot row of A
            s_temp[ind + dimA]     = Ainv[pivRC + dimA * ind];        // pivot row of Ainv
            s_temp[ind + dimA * 2] = A[ind + pivColOffset];           // pivot column of A
        }
        __syncthreads();
        T pvInv = static_cast<T>(1) / s_temp[pivRC];
        // Block-cooperative Gauss-Jordan update across dimA*dimA cells.
        // Each thread owns a unique (row, col) so writes are race-free.
        for (uint32_t ind = rank; ind < dimA * dimA; ind += size) {
            uint32_t row = ind % dimA, col = ind / dimA;
            if (row == pivRC) {
                A[row + dimA * col]    = s_temp[col] * pvInv;
                Ainv[row + dimA * col] = s_temp[col + dimA] * pvInv;
            } else {
                T multiplier = s_temp[row + dimA * 2] * pvInv;
                A[row + dimA * col]    -= multiplier * s_temp[col];
                Ainv[row + dimA * col] -= multiplier * s_temp[col + dimA];
            }
        }
        __syncthreads();
    }
}

template <typename T, uint32_t N>
__device__ void invertMatrix_dense(T *A, T *Ainv, T *s_temp)
{
    invertMatrix_dense<T>(N, A, Ainv, s_temp);
}
