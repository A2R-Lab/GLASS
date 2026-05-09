#pragma once
#include <cstdint>

// Gauss-Jordan inversion of dimA×dimA matrix A in-place.
// s_temp: (2*dimA+1)*sizeof(T) bytes
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
