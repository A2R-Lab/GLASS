#pragma once

// A represents n x n matrix, n entries, diagonal
// B = n x k, dense matrix
// C = n x k
// C = alpha * A * B

template<typename T>
__device__
void dimm_left(uint32_t n,
               uint32_t k,
               T alpha,
               T *A,
               T *B,
               T *C) {
    uint32_t element, row, col;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    const unsigned max = n * k;

    for (element = ind; element < max; element += stride) {
        row = element % n;
        col = element / n;

        C[col * n + row] = alpha * A[row] * B[col * n + row];
    }
}

// A represents n x n matrix, n entries, diagonal
// B = k x n, dense matrix
// C = k x n
// C = alpha * B * A

template<typename T>
__device__
void dimm_right(uint32_t n,
                uint32_t k,
                T alpha,
                T *A,
                T *B,
                T *C) {
    uint32_t element, row, col;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    const unsigned max = n * k;

    for (element = ind; element < max; element += stride) {
        row = element % k;
        col = element / k;

        C[col * k + row] = alpha * A[col] * B[col * k + row];
    }
}