#pragma once

// this file is for triangular matrix vector multiplication
// y = alpha * A * x or alpha * A' * x
// it is assumed that A is ALWAYS SQUARE LOWER triangular
// A is of size n x n, but only its triangular part is used

// by Shaohui Yang, 2024.07.01

template<typename T, bool TRANSPOSE_A = false>
__device__
void trmv(uint32_t n,
          T alpha,
          T *A,
          T *x,
          T *y) {
    T res;
    uint32_t row, col;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    if (TRANSPOSE_A) {
        for (row = ind; row < n; row += stride) {
            res = static_cast<T>(0);
            for (col = row; col < n; col++) {
                res += A[row * n + col] * x[col];
            }
            y[row] = alpha * res;
        }
    } else {
        for (row = ind; row < n; row += stride) {
            res = static_cast<T>(0);
            for (col = 0; col < row + 1; col++) {
                res += A[row + col * n] * x[col];
            }
            y[row] = alpha * res;
        }
    }
}