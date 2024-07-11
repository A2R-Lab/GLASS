#pragma once

// this file is for triangular matrix vector multiplication
// y = alpha * A * x or alpha * A' * x
// it is assumed that A is ALWAYS SQUARE LOWER triangular
// A is of size n(n+1)/2, column major

// by Shaohui Yang
// version 1: 2024.07.01, A of size n x n.
// version 1: 2024.07.11, A of size n(n+1)/2.

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
                uint32_t offset_row = (2 * n - row + 1) * row / 2;
                res += A[offset_row + col - row] * x[col];
                // if A is n x n, then
                // res += A[row * n + col] * x[col];
            }
            y[row] = alpha * res;
        }
    } else {
        for (row = ind; row < n; row += stride) {
            res = static_cast<T>(0);
            for (col = 0; col < row + 1; col++) {
                uint32_t offset_col = (2 * n - col + 1) * col / 2;
                res += A[row - col + offset_col] * x[col];
                // if A is n x n, then
                // res += A[row + col * n] * x[col];
            }
            y[row] = alpha * res;
        }
    }
}