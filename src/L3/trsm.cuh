#pragma once

// this file is for triangular matrix solve AX = B, X = inv(A)*B
// X will be in place of B
// it is assumed that A is ALWAYS SQUARE LOWER triangular
// A is of size n x n, but only its triangular part is used
// B is of size n x m

// by Shaohui Yang, 2024.07.01

template<typename T, bool TRANSPOSE_A = false>
__device__
void trsm(uint32_t n,
          uint32_t m,
          T *A,
          T *B) {
    T sum;
    uint32_t col; // parallel dimension
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    if (TRANSPOSE_A) {
        for (col = ind; col < m; col += stride) {
            // solve A x = b, b is one column of B
            // backward substitution
            for (int32_t i = n - 1; i >= 0; i--) {
                sum = static_cast<T>(0);
                for (uint32_t j = i + 1; j < n; j++) {
                    sum += A[i * n + j] * B[col * n + j];
                }
                B[col * n + i] = (B[col * n + i] - sum) / A[i * n + i];
            }
        }
    } else {
        for (col = ind; col < m; col += stride) {
            // solve A x = b, b is one column of B
            // forward substitution
            for (uint32_t i = 0; i < n; i++) {
                sum = static_cast<T>(0);
                for (uint32_t j = 0; j < i; j++) {
                    sum += A[j * n + i] * B[col * n + j];
                }
                B[col * n + i] = (B[col * n + i] - sum) / A[i * n + i];
            }
        }
    }

}