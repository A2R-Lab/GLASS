#pragma once

// A = n x n, n(n+1)/2 entries, lower triangular
// B = n x k, dense matrix
// C = n x k
// C = alpha * A * B or C = alpha * A' * B

template<typename T, bool TRANSPOSE_A = false>
__device__
void trmm_left(uint32_t n,
               uint32_t k,
               T alpha,
               T *A,
               T *B,
               T *C) {
    uint32_t element, row, col;
    T res;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    const unsigned max = n * k;

    if (TRANSPOSE_A) {
        for (element = ind; element < max; element += stride) {
            res = static_cast<T>(0);
            row = element % n;
            col = element / n;

            for (ind = row; ind < n; ind++) {
                uint32_t offset_row = (2 * n - row + 1) * row / 2;
                res += A[offset_row + ind - row] * B[col * n + ind];
                // if A is n x n, then
                // res += A[row * n + ind] * B[col * n + ind];
            }

            C[col * n + row] = alpha * res;
        }
    } else {
        for (element = ind; element < max; element += stride) {
            res = static_cast<T>(0);
            row = element % n;
            col = element / n;

            for (ind = 0; ind < row + 1; ind++) {
                uint32_t offset_ind = (2 * n - ind + 1) * ind / 2;
                res += A[offset_ind + row - ind] * B[col * n + ind];
                // if A is n x n, then
                // res += A[ind * n + row] * B[col * n + ind];
            }

            C[col * n + row] = alpha * res;
        }
    }
}

// A = n x n, n(n+1)/2 entries, lower triangular
// B = k x n, dense matrix
// C = k x n
// C = alpha * B * A or C = alpha * B * A'

template<typename T, bool TRANSPOSE_A = false>
__device__
void trmm_right(uint32_t n,
                uint32_t k,
                T alpha,
                T *A,
                T *B,
                T *C) {
    uint32_t element, row, col;
    T res;
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    const unsigned max = n * k;

    if (TRANSPOSE_A) {
        for (element = ind; element < max; element += stride) {
            res = static_cast<T>(0);
            row = element % k;
            col = element / k;

            for (ind = 0; ind < col + 1; ind++) {
                uint32_t offset_ind = (2 * n - ind + 1) * ind / 2;
                res += B[ind * k + row] * A[offset_ind + col - ind];
                // if A is n x n, then
                // res += B[ind*k + row] * A[ind*n + col];
            }

            C[col * k + row] = alpha * res;
        }
    } else {
        for (element = ind; element < max; element += stride) {
            res = static_cast<T>(0);
            row = element % k;
            col = element / k;

            for (ind = col; ind < n; ind++) {
                uint32_t offset_col = (2 * n - col + 1) * col / 2;
                res += B[ind * k + row] * A[offset_col + ind - col];
                // if A is n x n, then
                // res += B[ind*k + row] * A[col*n + ind];
            }

            C[col * k + row] = alpha * res;
        }
    }
}