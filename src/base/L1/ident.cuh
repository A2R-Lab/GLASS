#pragma once
#include <cstdint>

/**
 * @brief Load the identity matrix: `A = I_n` (column-major).
 *
 * Writes the `n×n` identity into `A` in column-major order. NumPy equivalent:
 * `A = np.eye(n)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n  Matrix dimension (number of rows/columns).
 * @param A  Output matrix of `n*n` elements (column-major).
 */
template <typename T>
__device__ void loadIdentity(uint32_t n, T *A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n*n; i += size) {
        uint32_t r = i % n, c = i / n;
        A[i] = static_cast<T>(r == c);
    }
}

/**
 * @brief Add a scaled identity to a matrix in place: `A += alpha * I`.
 *
 * Adds `alpha` to the diagonal of the `n×n` (column-major) matrix `A`. NumPy
 * equivalent: `A += alpha * np.eye(n)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param n      Matrix dimension (number of rows/columns).
 * @param A      In/out matrix of `n*n` elements (column-major).
 * @param alpha  Scalar added to each diagonal entry.
 */
template <typename T>
__device__ void addI(uint32_t n, T *A, T alpha)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < n*n; i += size)
        if (i % n == i / n) A[i] += alpha;
}

/**
 * @brief Load the identity matrix: `A = I_N` (column-major), compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `A = np.eye(N)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Matrix dimension (compile-time constant).
 * @param A  Output matrix of `N*N` elements (column-major).
 */
template <typename T, uint32_t N>
__device__ void loadIdentity(T *A)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N*N; i += size) {
        uint32_t r = i % N, c = i / N;
        A[i] = static_cast<T>(r == c);
    }
}

/**
 * @brief Add a scaled identity in place: `A += alpha * I`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `A += alpha * np.eye(N)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Matrix dimension (compile-time constant).
 * @param A      In/out matrix of `N*N` elements (column-major).
 * @param alpha  Scalar added to each diagonal entry.
 */
template <typename T, uint32_t N>
__device__ void addI(T *A, T alpha)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    for (uint32_t i = rank; i < N*N; i += size)
        if (i % N == i / N) A[i] += alpha;
}
