#pragma once
#include <cstdint>

#define _GLASS_RS \
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y; \
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;

/**
 * @brief Element-wise maximum: `c = max(a, b)`.
 *
 * NumPy equivalent: `np.maximum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_max(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b[i]); }

/**
 * @brief Element-wise minimum: `c = min(a, b)`.
 *
 * NumPy equivalent: `np.minimum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_min(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b[i]); }

/**
 * @brief Element-wise less-than: `c = (a < b)`.
 *
 * NumPy equivalent: `c = (a < b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i] < b[i]`, else 0).
 */
template <typename T> __device__ void elementwise_less_than(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b[i]; }

/**
 * @brief Element-wise greater-than: `c = (a > b)`.
 *
 * NumPy equivalent: `c = (a > b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i] > b[i]`, else 0).
 */
template <typename T> __device__ void elementwise_more_than(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] > b[i]; }

/**
 * @brief Element-wise less-than-or-equal: `c = (a <= b)`.
 *
 * NumPy equivalent: `c = (a <= b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i] <= b[i]`, else 0).
 */
template <typename T> __device__ void elementwise_less_than_or_eq(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] <= b[i]; }

/**
 * @brief Element-wise less-than-scalar: `c = (a < b)`.
 *
 * Compares every element of `a` against the scalar `b`. NumPy equivalent:
 * `c = (a < b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar threshold.
 * @param c  Output vector of length `N` (1 where `a[i] < b`, else 0).
 */
template <typename T> __device__ void elementwise_less_than_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] < b; }

/**
 * @brief Element-wise logical AND: `c = (a && b)`.
 *
 * NumPy equivalent: `c = np.logical_and(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where both are nonzero, else 0).
 */
template <typename T> __device__ void elementwise_and(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] && b[i]; }

/**
 * @brief Element-wise logical NOT: `c = !a`.
 *
 * NumPy equivalent: `c = np.logical_not(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param c  Output vector of length `N` (1 where `a[i]` is zero, else 0).
 */
template <typename T> __device__ void elementwise_not(uint32_t N, T *a, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = !a[i]; }

/**
 * @brief Element-wise absolute value: `b = |a|`.
 *
 * NumPy equivalent: `b = np.abs(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_abs(uint32_t N, T *a, T *b)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) b[i] = abs(a[i]); }

/**
 * @brief Element-wise (Hadamard) product: `c = a ⊙ b`.
 *
 * NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_mult(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b[i]; }

/**
 * @brief Element-wise subtraction: `c = a - b`.
 *
 * NumPy equivalent: `c = a - b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_sub(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] - b[i]; }

/**
 * @brief Element-wise addition: `c = a + b`.
 *
 * NumPy equivalent: `c = a + b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_add(uint32_t N, T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] + b[i]; }

/**
 * @brief Element-wise scalar multiply: `c = a * b`.
 *
 * Scales every element of `a` by the scalar `b`. NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar multiplier.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_mult_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b; }

/**
 * @brief Element-wise maximum against a scalar: `c = max(a, b)`.
 *
 * NumPy equivalent: `c = np.maximum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar lower bound.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_max_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b); }

/**
 * @brief Element-wise minimum against a scalar: `c = min(a, b)`.
 *
 * NumPy equivalent: `c = np.minimum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @param N  Number of elements.
 * @param a  Input vector of length `N`.
 * @param b  Scalar upper bound.
 * @param c  Output vector of length `N`.
 */
template <typename T> __device__ void elementwise_min_scalar(uint32_t N, T *a, T b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b); }

// compile-time size overloads
/**
 * @brief Element-wise maximum: `c = max(a, b)`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `np.maximum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N> __device__ void elementwise_max(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = max(a[i], b[i]); }

/**
 * @brief Element-wise minimum: `c = min(a, b)`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `np.minimum(a, b)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N> __device__ void elementwise_min(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = min(a[i], b[i]); }

/**
 * @brief Element-wise absolute value: `b = |a|`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `b = np.abs(a)`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Output vector of length `N`.
 */
template <typename T, uint32_t N> __device__ void elementwise_abs(T *a, T *b)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) b[i] = abs(a[i]); }

/**
 * @brief Element-wise (Hadamard) product: `c = a ⊙ b`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N> __device__ void elementwise_mult(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i]*b[i]; }

/**
 * @brief Element-wise subtraction: `c = a - b`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `c = a - b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N> __device__ void elementwise_sub(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] - b[i]; }

/**
 * @brief Element-wise addition: `c = a + b`, compile-time size.
 *
 * Compile-time-`N` overload. NumPy equivalent: `c = a + b`.
 *
 * @tparam T  Scalar type (e.g. `float`, `double`).
 * @tparam N  Number of elements (compile-time constant).
 * @param a  Input vector of length `N`.
 * @param b  Input vector of length `N`.
 * @param c  Output vector of length `N`.
 */
template <typename T, uint32_t N> __device__ void elementwise_add(T *a, T *b, T *c)
{ _GLASS_RS for (uint32_t i = rank; i < N; i += size) c[i] = a[i] + b[i]; }

#undef _GLASS_RS
