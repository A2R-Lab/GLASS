#pragma once
#include <cstdint>

// Per-thread compile-time strided dot product.
// Computes sum(x[i*SX] * y[i*SY]) for i in 0..N-1.
// No block-wide reduction — intended for use inside already-parallelized loops
// (e.g., GRiD generated kernels where the outer loop is already thread-parallel).
// With SX=SY=1 this is a plain scalar dot.  The inner loop is fully unrolled
// by the compiler since N, SX, SY are all compile-time constants.

/**
 * @brief Per-thread strided inner product: `Σ x[i*SX] * y[i*SY]` (DOT, strided).
 *
 * A single thread independently walks the full `N`-length product with
 * compile-time strides, returning the scalar result. There is NO block-wide
 * reduction — intended for use inside an already thread-parallel outer loop
 * (e.g. GRiD generated kernels). With `SX = SY = 1` this is a plain scalar dot;
 * the inner loop is fully unrolled. NumPy equivalent: `np.dot(x[::SX], y[::SY])`.
 *
 * @tparam T   Scalar type (e.g. `float`, `double`).
 * @tparam N   Number of products to accumulate (compile-time constant).
 * @tparam SX  Element stride into `x` (compile-time, default 1).
 * @tparam SY  Element stride into `y` (compile-time, default 1).
 * @param x  Input vector, accessed at indices `0, SX, 2*SX, …`.
 * @param y  Input vector, accessed at indices `0, SY, 2*SY, …`.
 * @return The strided inner product `Σ x[i*SX] * y[i*SY]`.
 */
template <typename T, uint32_t N, uint32_t SX = 1, uint32_t SY = 1>
__device__ T dot_strided(const T* x, const T* y)
{
    T res = static_cast<T>(0);
    for (uint32_t i = 0; i < N; i++)
        res += x[i * SX] * y[i * SY];
    return res;
}

/**
 * @brief Per-thread strided inner product, store-to-pointer overload.
 *
 * Same per-thread strided dot as the value-returning overload, but writes the
 * scalar result to `*out`. NumPy equivalent: `out[0] = np.dot(x[::SX], y[::SY])`.
 *
 * @tparam T   Scalar type (e.g. `float`, `double`).
 * @tparam N   Number of products to accumulate (compile-time constant).
 * @tparam SX  Element stride into `x` (compile-time, default 1).
 * @tparam SY  Element stride into `y` (compile-time, default 1).
 * @param x    Input vector, accessed at indices `0, SX, 2*SX, …`.
 * @param y    Input vector, accessed at indices `0, SY, 2*SY, …`.
 * @param out  Destination for the scalar result.
 */
template <typename T, uint32_t N, uint32_t SX = 1, uint32_t SY = 1>
__device__ void dot_strided(const T* x, const T* y, T* out)
{
    *out = dot_strided<T, N, SX, SY>(x, y);
}
