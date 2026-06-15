#pragma once
#include <cstdint>
#include <math.h>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// L1 cgrps variants — same semantics as glass:: but thread rank/size from the group

/**
 * @brief AXPY: `y = alpha*x + y` (cooperative-groups variant).
 *
 * In-place vector update over the thread group. NumPy equivalent: `y += alpha*x`.
 *
 * @tparam T  Scalar type.
 * @param n      Vector length.
 * @param alpha  Scalar multiplier on x.
 * @param x      Input vector.
 * @param y      In/out vector (accumulated into).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = alpha*x[i] + y[i];
}

/**
 * @brief Out-of-place AXPY: `z = alpha*x + y` (cooperative-groups variant).
 *
 * NumPy equivalent: `z = alpha*x + y`.
 *
 * @tparam T  Scalar type.
 * @param n      Vector length.
 * @param alpha  Scalar multiplier on x.
 * @param x,y    Input vectors.
 * @param z      Output vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y, T *z,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) z[i] = alpha*x[i] + y[i];
}

/**
 * @brief AXPBY: `z = alpha*x + beta*y` (cooperative-groups variant).
 *
 * NumPy equivalent: `z = alpha*x + beta*y`.
 *
 * @tparam T  Scalar type.
 * @param n      Vector length.
 * @param alpha  Scalar multiplier on x.
 * @param x      Input vector.
 * @param beta   Scalar multiplier on y.
 * @param y      Input vector.
 * @param z      Output vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void axpby(uint32_t n, T alpha, T *x, T beta, T *y, T *z,
                      cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) z[i] = alpha*x[i] + beta*y[i];
}

/**
 * @brief Compile-time-size AXPY: `y = alpha*x + y` (cooperative-groups variant).
 *
 * NumPy equivalent: `y += alpha*x`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Vector length.
 * @param alpha  Scalar multiplier on x.
 * @param x      Input vector.
 * @param y      In/out vector (accumulated into).
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) y[i] = alpha*x[i] + y[i];
}

/**
 * @brief Compile-time-size out-of-place AXPY: `z = alpha*x + y` (cooperative-groups variant).
 *
 * NumPy equivalent: `z = alpha*x + y`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Vector length.
 * @param alpha  Scalar multiplier on x.
 * @param x,y    Input vectors.
 * @param z      Output vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y, T *z,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) z[i] = alpha*x[i] + y[i];
}

/**
 * @brief Vector copy: `y = x` (cooperative-groups variant).
 *
 * NumPy equivalent: `y = x.copy()`.
 *
 * @tparam T  Scalar type.
 * @param n  Vector length.
 * @param x  Source vector.
 * @param y  Destination vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void copy(uint32_t n, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = x[i];
}

/**
 * @brief Scaled copy: `y = alpha*x` (cooperative-groups variant).
 *
 * NumPy equivalent: `y = alpha*x`.
 *
 * @tparam T  Scalar type.
 * @param n      Vector length.
 * @param alpha  Scalar multiplier on x.
 * @param x      Source vector.
 * @param y      Destination vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void copy(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = alpha*x[i];
}

/**
 * @brief Compile-time-size vector copy: `y = x` (cooperative-groups variant).
 *
 * NumPy equivalent: `y = x.copy()`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Vector length.
 * @param x  Source vector.
 * @param y  Destination vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t N>
__device__ void copy(T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) y[i] = x[i];
}

/**
 * @brief In-place scale: `x = alpha*x` (cooperative-groups variant).
 *
 * NumPy equivalent: `x *= alpha`.
 *
 * @tparam T  Scalar type.
 * @param n      Vector length.
 * @param alpha  Scalar multiplier.
 * @param x      In/out vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void scal(uint32_t n, T alpha, T *x,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) x[i] = alpha*x[i];
}

/**
 * @brief Out-of-place scale: `y = alpha*x` (cooperative-groups variant).
 *
 * NumPy equivalent: `y = alpha*x`.
 *
 * @tparam T  Scalar type.
 * @param n      Vector length.
 * @param alpha  Scalar multiplier.
 * @param x      Source vector.
 * @param y      Destination vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void scal(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = alpha*x[i];
}

/**
 * @brief Compile-time-size in-place scale: `x = alpha*x` (cooperative-groups variant).
 *
 * NumPy equivalent: `x *= alpha`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Vector length.
 * @param alpha  Scalar multiplier.
 * @param x      In/out vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t N>
__device__ void scal(T alpha, T *x,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) x[i] = alpha*x[i];
}

/**
 * @brief Swap two vectors element-wise: `x <-> y` (cooperative-groups variant).
 *
 * @tparam T  Scalar type.
 * @param n  Vector length.
 * @param x  In/out vector (swapped with y).
 * @param y  In/out vector (swapped with x).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void swap(uint32_t n, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}

/**
 * @brief Compile-time-size swap of two vectors: `x <-> y` (cooperative-groups variant).
 *
 * @tparam T  Scalar type.
 * @tparam N  Vector length.
 * @param x  In/out vector (swapped with y).
 * @param y  In/out vector (swapped with x).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t N>
__device__ void swap(T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}

/**
 * @brief Element-wise clamp in place: `x = clamp(x, l, u)` (cooperative-groups variant).
 *
 * Lower/upper bounds are per-element vectors. NumPy equivalent: `x = np.clip(x, l, u)`.
 *
 * @tparam T  Scalar type.
 * @param n  Vector length.
 * @param x  In/out vector (clamped).
 * @param l  Per-element lower bounds.
 * @param u  Per-element upper bounds.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void clip(uint32_t n, T *x, T *l, T *u,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size())
        x[i] = max(l[i], min(x[i], u[i]));
}

/**
 * @brief Fill a vector with a constant: `x = [alpha, ...]` (cooperative-groups variant).
 *
 * NumPy equivalent: `x = np.full(n, alpha)`.
 *
 * @tparam T  Scalar type.
 * @param n      Vector length.
 * @param alpha  Fill value.
 * @param x      Output vector.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void set_const(uint32_t n, T alpha, T *x,
                           cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) x[i] = alpha;
}

/**
 * @brief Load the identity matrix: `A = I_n` (column-major, cooperative-groups variant).
 *
 * NumPy equivalent: `A = np.eye(n)`.
 *
 * @tparam T  Scalar type.
 * @param n  Matrix dimension (A is n x n).
 * @param A  Output matrix (column-major), set to the identity.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void loadIdentity(uint32_t n, T *A,
                              cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n*n; i += g.size()) {
        uint32_t r = i % n, c = i / n;
        A[i] = static_cast<T>(r == c);
    }
}

/**
 * @brief Add a scaled identity to a matrix in place: `A += alpha*I` (cooperative-groups variant).
 *
 * Only the diagonal is modified. NumPy equivalent: `A += alpha*np.eye(n)`.
 *
 * @tparam T  Scalar type.
 * @param n      Matrix dimension (A is n x n, column-major).
 * @param A      In/out matrix.
 * @param alpha  Scalar added to each diagonal element.
 * @param g      Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void addI(uint32_t n, T *A, T alpha,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n*n; i += g.size())
        if (i % n == i / n) A[i] += alpha;
}

// reductions
/**
 * @brief Sum reduction in place: `x[0] = sum(x)` (cooperative-groups variant).
 *
 * Halving (tree) reduction over the group; the result lands in `x[0]` and the
 * rest of `x` is overwritten. NumPy equivalent: `x[0] = np.sum(x)`.
 *
 * @tparam T  Scalar type.
 * @param n  Vector length.
 * @param x  In/out vector; on return `x[0]` holds the sum.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void reduce(uint32_t n, T *x,
                       cgrps::thread_group g = cgrps::this_thread_block())
{
    uint32_t rank = g.thread_rank(), size = g.size();
    uint32_t left = n;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += size) x[i] += x[i + left];
        if (rank == 0 && odd) x[0] += x[2*left];
        g.sync();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] += x[i]; }
}

/**
 * @brief Compile-time-size sum reduction: `x[0] = sum(x)` (cooperative-groups variant).
 *
 * NumPy equivalent: `x[0] = np.sum(x)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Vector length.
 * @param x  In/out vector; on return `x[0]` holds the sum.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t N>
__device__ void reduce(T *x, cgrps::thread_group g = cgrps::this_thread_block())
{
    reduce<T>(N, x, g);
}

/**
 * @brief Dot product: `y[0] = dot(x, y)` (cooperative-groups variant).
 *
 * Destructive: `y` is used as scratch — it is multiplied element-wise by `x` and
 * then reduced, leaving the result in `y[0]`. NumPy equivalent:
 * `y[0] = np.dot(x, y)`.
 *
 * @tparam T  Scalar type.
 * @param n  Vector length.
 * @param x  Input vector.
 * @param y  In/out vector (overwritten; on return `y[0]` holds the dot product).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void dot(uint32_t n, T *x, T *y,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] *= x[i];
    g.sync();
    reduce<T>(n, y, g);
}

/**
 * @brief Compile-time-size dot product: `y[0] = dot(x, y)` (cooperative-groups variant).
 *
 * Destructive: `y` is overwritten and the result lands in `y[0]`. NumPy
 * equivalent: `y[0] = np.dot(x, y)`.
 *
 * @tparam T  Scalar type.
 * @tparam N  Vector length.
 * @param x  Input vector.
 * @param y  In/out vector (overwritten; on return `y[0]` holds the dot product).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T, uint32_t N>
__device__ void dot(T *x, T *y,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) y[i] *= x[i];
    g.sync();
    reduce<T, N>(y, g);
}

/**
 * @brief Out-of-place transpose: `b = a^T` (column-major, cooperative-groups variant).
 *
 * `a` is N x M, `b` is M x N. NumPy equivalent: `b = a.T`.
 *
 * @tparam T  Scalar type.
 * @param N  Rows of a (columns of b).
 * @param M  Columns of a (rows of b).
 * @param a  Input matrix (column-major, N x M).
 * @param b  Output matrix (column-major, M x N).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void transpose(uint32_t N, uint32_t M, T *a, T *b,
                           cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N*M; i += g.size()) {
        uint32_t col = i / N, row = i % N;
        b[col + M*row] = a[row + N*col];
    }
    g.sync();
}

/**
 * @brief In-place square transpose: `a = a^T` for an N x N matrix (cooperative-groups variant).
 *
 * Swaps the strict upper/lower triangles in place.
 *
 * @tparam T  Scalar type.
 * @param N  Matrix dimension (a is N x N, column-major).
 * @param a  In/out matrix, transposed in place.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void transpose(uint32_t N, T *a,
                           cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t idx = g.thread_rank(); idx < N*N; idx += g.size()) {
        uint32_t i = idx % N, j = idx / N;
        if (i < j) {
            uint32_t sw = i*N + j;
            T tmp = a[idx]; a[idx] = a[sw]; a[sw] = tmp;
        }
    }
    g.sync();
}

/**
 * @brief Element-wise maximum: `c = max(a, b)` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = np.maximum(a, b)`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_max(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = max(a[i], b[i]); }

/**
 * @brief Element-wise minimum: `c = min(a, b)` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = np.minimum(a, b)`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_min(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = min(a[i], b[i]); }

/**
 * @brief Element-wise less-than: `c = (a < b)` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = a < b`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector (1/0 truth values).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_less_than(uint32_t N, T *a, T *b, T *c,
                                       cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] < b[i]; }

/**
 * @brief Element-wise greater-than: `c = (a > b)` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = a > b`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector (1/0 truth values).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_more_than(uint32_t N, T *a, T *b, T *c,
                                       cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] > b[i]; }

/**
 * @brief Element-wise less-than-or-equal: `c = (a <= b)` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = a <= b`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector (1/0 truth values).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_less_than_or_eq(uint32_t N, T *a, T *b, T *c,
                                              cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] <= b[i]; }

/**
 * @brief Element-wise logical AND: `c = (a && b)` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = np.logical_and(a, b)`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector (1/0 truth values).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_and(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] && b[i]; }

/**
 * @brief Element-wise logical NOT: `c = !a` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = np.logical_not(a)`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a  Input vector.
 * @param c  Output vector (1/0 truth values).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_not(uint32_t N, T *a, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = !a[i]; }

/**
 * @brief Element-wise absolute value: `b = |a|` (cooperative-groups variant).
 *
 * NumPy equivalent: `b = np.abs(a)`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a  Input vector.
 * @param b  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_abs(uint32_t N, T *a, T *b,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) b[i] = abs(a[i]); }

/**
 * @brief Element-wise (Hadamard) product: `c = a * b` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_mult(uint32_t N, T *a, T *b, T *c,
                                  cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i]*b[i]; }

/**
 * @brief Element-wise subtraction: `c = a - b` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = a - b`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_sub(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] - b[i]; }

/**
 * @brief Element-wise addition: `c = a + b` (cooperative-groups variant).
 *
 * NumPy equivalent: `c = a + b`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a,b  Input vectors.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_add(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] + b[i]; }

/**
 * @brief Element-wise scalar multiply: `c = a * b` (scalar b, cooperative-groups variant).
 *
 * NumPy equivalent: `c = a * b`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a  Input vector.
 * @param b  Scalar multiplier.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_mult_scalar(uint32_t N, T *a, T b, T *c,
                                         cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i]*b; }

/**
 * @brief Element-wise max against a scalar: `c = max(a, b)` (scalar b, cooperative-groups variant).
 *
 * NumPy equivalent: `c = np.maximum(a, b)`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a  Input vector.
 * @param b  Scalar threshold.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_max_scalar(uint32_t N, T *a, T b, T *c,
                                        cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = max(a[i], b); }

/**
 * @brief Element-wise min against a scalar: `c = min(a, b)` (scalar b, cooperative-groups variant).
 *
 * NumPy equivalent: `c = np.minimum(a, b)`.
 *
 * @tparam T  Scalar type.
 * @param N  Vector length.
 * @param a  Input vector.
 * @param b  Scalar threshold.
 * @param c  Output vector.
 * @param g  Cooperative thread group (defaults to the whole block).
 */
template <typename T>
__device__ void elementwise_min_scalar(uint32_t N, T *a, T b, T *c,
                                        cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = min(a[i], b); }

/**
 * @brief Euclidean (L2) norm in place: `x[0] = ||x||_2` (cooperative-groups variant).
 *
 * Destructive: squares each element in place, sum-reduces, then takes the square
 * root; the result lands in `x[0]`. NumPy equivalent: `x[0] = np.linalg.norm(x)`.
 *
 * @tparam T  Scalar type.
 * @param n  Vector length.
 * @param x  In/out vector (overwritten; on return `x[0]` holds the L2 norm).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
// l2norm: squares each element in-place, reduces, takes sqrt. Result in x[0].
template <typename T>
__device__ void l2norm(uint32_t n, T *x,
                       cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) x[i] *= x[i];
    g.sync();
    reduce<T>(n, x, g);
    if (g.thread_rank() == 0) x[0] = sqrtf(x[0]);
}

/**
 * @brief Infinity norm in place: `x[0] = ||x||_inf` (cooperative-groups variant).
 *
 * Destructive halving reduction over absolute values; the max-abs result lands
 * in `x[0]`. NumPy equivalent: `x[0] = np.max(np.abs(x))`.
 *
 * @tparam T  Scalar type.
 * @param n  Vector length.
 * @param x  In/out vector (overwritten; on return `x[0]` holds the inf norm).
 * @param g  Cooperative thread group (defaults to the whole block).
 */
// infnorm: max of absolute values. Result in x[0].
template <typename T>
__device__ void infnorm(uint32_t n, T *x,
                         cgrps::thread_group g = cgrps::this_thread_block())
{
    uint32_t rank = g.thread_rank(), stride = g.size();
    uint32_t left = n;
    while (left > 3) {
        bool odd = left % 2;
        left = (left - odd) / 2;
        for (uint32_t i = rank; i < left; i += stride)
            x[i] = max(abs(x[i]), abs(x[i + left]));
        if (rank == 0 && odd) x[0] = max(abs(x[0]), abs(x[2*left]));
        g.sync();
    }
    if (rank == 0) { for (uint32_t i = 1; i < left; i++) x[0] = max(abs(x[0]), abs(x[i])); }
}

/**
 * @brief Sum of absolute values: `out[0] = sum(|x|)` (cooperative-groups variant).
 *
 * Writes `|x|` into the size-n scratch `out`, then sum-reduces it; the result
 * lands in `out[0]` and `x` is left unmodified. NumPy equivalent:
 * `out[0] = np.sum(np.abs(x))`.
 *
 * @tparam T  Scalar type.
 * @param n    Vector length.
 * @param x    Input vector.
 * @param out  Size-n scratch/output; on return `out[0]` holds the result.
 * @param g    Cooperative thread group (defaults to the whole block).
 */
// asum: sum of absolute values. out is size-n scratch+output; result in out[0].
template <typename T>
__device__ void asum(uint32_t n, T *x, T *out,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) out[i] = abs(x[i]);
    g.sync();
    reduce<T>(n, out, g);
}
