#pragma once
#include <cstdint>
#include <math.h>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// L1 cgrps variants — same semantics as glass:: but thread rank/size from the group

template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = alpha*x[i] + y[i];
}

template <typename T>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y, T *z,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) z[i] = alpha*x[i] + y[i];
}

template <typename T>
__device__ void axpby(uint32_t n, T alpha, T *x, T beta, T *y, T *z,
                      cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) z[i] = alpha*x[i] + beta*y[i];
}

template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) y[i] = alpha*x[i] + y[i];
}

template <typename T, uint32_t N>
__device__ void axpy(T alpha, T *x, T *y, T *z,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) z[i] = alpha*x[i] + y[i];
}

template <typename T>
__device__ void copy(uint32_t n, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = x[i];
}

template <typename T>
__device__ void copy(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = alpha*x[i];
}

template <typename T, uint32_t N>
__device__ void copy(T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) y[i] = x[i];
}

template <typename T>
__device__ void scal(uint32_t n, T alpha, T *x,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) x[i] = alpha*x[i];
}

template <typename T>
__device__ void scal(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] = alpha*x[i];
}

template <typename T, uint32_t N>
__device__ void scal(T alpha, T *x,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) x[i] = alpha*x[i];
}

template <typename T>
__device__ void swap(uint32_t n, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}

template <typename T, uint32_t N>
__device__ void swap(T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) {
        T tmp = x[i]; x[i] = y[i]; y[i] = tmp;
    }
}

template <typename T>
__device__ void clip(uint32_t n, T *x, T *l, T *u,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size())
        x[i] = max(l[i], min(x[i], u[i]));
}

template <typename T>
__device__ void set_const(uint32_t n, T alpha, T *x,
                           cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) x[i] = alpha;
}

template <typename T>
__device__ void loadIdentity(uint32_t n, T *A,
                              cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n*n; i += g.size()) {
        uint32_t r = i % n, c = i / n;
        A[i] = static_cast<T>(r == c);
    }
}

template <typename T>
__device__ void addI(uint32_t n, T *A, T alpha,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n*n; i += g.size())
        if (i % n == i / n) A[i] += alpha;
}

// reductions
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

template <typename T, uint32_t N>
__device__ void reduce(T *x, cgrps::thread_group g = cgrps::this_thread_block())
{
    reduce<T>(N, x, g);
}

template <typename T>
__device__ void dot(uint32_t n, T *x, T *y,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) y[i] *= x[i];
    g.sync();
    reduce<T>(n, y, g);
}

template <typename T, uint32_t N>
__device__ void dot(T *x, T *y,
                    cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < N; i += g.size()) y[i] *= x[i];
    g.sync();
    reduce<T, N>(y, g);
}

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

template <typename T>
__device__ void elementwise_max(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = max(a[i], b[i]); }

template <typename T>
__device__ void elementwise_min(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = min(a[i], b[i]); }

template <typename T>
__device__ void elementwise_less_than(uint32_t N, T *a, T *b, T *c,
                                       cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] < b[i]; }

template <typename T>
__device__ void elementwise_more_than(uint32_t N, T *a, T *b, T *c,
                                       cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] > b[i]; }

template <typename T>
__device__ void elementwise_less_than_or_eq(uint32_t N, T *a, T *b, T *c,
                                              cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] <= b[i]; }

template <typename T>
__device__ void elementwise_and(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] && b[i]; }

template <typename T>
__device__ void elementwise_not(uint32_t N, T *a, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = !a[i]; }

template <typename T>
__device__ void elementwise_abs(uint32_t N, T *a, T *b,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) b[i] = abs(a[i]); }

template <typename T>
__device__ void elementwise_mult(uint32_t N, T *a, T *b, T *c,
                                  cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i]*b[i]; }

template <typename T>
__device__ void elementwise_sub(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] - b[i]; }

template <typename T>
__device__ void elementwise_add(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i] + b[i]; }

template <typename T>
__device__ void elementwise_mult_scalar(uint32_t N, T *a, T b, T *c,
                                         cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = a[i]*b; }

template <typename T>
__device__ void elementwise_max_scalar(uint32_t N, T *a, T b, T *c,
                                        cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = max(a[i], b); }

template <typename T>
__device__ void elementwise_min_scalar(uint32_t N, T *a, T b, T *c,
                                        cgrps::thread_group g = cgrps::this_thread_block())
{ for (uint32_t i = g.thread_rank(); i < N; i += g.size()) c[i] = min(a[i], b); }

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

// asum: sum of absolute values. out is size-n scratch+output; result in out[0].
template <typename T>
__device__ void asum(uint32_t n, T *x, T *out,
                     cgrps::thread_group g = cgrps::this_thread_block())
{
    for (uint32_t i = g.thread_rank(); i < n; i += g.size()) out[i] = abs(x[i]);
    g.sync();
    reduce<T>(n, out, g);
}
