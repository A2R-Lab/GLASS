#pragma once
#include <cstdint>
#include <math.h>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// L1 cooperative-groups surface. Each op is a thin wrapper over the shared
// `glass::*_impl` body (src/base/L1/*.cuh), driven by a GroupBarrier instead of
// BlockBarrier — identical numerics, with rank/size/sync taken from the
// thread_group. Detailed semantics live on the matching `glass::` op; every op
// carries the uniform `bool TRAILING_SYNC=true` flag (pass false to elide the
// trailing barrier when the caller owns the next sync).

// Barrier policy for the cooperative-groups surface (found by enclosing-namespace
// lookup from glass::cgrps::; the shared glass::*_impl bodies live in glass::).
struct GroupBarrier {
    cgrps::thread_group g;
    __device__ __forceinline__ uint32_t rank() const { return g.thread_rank(); }
    __device__ __forceinline__ uint32_t size() const { return g.size(); }
    __device__ __forceinline__ void sync() const { cgrps::sync(g); }
};

// ── axpy / axpby ──────────────────────────────────────────────────────────────
/** @brief AXPY `y = alpha*x + y` (cgrps variant; see glass::axpy). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ axpy_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, alpha, x, y); }

/** @brief Out-of-place AXPY `z = alpha*x + y` (cgrps variant; see glass::axpy). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void axpy(uint32_t n, T alpha, T *x, T *y, T *z,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ axpy_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, alpha, x, y, z); }

/** @brief AXPBY `z = alpha*x + beta*y` (cgrps variant; see glass::axpby). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void axpby(uint32_t n, T alpha, T *x, T beta, T *y, T *z,
                      cgrps::thread_group g = cgrps::this_thread_block())
{ axpby_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, alpha, x, beta, y, z); }

/** @brief Compile-time-size AXPY `y = alpha*x + y` (cgrps variant). */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void axpy(T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ axpy_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, alpha, x, y); }

/** @brief Compile-time-size out-of-place AXPY `z = alpha*x + y` (cgrps variant). */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void axpy(T alpha, T *x, T *y, T *z,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ axpy_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, alpha, x, y, z); }

// ── copy ──────────────────────────────────────────────────────────────────────
/** @brief Vector copy `y = x` (cgrps variant; see glass::copy). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void copy(uint32_t n, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ copy_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x, y); }

/** @brief Scaled copy `y = alpha*x` (cgrps variant; see glass::copy). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void copy(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ copy_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, alpha, x, y); }

/** @brief Compile-time-size vector copy `y = x` (cgrps variant). */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void copy(T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ copy_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, x, y); }

// ── scal ──────────────────────────────────────────────────────────────────────
/** @brief In-place scale `x = alpha*x` (cgrps variant; see glass::scal). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void scal(uint32_t n, T alpha, T *x,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ scal_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, alpha, x); }

/** @brief Out-of-place scale `y = alpha*x` (cgrps variant; see glass::scal). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void scal(uint32_t n, T alpha, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ scal_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, alpha, x, y); }

/** @brief Compile-time-size in-place scale `x = alpha*x` (cgrps variant). */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void scal(T alpha, T *x,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ scal_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, alpha, x); }

// ── swap ──────────────────────────────────────────────────────────────────────
/** @brief Swap two vectors `x <-> y` (cgrps variant; see glass::swap). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void swap(uint32_t n, T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ swap_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x, y); }

/** @brief Compile-time-size swap `x <-> y` (cgrps variant). */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void swap(T *x, T *y,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ swap_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, x, y); }

// ── clip / set_const / identity ───────────────────────────────────────────────
/** @brief Element-wise clamp `x = clamp(x, l, u)` (cgrps variant; see glass::clip). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void clip(uint32_t n, T *x, T *l, T *u,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ clip_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x, l, u); }

/** @brief Fill with a constant `x = alpha` (cgrps variant; see glass::set_const). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void set_const(uint32_t n, T alpha, T *x,
                           cgrps::thread_group g = cgrps::this_thread_block())
{ set_const_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, alpha, x); }

/** @brief Load the identity `A = I_n` (cgrps variant; see glass::loadIdentity). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void loadIdentity(uint32_t n, T *A,
                              cgrps::thread_group g = cgrps::this_thread_block())
{ loadIdentity_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, A); }

/** @brief Add a scaled identity `A += alpha*I` (cgrps variant; see glass::addI). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void addI(uint32_t n, T *A, T alpha,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ addI_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, A, alpha); }

// ── reductions ────────────────────────────────────────────────────────────────
/** @brief Sum reduction `x[0] = sum(x)` (cgrps variant; see glass::reduce). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void reduce(uint32_t n, T *x,
                       cgrps::thread_group g = cgrps::this_thread_block())
{ reduce_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x); }

/** @brief Compile-time-size sum reduction `x[0] = sum(x)` (cgrps variant). */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void reduce(T *x, cgrps::thread_group g = cgrps::this_thread_block())
{ reduce_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, x); }

/** @brief Dot product `y[0] = dot(x, y)` (destructive; cgrps variant; see glass::dot). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void dot(uint32_t n, T *x, T *y,
                    cgrps::thread_group g = cgrps::this_thread_block())
{ dot_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x, y); }

/** @brief Compile-time-size dot product `y[0] = dot(x, y)` (cgrps variant). */
template <typename T, uint32_t N, bool TRAILING_SYNC = true>
__device__ void dot(T *x, T *y,
                    cgrps::thread_group g = cgrps::this_thread_block())
{ dot_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, x, y); }

// ── transpose ─────────────────────────────────────────────────────────────────
/** @brief Out-of-place transpose `b = a^T` (cgrps variant; see glass::transpose). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void transpose(uint32_t N, uint32_t M, T *a, T *b,
                           cgrps::thread_group g = cgrps::this_thread_block())
{ transpose_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, M, a, b); }

/** @brief In-place square transpose `a = a^T` (cgrps variant; see glass::transpose). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void transpose(uint32_t N, T *a,
                           cgrps::thread_group g = cgrps::this_thread_block())
{ transpose_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a); }

// ── elementwise logic ─────────────────────────────────────────────────────────
/** @brief Element-wise max `c = max(a, b)` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_max(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_max_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise min `c = min(a, b)` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_min(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_min_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise less-than `c = (a < b)` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_less_than(uint32_t N, T *a, T *b, T *c,
                                       cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_less_than_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise greater-than `c = (a > b)` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_more_than(uint32_t N, T *a, T *b, T *c,
                                       cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_more_than_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise less-than-or-equal `c = (a <= b)` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_less_than_or_eq(uint32_t N, T *a, T *b, T *c,
                                              cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_less_than_or_eq_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise logical AND `c = (a && b)` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_and(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_and_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise logical NOT `c = !a` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_not(uint32_t N, T *a, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_not_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, c); }

/** @brief Element-wise absolute value `b = |a|` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_abs(uint32_t N, T *a, T *b,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_abs_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b); }

/** @brief Element-wise (Hadamard) product `c = a * b` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_mult(uint32_t N, T *a, T *b, T *c,
                                  cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_mult_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise subtraction `c = a - b` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_sub(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_sub_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise addition `c = a + b` (cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_add(uint32_t N, T *a, T *b, T *c,
                                 cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_add_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise scalar multiply `c = a * b` (scalar b; cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_mult_scalar(uint32_t N, T *a, T b, T *c,
                                         cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_mult_scalar_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise max against a scalar `c = max(a, b)` (scalar b; cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_max_scalar(uint32_t N, T *a, T b, T *c,
                                        cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_max_scalar_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

/** @brief Element-wise min against a scalar `c = min(a, b)` (scalar b; cgrps variant). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void elementwise_min_scalar(uint32_t N, T *a, T b, T *c,
                                        cgrps::thread_group g = cgrps::this_thread_block())
{ elementwise_min_scalar_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, N, a, b, c); }

// ── norms ─────────────────────────────────────────────────────────────────────
/** @brief Euclidean (L2) norm `x[0] = ||x||_2` (destructive; cgrps variant; see glass::nrm2_impl). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void nrm2(uint32_t n, T *x,
                       cgrps::thread_group g = cgrps::this_thread_block())
{ nrm2_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x); }

/** @brief Infinity norm `x[0] = ||x||_inf` (destructive; cgrps variant; see glass::infnorm). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void infnorm(uint32_t n, T *x,
                         cgrps::thread_group g = cgrps::this_thread_block())
{ infnorm_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x); }

/** @brief Sum of absolute values `out[0] = sum(|x|)` (cgrps variant; see glass::asum_impl). */
template <typename T, bool TRAILING_SYNC = true>
__device__ void asum(uint32_t n, T *x, T *out,
                     cgrps::thread_group g = cgrps::this_thread_block())
{ asum_impl<GroupBarrier, T, TRAILING_SYNC>(GroupBarrier{g}, n, x, out); }
