#pragma once
/**
 * @file lapack_thread.cuh
 * @brief cuSOLVERDx thread-execution LAPACK wrappers for
 *        `glass::nvidia::thread::`.
 *
 * Each calling CUDA thread owns one complete, compile-time-size problem. The
 * operands may live in global, shared, or local memory; the wrappers allocate
 * no dynamic shared memory and perform no block-wide synchronization. This
 * makes the surface suitable for batches of small independent systems.
 *
 * The API mirrors the operations exposed by `glass::nvidia::block`, but omits
 * block-only scratch, BlockDim, and trailing-sync parameters. All matrices are
 * packed column-major. Requires cuSOLVERDx 0.4 or newer and the same MathDx
 * device-link flags as the block backend.
 */
#include <cstdint>

#ifndef SMS
#define SMS 860
#endif

namespace thread {

/** Factor an N-by-N SPD matrix in place as A = L*L^T. */
template <typename T, uint32_t N, uint32_t SM_VAL = SMS>
__device__ inline void potrf(T* A)
{
    using solver = decltype(
        cusolverdx::Size<N, N>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::potrf>()
        + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()
        + cusolverdx::Arrangement<cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    int info = 0;
    solver().execute(A, &info);
}

/** Solve L*X = alpha*B in place for a lower-triangular M-by-M L. */
template <typename T, uint32_t M, uint32_t N, uint32_t SM_VAL = SMS>
__device__ inline void trsm(T alpha, const T* L, T* B)
{
    for (uint32_t i = 0; i < M * N; ++i) {
        B[i] *= alpha;
    }
    using solver = decltype(
        cusolverdx::Size<M, N>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::trsm>()
        + cusolverdx::Side<cusolverdx::side::left>()
        + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()
        + cusolverdx::TransposeMode<cusolverdx::non_trans>()
        + cusolverdx::Diag<cusolverdx::diag::non_unit>()
        + cusolverdx::Arrangement<cusolverdx::col_major,
                                  cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    solver().execute(L, B);
}

/** Factor A and solve A*X = B in place for NRHS right-hand sides. */
template <typename T, uint32_t N, uint32_t NRHS, uint32_t SM_VAL = SMS>
__device__ inline void posv(T* A, T* B)
{
    using solver = decltype(
        cusolverdx::Size<N, N, NRHS>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::posv>()
        + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()
        + cusolverdx::Arrangement<cusolverdx::col_major,
                                  cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    int info = 0;
    solver().execute(A, B, &info);
}

/** Solve L*L^T*X = B from a precomputed lower Cholesky factor. */
template <typename T, uint32_t N, uint32_t NRHS, uint32_t SM_VAL = SMS>
__device__ inline void potrs(const T* L, T* B)
{
    using solver = decltype(
        cusolverdx::Size<N, N, NRHS>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::potrs>()
        + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()
        + cusolverdx::Arrangement<cusolverdx::col_major,
                                  cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    solver().execute(L, B);
}

/** Compute an unpivoted LU factorization in place. */
template <typename T, uint32_t N, uint32_t SM_VAL = SMS>
__device__ inline void getrf_no_pivot(T* A)
{
    using solver = decltype(
        cusolverdx::Size<N, N>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::getrf_no_pivot>()
        + cusolverdx::Arrangement<cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    int info = 0;
    solver().execute(A, &info);
}

/** Solve A*X = B from a precomputed unpivoted LU factorization. */
template <typename T, uint32_t N, uint32_t NRHS, uint32_t SM_VAL = SMS>
__device__ inline void getrs_no_pivot(const T* LU, T* B)
{
    using solver = decltype(
        cusolverdx::Size<N, N, NRHS>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::getrs_no_pivot>()
        + cusolverdx::Arrangement<cusolverdx::col_major,
                                  cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    solver().execute(LU, B);
}

/** Factor A without pivoting and solve A*X = B in place. */
template <typename T, uint32_t N, uint32_t NRHS, uint32_t SM_VAL = SMS>
__device__ inline void gesv_no_pivot(T* A, T* B)
{
    using solver = decltype(
        cusolverdx::Size<N, N, NRHS>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::gesv_no_pivot>()
        + cusolverdx::Arrangement<cusolverdx::col_major,
                                  cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    int info = 0;
    solver().execute(A, B, &info);
}

/** Compute a packed column-major M-by-N QR factorization in place. */
template <typename T, uint32_t M, uint32_t N, uint32_t SM_VAL = SMS>
__device__ inline void geqrf(T* A, T* tau)
{
    using solver = decltype(
        cusolverdx::Size<M, N>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::geqrf>()
        + cusolverdx::Arrangement<cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    solver().execute(A, tau);
}

/** Solve an M-by-N least-squares problem with NRHS right-hand sides. */
template <typename T, uint32_t M, uint32_t N, uint32_t NRHS,
          uint32_t SM_VAL = SMS>
__device__ inline void gels(T* A, T* tau, T* B)
{
    using solver = decltype(
        cusolverdx::Size<M, N, NRHS>()
        + cusolverdx::Precision<T>()
        + cusolverdx::Type<cusolverdx::type::real>()
        + cusolverdx::Function<cusolverdx::function::gels>()
        + cusolverdx::Arrangement<cusolverdx::col_major,
                                  cusolverdx::col_major>()
        + cusolverdx::Thread()
        + cusolverdx::SM<SM_VAL>());
    solver().execute(A, tau, B);
}

}  // namespace thread
