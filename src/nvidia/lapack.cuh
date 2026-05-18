#pragma once
#include <cstdint>
// cusolverdx.hpp and cusolverdx_io.hpp must be included at GLOBAL scope
// (glass-nvidia.cuh handles this) — re-including them inside `namespace
// glass::nvidia` would nest the cusolverdx symbols at the wrong scope.
#include "./types.cuh"

// glass::nvidia LAPACK — cuSOLVERDx-backed Cholesky and triangular solve.
//
// Currently supported (matching GRiD's chol_InPlace + trsm contract):
//   chol_inplace<T, N>(A, smem)
//       In-place Cholesky factorization of an N×N column-major SPD matrix:
//       A := L  s.t.  A = L * L^T  (lower-triangular fill mode).
//
//   trsm<T, M, N>(alpha, L, B, smem)
//       Triangular solve: solve L * X = alpha * B in place into B.
//       Side=left, FillMode=lower, TransposeMode=non_trans, Diag=non_unit,
//       both A and B column-major.
//
// As with gemm/gemv, all sizes are compile-time. Call DEFINE_NVIDIA_CHOL(N) /
// DEFINE_NVIDIA_TRSM(M, N) per shape, plus *_BLOCKDIM / *_SM variants when you
// need a pinned launch thread count or non-default SM.
//
// Example:
//   DEFINE_NVIDIA_CHOL_BLOCKDIM(7, 352)
//   constexpr auto smem    = glass::nvidia::chol_inplace_smem_size<float, 7, 352>();
//   constexpr auto threads = glass::nvidia::chol_inplace_threads<float, 7, 352>();
//   kernel<<<1, threads, smem>>>(d_A);
//   glass::nvidia::chol_inplace<float, 7, 352>(A, smem_ptr);

#ifndef SMS
#define SMS 860
#endif

// ---------------------------------------------------------------------------
// Primary templates
// ---------------------------------------------------------------------------

template <typename T, uint32_t N, uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS,
          bool TRAILING_SYNC = true>
__device__ void chol_inplace(T* A, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::chol_inplace<T,N,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_CHOL* in your .cu file.");
}

template <typename T, uint32_t N, uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t chol_inplace_smem_size() { return 0; }

template <typename T, uint32_t N, uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t chol_inplace_threads() { return 256; }

template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS,
          bool TRAILING_SYNC = true>
__device__ void trsm(T alpha, T* L, T* B, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::trsm<T,M,N,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_TRSM* in your .cu file.");
}

template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t trsm_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t trsm_threads() { return 256; }

// ---------------------------------------------------------------------------
// Private core macros — Cholesky
// ---------------------------------------------------------------------------

// ARCH (not SM) avoids token collision with cusolverdx::SM<>.
#define _GLASS_CHOL_NO_BD(N, ARCH)                                                            \
    namespace _nvidia_chol_impl_##N##_bd0_sm##ARCH {                                          \
        using SOLVER = decltype(                                                              \
            cusolverdx::Size<N, N>()                                                          \
            + cusolverdx::Precision<float>()                                                  \
            + cusolverdx::Type<cusolverdx::type::real>()                                      \
            + cusolverdx::Function<cusolverdx::function::potrf>()                             \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                            \
            + cusolverdx::Arrangement<cusolverdx::col_major>()                                \
            + cusolverdx::SM<ARCH>()                                                          \
            + cusolverdx::Block());                                                           \
        static constexpr uint32_t block_threads =                                             \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                       \
                                  SOLVER::block_dim.y *                                       \
                                  SOLVER::block_dim.z);                                       \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                 \
        template <bool TRAILING_SYNC>                                                         \
        __device__ inline void run(float* A, char* smem) {                                    \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                \
            float* As = reinterpret_cast<float*>(smem);                                       \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, SOLVER::lda); \
            __syncthreads();                                                                  \
            int info = 0;                                                                     \
            SOLVER().execute(As, &info);                                                      \
            __syncthreads();                                                                  \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, SOLVER::lda, A, N); \
            if constexpr (TRAILING_SYNC) {                                                    \
                __syncthreads();                                                              \
            }                                                                                 \
        }                                                                                     \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void chol_inplace<float, N, 0, ARCH, true>(float* A, char* smem)        \
    {                                                                                         \
        _nvidia_chol_impl_##N##_bd0_sm##ARCH::template run<true>(A, smem);                    \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void chol_inplace<float, N, 0, ARCH, false>(float* A, char* smem)       \
    {                                                                                         \
        _nvidia_chol_impl_##N##_bd0_sm##ARCH::template run<false>(A, smem);                   \
    }                                                                                         \
    template <>                                                                               \
    constexpr std::size_t chol_inplace_smem_size<float, N, 0, ARCH>()                         \
    {                                                                                         \
        return _nvidia_chol_impl_##N##_bd0_sm##ARCH::smem_bytes;                              \
    }                                                                                         \
    template <>                                                                               \
    constexpr uint32_t chol_inplace_threads<float, N, 0, ARCH>()                              \
    {                                                                                         \
        return _nvidia_chol_impl_##N##_bd0_sm##ARCH::block_threads;                           \
    }

#define _GLASS_CHOL_BD(N, TC, ARCH)                                                           \
    namespace _nvidia_chol_impl_##N##_bd##TC##_sm##ARCH {                                     \
        using SOLVER = decltype(                                                              \
            cusolverdx::Size<N, N>()                                                          \
            + cusolverdx::Precision<float>()                                                  \
            + cusolverdx::Type<cusolverdx::type::real>()                                      \
            + cusolverdx::Function<cusolverdx::function::potrf>()                             \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                            \
            + cusolverdx::Arrangement<cusolverdx::col_major>()                                \
            + cusolverdx::SM<ARCH>()                                                          \
            + cusolverdx::Block()                                                             \
            + cusolverdx::BlockDim<TC>());                                                    \
        static constexpr uint32_t block_threads =                                             \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                       \
                                  SOLVER::block_dim.y *                                       \
                                  SOLVER::block_dim.z);                                       \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                 \
        template <bool TRAILING_SYNC>                                                         \
        __device__ inline void run(float* A, char* smem) {                                    \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                \
            float* As = reinterpret_cast<float*>(smem);                                       \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, SOLVER::lda); \
            __syncthreads();                                                                  \
            int info = 0;                                                                     \
            SOLVER().execute(As, &info);                                                      \
            __syncthreads();                                                                  \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, SOLVER::lda, A, N); \
            if constexpr (TRAILING_SYNC) {                                                    \
                __syncthreads();                                                              \
            }                                                                                 \
        }                                                                                     \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void chol_inplace<float, N, TC, ARCH, true>(float* A, char* smem)       \
    {                                                                                         \
        _nvidia_chol_impl_##N##_bd##TC##_sm##ARCH::template run<true>(A, smem);               \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void chol_inplace<float, N, TC, ARCH, false>(float* A, char* smem)      \
    {                                                                                         \
        _nvidia_chol_impl_##N##_bd##TC##_sm##ARCH::template run<false>(A, smem);              \
    }                                                                                         \
    template <>                                                                               \
    constexpr std::size_t chol_inplace_smem_size<float, N, TC, ARCH>()                        \
    {                                                                                         \
        return _nvidia_chol_impl_##N##_bd##TC##_sm##ARCH::smem_bytes;                         \
    }                                                                                         \
    template <>                                                                               \
    constexpr uint32_t chol_inplace_threads<float, N, TC, ARCH>()                             \
    {                                                                                         \
        return _nvidia_chol_impl_##N##_bd##TC##_sm##ARCH::block_threads;                      \
    }

#define _GLASS_CHOL_NO_BD_E(N, ARCH)        _GLASS_CHOL_NO_BD(N, ARCH)
#define _GLASS_CHOL_BD_E(N, TC, ARCH)       _GLASS_CHOL_BD(N, TC, ARCH)

// ---------------------------------------------------------------------------
// Private core macros — TRSM (left, lower, non_trans, non_unit, col_major)
// ---------------------------------------------------------------------------

#define _GLASS_TRSM_NO_BD(M, N, ARCH)                                                         \
    namespace _nvidia_trsm_impl_##M##x##N##_bd0_sm##ARCH {                                    \
        using SOLVER = decltype(                                                              \
            cusolverdx::Size<M, N>()                                                          \
            + cusolverdx::Precision<float>()                                                  \
            + cusolverdx::Type<cusolverdx::type::real>()                                      \
            + cusolverdx::Function<cusolverdx::function::trsm>()                              \
            + cusolverdx::Side<cusolverdx::side::left>()                                      \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                            \
            + cusolverdx::TransposeMode<cusolverdx::non_trans>()                              \
            + cusolverdx::Diag<cusolverdx::diag::non_unit>()                                  \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()         \
            + cusolverdx::SM<ARCH>()                                                          \
            + cusolverdx::Block());                                                           \
        static constexpr uint32_t block_threads =                                             \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                       \
                                  SOLVER::block_dim.y *                                       \
                                  SOLVER::block_dim.z);                                       \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                 \
        template <bool TRAILING_SYNC>                                                         \
        __device__ inline void run(float alpha, float* L, float* B, char* smem) {             \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                \
            float* Ls = reinterpret_cast<float*>(smem);                                       \
            float* Bs = Ls + (M * M);                                                         \
            cusolverdx::copy_2d<SOLVER, M, M, cusolverdx::col_major>(L, M, Ls, M);           \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(B, M, Bs, M);           \
            __syncthreads();                                                                  \
            /* cuSOLVERDx trsm has no alpha — pre-multiply B by alpha. */                    \
            const uint32_t _gn_rank = threadIdx.x + threadIdx.y * blockDim.x +                \
                                      threadIdx.z * blockDim.x * blockDim.y;                  \
            const uint32_t _gn_size = blockDim.x * blockDim.y * blockDim.z;                   \
            if (alpha != 1.0f) {                                                              \
                for (uint32_t _gn_i = _gn_rank; _gn_i < (M) * (N); _gn_i += _gn_size)         \
                    Bs[_gn_i] *= alpha;                                                       \
                __syncthreads();                                                              \
            }                                                                                 \
            SOLVER().execute(Ls, M, Bs, M);                                                   \
            __syncthreads();                                                                  \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(Bs, M, B, M);           \
            if constexpr (TRAILING_SYNC) {                                                    \
                __syncthreads();                                                              \
            }                                                                                 \
        }                                                                                     \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void trsm<float, M, N, 0, ARCH, true>                                   \
        (float alpha, float* L, float* B, char* smem)                                         \
    {                                                                                         \
        _nvidia_trsm_impl_##M##x##N##_bd0_sm##ARCH::template run<true>(alpha, L, B, smem);    \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void trsm<float, M, N, 0, ARCH, false>                                  \
        (float alpha, float* L, float* B, char* smem)                                         \
    {                                                                                         \
        _nvidia_trsm_impl_##M##x##N##_bd0_sm##ARCH::template run<false>(alpha, L, B, smem);   \
    }                                                                                         \
    template <>                                                                               \
    constexpr std::size_t trsm_smem_size<float, M, N, 0, ARCH>()                              \
    {                                                                                         \
        return _nvidia_trsm_impl_##M##x##N##_bd0_sm##ARCH::smem_bytes;                        \
    }                                                                                         \
    template <>                                                                               \
    constexpr uint32_t trsm_threads<float, M, N, 0, ARCH>()                                   \
    {                                                                                         \
        return _nvidia_trsm_impl_##M##x##N##_bd0_sm##ARCH::block_threads;                     \
    }

#define _GLASS_TRSM_BD(M, N, TC, ARCH)                                                        \
    namespace _nvidia_trsm_impl_##M##x##N##_bd##TC##_sm##ARCH {                               \
        using SOLVER = decltype(                                                              \
            cusolverdx::Size<M, N>()                                                          \
            + cusolverdx::Precision<float>()                                                  \
            + cusolverdx::Type<cusolverdx::type::real>()                                      \
            + cusolverdx::Function<cusolverdx::function::trsm>()                              \
            + cusolverdx::Side<cusolverdx::side::left>()                                      \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                            \
            + cusolverdx::TransposeMode<cusolverdx::non_trans>()                              \
            + cusolverdx::Diag<cusolverdx::diag::non_unit>()                                  \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()         \
            + cusolverdx::SM<ARCH>()                                                          \
            + cusolverdx::Block()                                                             \
            + cusolverdx::BlockDim<TC>());                                                    \
        static constexpr uint32_t block_threads =                                             \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                       \
                                  SOLVER::block_dim.y *                                       \
                                  SOLVER::block_dim.z);                                       \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                 \
        template <bool TRAILING_SYNC>                                                         \
        __device__ inline void run(float alpha, float* L, float* B, char* smem) {             \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                \
            float* Ls = reinterpret_cast<float*>(smem);                                       \
            float* Bs = Ls + (M * M);                                                         \
            cusolverdx::copy_2d<SOLVER, M, M, cusolverdx::col_major>(L, M, Ls, M);           \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(B, M, Bs, M);           \
            __syncthreads();                                                                  \
            /* cuSOLVERDx trsm has no alpha — pre-multiply B by alpha. */                    \
            const uint32_t _gn_rank = threadIdx.x + threadIdx.y * blockDim.x +                \
                                      threadIdx.z * blockDim.x * blockDim.y;                  \
            const uint32_t _gn_size = blockDim.x * blockDim.y * blockDim.z;                   \
            if (alpha != 1.0f) {                                                              \
                for (uint32_t _gn_i = _gn_rank; _gn_i < (M) * (N); _gn_i += _gn_size)         \
                    Bs[_gn_i] *= alpha;                                                       \
                __syncthreads();                                                              \
            }                                                                                 \
            SOLVER().execute(Ls, M, Bs, M);                                                   \
            __syncthreads();                                                                  \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(Bs, M, B, M);           \
            if constexpr (TRAILING_SYNC) {                                                    \
                __syncthreads();                                                              \
            }                                                                                 \
        }                                                                                     \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void trsm<float, M, N, TC, ARCH, true>                                  \
        (float alpha, float* L, float* B, char* smem)                                         \
    {                                                                                         \
        _nvidia_trsm_impl_##M##x##N##_bd##TC##_sm##ARCH::template run<true>(alpha, L, B, smem);\
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void trsm<float, M, N, TC, ARCH, false>                                 \
        (float alpha, float* L, float* B, char* smem)                                         \
    {                                                                                         \
        _nvidia_trsm_impl_##M##x##N##_bd##TC##_sm##ARCH::template run<false>(alpha, L, B, smem);\
    }                                                                                         \
    template <>                                                                               \
    constexpr std::size_t trsm_smem_size<float, M, N, TC, ARCH>()                             \
    {                                                                                         \
        return _nvidia_trsm_impl_##M##x##N##_bd##TC##_sm##ARCH::smem_bytes;                   \
    }                                                                                         \
    template <>                                                                               \
    constexpr uint32_t trsm_threads<float, M, N, TC, ARCH>()                                  \
    {                                                                                         \
        return _nvidia_trsm_impl_##M##x##N##_bd##TC##_sm##ARCH::block_threads;                \
    }

// _E indirection wrappers force ARCH (often `SMS` macro) to be expanded
// before token pasting freezes it as the literal text "SMS".
#define _GLASS_TRSM_NO_BD_E(M, N, ARCH)         _GLASS_TRSM_NO_BD(M, N, ARCH)
#define _GLASS_TRSM_BD_E(M, N, TC, ARCH)        _GLASS_TRSM_BD(M, N, TC, ARCH)

// ---------------------------------------------------------------------------
// Public DEFINE_NVIDIA_CHOL* / DEFINE_NVIDIA_TRSM* convenience macros
// (route through _E indirection for SMS expansion)
// ---------------------------------------------------------------------------

#define DEFINE_NVIDIA_CHOL(N)                       _GLASS_CHOL_NO_BD_E(N, SMS)
#define DEFINE_NVIDIA_CHOL_BLOCKDIM(N, TC)          _GLASS_CHOL_BD_E(N, TC, SMS)
#define DEFINE_NVIDIA_CHOL_SM(N, SM)                _GLASS_CHOL_NO_BD_E(N, SM)
#define DEFINE_NVIDIA_CHOL_BLOCKDIM_SM(N, TC, SM)   _GLASS_CHOL_BD_E(N, TC, SM)

#define DEFINE_NVIDIA_TRSM(M, N)                       _GLASS_TRSM_NO_BD_E(M, N, SMS)
#define DEFINE_NVIDIA_TRSM_BLOCKDIM(M, N, TC)          _GLASS_TRSM_BD_E(M, N, TC, SMS)
#define DEFINE_NVIDIA_TRSM_SM(M, N, SM)                _GLASS_TRSM_NO_BD_E(M, N, SM)
#define DEFINE_NVIDIA_TRSM_BLOCKDIM_SM(M, N, TC, SM)   _GLASS_TRSM_BD_E(M, N, TC, SM)

// =============================================================================
// Part 2 — Expanded cuSOLVERDx solver suite
//   posv          — SPD factor + solve in one call
//   potrs         — SPD solve given the L factor from chol_inplace
//   getrf_no_pivot — LU (no pivoting) factor in place
//   getrs_no_pivot — LU solve given the LU factor
//   gesv_no_pivot  — LU + solve in one call
//   geqrf          — QR factorization
//   gels           — least-squares solve (over- or under-determined)
//
// All seven follow the same primary-template + private-core-macro + _E
// indirection + public DEFINE convenience-macro pattern as chol_inplace/trsm.
// Each provides _smem_size and _threads constexpr queries.
// =============================================================================

// --- posv (SPD factor + solve) ----------------------------------------------
// Solves A·X = B for SPD A in one call.  A is overwritten with L (lower);
// B is overwritten with X.  Smem layout: [As: N*N] [Bs: N*NRHS] [solver_smem].

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS,
          bool TRAILING_SYNC = true>
__device__ void posv(T* A, T* B, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::posv<T,N,NRHS,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_POSV* in your .cu file.");
}

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t posv_smem_size() { return 0; }

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t posv_threads() { return 256; }

#define _GLASS_POSV_NO_BD(N, NRHS, ARCH)                                                         \
    namespace _nvidia_posv_impl_##N##x##NRHS##_bd0_sm##ARCH {                                    \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::posv>()                                 \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                               \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block());                                                              \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        template <bool TRAILING_SYNC>                                                            \
        __device__ inline void run(float* A, float* B, char* smem) {                             \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            float* Bs = As + (N * N);                                                            \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            int info = 0;                                                                        \
            SOLVER().execute(As, N, Bs, N, &info);                                               \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, N, A, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            if constexpr (TRAILING_SYNC) {                                                       \
                __syncthreads();                                                                 \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void posv<float, N, NRHS, 0, ARCH, true>(float* A, float* B, char* smem)  \
    { _nvidia_posv_impl_##N##x##NRHS##_bd0_sm##ARCH::template run<true>(A, B, smem); }           \
    template <>                                                                                  \
    __device__ inline void posv<float, N, NRHS, 0, ARCH, false>(float* A, float* B, char* smem) \
    { _nvidia_posv_impl_##N##x##NRHS##_bd0_sm##ARCH::template run<false>(A, B, smem); }          \
    template <>                                                                                  \
    constexpr std::size_t posv_smem_size<float, N, NRHS, 0, ARCH>()                             \
    { return _nvidia_posv_impl_##N##x##NRHS##_bd0_sm##ARCH::smem_bytes; }                        \
    template <>                                                                                  \
    constexpr uint32_t posv_threads<float, N, NRHS, 0, ARCH>()                                  \
    { return _nvidia_posv_impl_##N##x##NRHS##_bd0_sm##ARCH::block_threads; }

#define _GLASS_POSV_BD(N, NRHS, TC, ARCH)                                                        \
    namespace _nvidia_posv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH {                               \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::posv>()                                 \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                               \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block()                                                                \
            + cusolverdx::BlockDim<TC>());                                                       \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        template <bool TRAILING_SYNC>                                                            \
        __device__ inline void run(float* A, float* B, char* smem) {                             \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            float* Bs = As + (N * N);                                                            \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            int info = 0;                                                                        \
            SOLVER().execute(As, N, Bs, N, &info);                                               \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, N, A, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            if constexpr (TRAILING_SYNC) {                                                       \
                __syncthreads();                                                                 \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void posv<float, N, NRHS, TC, ARCH, true>(float* A, float* B, char* smem) \
    { _nvidia_posv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::template run<true>(A, B, smem); }      \
    template <>                                                                                  \
    __device__ inline void posv<float, N, NRHS, TC, ARCH, false>(float* A, float* B, char* smem)\
    { _nvidia_posv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::template run<false>(A, B, smem); }     \
    template <>                                                                                  \
    constexpr std::size_t posv_smem_size<float, N, NRHS, TC, ARCH>()                            \
    { return _nvidia_posv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::smem_bytes; }                   \
    template <>                                                                                  \
    constexpr uint32_t posv_threads<float, N, NRHS, TC, ARCH>()                                 \
    { return _nvidia_posv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::block_threads; }

#define _GLASS_POSV_NO_BD_E(N, NRHS, ARCH)        _GLASS_POSV_NO_BD(N, NRHS, ARCH)
#define _GLASS_POSV_BD_E(N, NRHS, TC, ARCH)       _GLASS_POSV_BD(N, NRHS, TC, ARCH)

#define DEFINE_NVIDIA_POSV(N, NRHS)                          _GLASS_POSV_NO_BD_E(N, NRHS, SMS)
#define DEFINE_NVIDIA_POSV_BLOCKDIM(N, NRHS, TC)             _GLASS_POSV_BD_E(N, NRHS, TC, SMS)
#define DEFINE_NVIDIA_POSV_SM(N, NRHS, SM)                   _GLASS_POSV_NO_BD_E(N, NRHS, SM)
#define DEFINE_NVIDIA_POSV_BLOCKDIM_SM(N, NRHS, TC, SM)      _GLASS_POSV_BD_E(N, NRHS, TC, SM)

// --- potrs (SPD solve given L) ----------------------------------------------
// Solves L·L^T·X = B given the L factor from chol_inplace.  B is overwritten
// with X.  Smem layout: [Ls: N*N] [Bs: N*NRHS] [solver_smem].

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
__device__ void potrs(const T* L, T* B, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::potrs<T,N,NRHS,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_POTRS* in your .cu file.");
}

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t potrs_smem_size() { return 0; }

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t potrs_threads() { return 256; }

#define _GLASS_POTRS_NO_BD(N, NRHS, ARCH)                                                        \
    namespace _nvidia_potrs_impl_##N##x##NRHS##_bd0_sm##ARCH {                                   \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::potrs>()                                \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                               \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block());                                                              \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(const float* L, float* B, char* smem) {                       \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* Ls = reinterpret_cast<float*>(smem);                                          \
            float* Bs = Ls + (N * N);                                                            \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(L, N, Ls, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            SOLVER().execute(Ls, N, Bs, N);                                                      \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void potrs<float, N, NRHS, 0, ARCH>(const float* L, float* B, char* smem) \
    { _nvidia_potrs_impl_##N##x##NRHS##_bd0_sm##ARCH::run(L, B, smem); }                         \
    template <>                                                                                  \
    constexpr std::size_t potrs_smem_size<float, N, NRHS, 0, ARCH>()                            \
    { return _nvidia_potrs_impl_##N##x##NRHS##_bd0_sm##ARCH::smem_bytes; }                       \
    template <>                                                                                  \
    constexpr uint32_t potrs_threads<float, N, NRHS, 0, ARCH>()                                 \
    { return _nvidia_potrs_impl_##N##x##NRHS##_bd0_sm##ARCH::block_threads; }

#define _GLASS_POTRS_BD(N, NRHS, TC, ARCH)                                                       \
    namespace _nvidia_potrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH {                              \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::potrs>()                                \
            + cusolverdx::FillMode<cusolverdx::fill_mode::lower>()                               \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block()                                                                \
            + cusolverdx::BlockDim<TC>());                                                       \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(const float* L, float* B, char* smem) {                       \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* Ls = reinterpret_cast<float*>(smem);                                          \
            float* Bs = Ls + (N * N);                                                            \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(L, N, Ls, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            SOLVER().execute(Ls, N, Bs, N);                                                      \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void potrs<float, N, NRHS, TC, ARCH>(const float* L, float* B, char* smem)\
    { _nvidia_potrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::run(L, B, smem); }                    \
    template <>                                                                                  \
    constexpr std::size_t potrs_smem_size<float, N, NRHS, TC, ARCH>()                           \
    { return _nvidia_potrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::smem_bytes; }                  \
    template <>                                                                                  \
    constexpr uint32_t potrs_threads<float, N, NRHS, TC, ARCH>()                                \
    { return _nvidia_potrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::block_threads; }

#define _GLASS_POTRS_NO_BD_E(N, NRHS, ARCH)        _GLASS_POTRS_NO_BD(N, NRHS, ARCH)
#define _GLASS_POTRS_BD_E(N, NRHS, TC, ARCH)       _GLASS_POTRS_BD(N, NRHS, TC, ARCH)

#define DEFINE_NVIDIA_POTRS(N, NRHS)                         _GLASS_POTRS_NO_BD_E(N, NRHS, SMS)
#define DEFINE_NVIDIA_POTRS_BLOCKDIM(N, NRHS, TC)            _GLASS_POTRS_BD_E(N, NRHS, TC, SMS)
#define DEFINE_NVIDIA_POTRS_SM(N, NRHS, SM)                  _GLASS_POTRS_NO_BD_E(N, NRHS, SM)
#define DEFINE_NVIDIA_POTRS_BLOCKDIM_SM(N, NRHS, TC, SM)     _GLASS_POTRS_BD_E(N, NRHS, TC, SM)

// --- getrf_no_pivot (LU factor, no pivoting) --------------------------------
// In-place LU factorization without pivoting.  A := L (unit lower) + U (upper).
// Smem layout: [As: N*N] [solver_smem].

template <typename T, uint32_t N,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
__device__ void getrf_no_pivot(T* A, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::getrf_no_pivot<T,N,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_GETRF* in your .cu file.");
}

template <typename T, uint32_t N, uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t getrf_no_pivot_smem_size() { return 0; }

template <typename T, uint32_t N, uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t getrf_no_pivot_threads() { return 256; }

#define _GLASS_GETRF_NO_BD(N, ARCH)                                                              \
    namespace _nvidia_getrf_impl_##N##_bd0_sm##ARCH {                                            \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N>()                                                             \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::getrf_no_pivot>()                       \
            + cusolverdx::Arrangement<cusolverdx::col_major>()                                   \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block());                                                              \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(float* A, char* smem) {                                       \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, N);              \
            __syncthreads();                                                                     \
            int info = 0;                                                                        \
            SOLVER().execute(As, N, &info);                                                      \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, N, A, N);              \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void getrf_no_pivot<float, N, 0, ARCH>(float* A, char* smem)              \
    { _nvidia_getrf_impl_##N##_bd0_sm##ARCH::run(A, smem); }                                     \
    template <>                                                                                  \
    constexpr std::size_t getrf_no_pivot_smem_size<float, N, 0, ARCH>()                         \
    { return _nvidia_getrf_impl_##N##_bd0_sm##ARCH::smem_bytes; }                                \
    template <>                                                                                  \
    constexpr uint32_t getrf_no_pivot_threads<float, N, 0, ARCH>()                              \
    { return _nvidia_getrf_impl_##N##_bd0_sm##ARCH::block_threads; }

#define _GLASS_GETRF_BD(N, TC, ARCH)                                                             \
    namespace _nvidia_getrf_impl_##N##_bd##TC##_sm##ARCH {                                       \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N>()                                                             \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::getrf_no_pivot>()                       \
            + cusolverdx::Arrangement<cusolverdx::col_major>()                                   \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block()                                                                \
            + cusolverdx::BlockDim<TC>());                                                       \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(float* A, char* smem) {                                       \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, N);              \
            __syncthreads();                                                                     \
            int info = 0;                                                                        \
            SOLVER().execute(As, N, &info);                                                      \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, N, A, N);              \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void getrf_no_pivot<float, N, TC, ARCH>(float* A, char* smem)             \
    { _nvidia_getrf_impl_##N##_bd##TC##_sm##ARCH::run(A, smem); }                                \
    template <>                                                                                  \
    constexpr std::size_t getrf_no_pivot_smem_size<float, N, TC, ARCH>()                        \
    { return _nvidia_getrf_impl_##N##_bd##TC##_sm##ARCH::smem_bytes; }                           \
    template <>                                                                                  \
    constexpr uint32_t getrf_no_pivot_threads<float, N, TC, ARCH>()                             \
    { return _nvidia_getrf_impl_##N##_bd##TC##_sm##ARCH::block_threads; }

#define _GLASS_GETRF_NO_BD_E(N, ARCH)        _GLASS_GETRF_NO_BD(N, ARCH)
#define _GLASS_GETRF_BD_E(N, TC, ARCH)       _GLASS_GETRF_BD(N, TC, ARCH)

#define DEFINE_NVIDIA_GETRF(N)                          _GLASS_GETRF_NO_BD_E(N, SMS)
#define DEFINE_NVIDIA_GETRF_BLOCKDIM(N, TC)             _GLASS_GETRF_BD_E(N, TC, SMS)
#define DEFINE_NVIDIA_GETRF_SM(N, SM)                   _GLASS_GETRF_NO_BD_E(N, SM)
#define DEFINE_NVIDIA_GETRF_BLOCKDIM_SM(N, TC, SM)      _GLASS_GETRF_BD_E(N, TC, SM)

// --- getrs_no_pivot (LU solve given the factor) -----------------------------
// Solves LU·X = B given the LU factor from getrf_no_pivot.  B is overwritten.
// Smem layout: [LUs: N*N] [Bs: N*NRHS] [solver_smem].

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
__device__ void getrs_no_pivot(const T* LU, T* B, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::getrs_no_pivot<T,N,NRHS,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_GETRS* in your .cu file.");
}

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t getrs_no_pivot_smem_size() { return 0; }

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t getrs_no_pivot_threads() { return 256; }

#define _GLASS_GETRS_NO_BD(N, NRHS, ARCH)                                                        \
    namespace _nvidia_getrs_impl_##N##x##NRHS##_bd0_sm##ARCH {                                   \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::getrs_no_pivot>()                       \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block());                                                              \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(const float* LU, float* B, char* smem) {                      \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* LUs = reinterpret_cast<float*>(smem);                                         \
            float* Bs = LUs + (N * N);                                                           \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(LU, N, LUs, N);            \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            SOLVER().execute(LUs, N, Bs, N);                                                     \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void getrs_no_pivot<float, N, NRHS, 0, ARCH>(const float* LU, float* B, char* smem) \
    { _nvidia_getrs_impl_##N##x##NRHS##_bd0_sm##ARCH::run(LU, B, smem); }                        \
    template <>                                                                                  \
    constexpr std::size_t getrs_no_pivot_smem_size<float, N, NRHS, 0, ARCH>()                   \
    { return _nvidia_getrs_impl_##N##x##NRHS##_bd0_sm##ARCH::smem_bytes; }                       \
    template <>                                                                                  \
    constexpr uint32_t getrs_no_pivot_threads<float, N, NRHS, 0, ARCH>()                        \
    { return _nvidia_getrs_impl_##N##x##NRHS##_bd0_sm##ARCH::block_threads; }

#define _GLASS_GETRS_BD(N, NRHS, TC, ARCH)                                                       \
    namespace _nvidia_getrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH {                              \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::getrs_no_pivot>()                       \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block()                                                                \
            + cusolverdx::BlockDim<TC>());                                                       \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(const float* LU, float* B, char* smem) {                      \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* LUs = reinterpret_cast<float*>(smem);                                         \
            float* Bs = LUs + (N * N);                                                           \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(LU, N, LUs, N);            \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            SOLVER().execute(LUs, N, Bs, N);                                                     \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void getrs_no_pivot<float, N, NRHS, TC, ARCH>(const float* LU, float* B, char* smem) \
    { _nvidia_getrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::run(LU, B, smem); }                   \
    template <>                                                                                  \
    constexpr std::size_t getrs_no_pivot_smem_size<float, N, NRHS, TC, ARCH>()                  \
    { return _nvidia_getrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::smem_bytes; }                  \
    template <>                                                                                  \
    constexpr uint32_t getrs_no_pivot_threads<float, N, NRHS, TC, ARCH>()                       \
    { return _nvidia_getrs_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::block_threads; }

#define _GLASS_GETRS_NO_BD_E(N, NRHS, ARCH)        _GLASS_GETRS_NO_BD(N, NRHS, ARCH)
#define _GLASS_GETRS_BD_E(N, NRHS, TC, ARCH)       _GLASS_GETRS_BD(N, NRHS, TC, ARCH)

#define DEFINE_NVIDIA_GETRS(N, NRHS)                         _GLASS_GETRS_NO_BD_E(N, NRHS, SMS)
#define DEFINE_NVIDIA_GETRS_BLOCKDIM(N, NRHS, TC)            _GLASS_GETRS_BD_E(N, NRHS, TC, SMS)
#define DEFINE_NVIDIA_GETRS_SM(N, NRHS, SM)                  _GLASS_GETRS_NO_BD_E(N, NRHS, SM)
#define DEFINE_NVIDIA_GETRS_BLOCKDIM_SM(N, NRHS, TC, SM)     _GLASS_GETRS_BD_E(N, NRHS, TC, SM)

// --- gesv_no_pivot (LU + solve in one call) ---------------------------------
// LU-factor A (no pivoting) and solve A·X = B.  A is destroyed; B := X.
// Smem layout: [As: N*N] [Bs: N*NRHS] [solver_smem].

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS,
          bool TRAILING_SYNC = true>
__device__ void gesv_no_pivot(T* A, T* B, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::gesv_no_pivot<T,N,NRHS,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_GESV* in your .cu file.");
}

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t gesv_no_pivot_smem_size() { return 0; }

template <typename T, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t gesv_no_pivot_threads() { return 256; }

#define _GLASS_GESV_NO_BD(N, NRHS, ARCH)                                                         \
    namespace _nvidia_gesv_impl_##N##x##NRHS##_bd0_sm##ARCH {                                    \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::gesv_no_pivot>()                        \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block());                                                              \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        template <bool TRAILING_SYNC>                                                            \
        __device__ inline void run(float* A, float* B, char* smem) {                             \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            float* Bs = As + (N * N);                                                            \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            int info = 0;                                                                        \
            SOLVER().execute(As, N, Bs, N, &info);                                               \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, N, A, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            if constexpr (TRAILING_SYNC) {                                                       \
                __syncthreads();                                                                 \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void gesv_no_pivot<float, N, NRHS, 0, ARCH, true>(float* A, float* B, char* smem) \
    { _nvidia_gesv_impl_##N##x##NRHS##_bd0_sm##ARCH::template run<true>(A, B, smem); }           \
    template <>                                                                                  \
    __device__ inline void gesv_no_pivot<float, N, NRHS, 0, ARCH, false>(float* A, float* B, char* smem) \
    { _nvidia_gesv_impl_##N##x##NRHS##_bd0_sm##ARCH::template run<false>(A, B, smem); }          \
    template <>                                                                                  \
    constexpr std::size_t gesv_no_pivot_smem_size<float, N, NRHS, 0, ARCH>()                    \
    { return _nvidia_gesv_impl_##N##x##NRHS##_bd0_sm##ARCH::smem_bytes; }                        \
    template <>                                                                                  \
    constexpr uint32_t gesv_no_pivot_threads<float, N, NRHS, 0, ARCH>()                         \
    { return _nvidia_gesv_impl_##N##x##NRHS##_bd0_sm##ARCH::block_threads; }

#define _GLASS_GESV_BD(N, NRHS, TC, ARCH)                                                        \
    namespace _nvidia_gesv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH {                               \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<N, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::gesv_no_pivot>()                        \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block()                                                                \
            + cusolverdx::BlockDim<TC>());                                                       \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        template <bool TRAILING_SYNC>                                                            \
        __device__ inline void run(float* A, float* B, char* smem) {                             \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            float* Bs = As + (N * N);                                                            \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(A, N, As, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(B, N, Bs, N);           \
            __syncthreads();                                                                     \
            int info = 0;                                                                        \
            SOLVER().execute(As, N, Bs, N, &info);                                               \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, N, cusolverdx::col_major>(As, N, A, N);              \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, N, B, N);           \
            if constexpr (TRAILING_SYNC) {                                                       \
                __syncthreads();                                                                 \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void gesv_no_pivot<float, N, NRHS, TC, ARCH, true>(float* A, float* B, char* smem) \
    { _nvidia_gesv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::template run<true>(A, B, smem); }      \
    template <>                                                                                  \
    __device__ inline void gesv_no_pivot<float, N, NRHS, TC, ARCH, false>(float* A, float* B, char* smem) \
    { _nvidia_gesv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::template run<false>(A, B, smem); }     \
    template <>                                                                                  \
    constexpr std::size_t gesv_no_pivot_smem_size<float, N, NRHS, TC, ARCH>()                   \
    { return _nvidia_gesv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::smem_bytes; }                   \
    template <>                                                                                  \
    constexpr uint32_t gesv_no_pivot_threads<float, N, NRHS, TC, ARCH>()                        \
    { return _nvidia_gesv_impl_##N##x##NRHS##_bd##TC##_sm##ARCH::block_threads; }

#define _GLASS_GESV_NO_BD_E(N, NRHS, ARCH)        _GLASS_GESV_NO_BD(N, NRHS, ARCH)
#define _GLASS_GESV_BD_E(N, NRHS, TC, ARCH)       _GLASS_GESV_BD(N, NRHS, TC, ARCH)

#define DEFINE_NVIDIA_GESV(N, NRHS)                          _GLASS_GESV_NO_BD_E(N, NRHS, SMS)
#define DEFINE_NVIDIA_GESV_BLOCKDIM(N, NRHS, TC)             _GLASS_GESV_BD_E(N, NRHS, TC, SMS)
#define DEFINE_NVIDIA_GESV_SM(N, NRHS, SM)                   _GLASS_GESV_NO_BD_E(N, NRHS, SM)
#define DEFINE_NVIDIA_GESV_BLOCKDIM_SM(N, NRHS, TC, SM)      _GLASS_GESV_BD_E(N, NRHS, TC, SM)

// --- geqrf (QR factorization) -----------------------------------------------
// In-place QR factorization of an M×N matrix.  A := QR (R upper, Householder
// reflectors below diag).  `tau` is a min(M,N)-element output array.
// Smem layout: [As: M*N] [tau: min(M,N)] [solver_smem].
// The caller must provide `tau` storage of size min(M,N) floats (in shared,
// pointed at by `tau`); call site responsibility.

template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
__device__ void geqrf(T* A, T* tau, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::geqrf<T,M,N,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_GEQRF* in your .cu file.");
}

template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t geqrf_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t geqrf_threads() { return 256; }

#define _GLASS_GEQRF_NO_BD(M, N, ARCH)                                                           \
    namespace _nvidia_geqrf_impl_##M##x##N##_bd0_sm##ARCH {                                      \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<M, N>()                                                             \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::geqrf>()                                \
            + cusolverdx::Arrangement<cusolverdx::col_major>()                                   \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block());                                                              \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(float* A, float* tau, char* smem) {                           \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(A, M, As, M);              \
            __syncthreads();                                                                     \
            SOLVER().execute(As, M, tau);                                                        \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(As, M, A, M);              \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void geqrf<float, M, N, 0, ARCH>(float* A, float* tau, char* smem)        \
    { _nvidia_geqrf_impl_##M##x##N##_bd0_sm##ARCH::run(A, tau, smem); }                          \
    template <>                                                                                  \
    constexpr std::size_t geqrf_smem_size<float, M, N, 0, ARCH>()                               \
    { return _nvidia_geqrf_impl_##M##x##N##_bd0_sm##ARCH::smem_bytes; }                          \
    template <>                                                                                  \
    constexpr uint32_t geqrf_threads<float, M, N, 0, ARCH>()                                    \
    { return _nvidia_geqrf_impl_##M##x##N##_bd0_sm##ARCH::block_threads; }

#define _GLASS_GEQRF_BD(M, N, TC, ARCH)                                                          \
    namespace _nvidia_geqrf_impl_##M##x##N##_bd##TC##_sm##ARCH {                                 \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<M, N>()                                                             \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::geqrf>()                                \
            + cusolverdx::Arrangement<cusolverdx::col_major>()                                   \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block()                                                                \
            + cusolverdx::BlockDim<TC>());                                                       \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        __device__ inline void run(float* A, float* tau, char* smem) {                           \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            float* As = reinterpret_cast<float*>(smem);                                          \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(A, M, As, M);              \
            __syncthreads();                                                                     \
            SOLVER().execute(As, M, tau);                                                        \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(As, M, A, M);              \
            __syncthreads();                                                                     \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void geqrf<float, M, N, TC, ARCH>(float* A, float* tau, char* smem)       \
    { _nvidia_geqrf_impl_##M##x##N##_bd##TC##_sm##ARCH::run(A, tau, smem); }                     \
    template <>                                                                                  \
    constexpr std::size_t geqrf_smem_size<float, M, N, TC, ARCH>()                              \
    { return _nvidia_geqrf_impl_##M##x##N##_bd##TC##_sm##ARCH::smem_bytes; }                     \
    template <>                                                                                  \
    constexpr uint32_t geqrf_threads<float, M, N, TC, ARCH>()                                   \
    { return _nvidia_geqrf_impl_##M##x##N##_bd##TC##_sm##ARCH::block_threads; }

#define _GLASS_GEQRF_NO_BD_E(M, N, ARCH)        _GLASS_GEQRF_NO_BD(M, N, ARCH)
#define _GLASS_GEQRF_BD_E(M, N, TC, ARCH)       _GLASS_GEQRF_BD(M, N, TC, ARCH)

#define DEFINE_NVIDIA_GEQRF(M, N)                          _GLASS_GEQRF_NO_BD_E(M, N, SMS)
#define DEFINE_NVIDIA_GEQRF_BLOCKDIM(M, N, TC)             _GLASS_GEQRF_BD_E(M, N, TC, SMS)
#define DEFINE_NVIDIA_GEQRF_SM(M, N, SM)                   _GLASS_GEQRF_NO_BD_E(M, N, SM)
#define DEFINE_NVIDIA_GEQRF_BLOCKDIM_SM(M, N, TC, SM)      _GLASS_GEQRF_BD_E(M, N, TC, SM)

// --- gels (least-squares solve) ---------------------------------------------
// Solves min ||A·X - B|| for over- or under-determined A.  cuSOLVERDx picks
// QR (M >= N) or LQ (M < N) internally.  A is destroyed; B := X.
// `tau` workspace is min(M,N) floats.
// Smem layout: [As: M*N] [Bs: max(M,N)*NRHS] [solver_smem]; tau is caller-provided.

template <typename T, uint32_t M, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS,
          bool TRAILING_SYNC = true>
__device__ void gels(T* A, T* tau, T* B, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::gels<T,M,N,NRHS,BLOCK_THREADS,SM_VAL> not available — "
        "add DEFINE_NVIDIA_GELS* in your .cu file.");
}

template <typename T, uint32_t M, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr std::size_t gels_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N, uint32_t NRHS,
          uint32_t BLOCK_THREADS = 0, uint32_t SM_VAL = SMS>
constexpr uint32_t gels_threads() { return 256; }

#define _GLASS_GELS_BMAX(M, N) ((M) > (N) ? (M) : (N))

#define _GLASS_GELS_NO_BD(M, N, NRHS, ARCH)                                                      \
    namespace _nvidia_gels_impl_##M##x##N##x##NRHS##_bd0_sm##ARCH {                              \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<M, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::gels>()                                 \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block());                                                              \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        template <bool TRAILING_SYNC>                                                            \
        __device__ inline void run(float* A, float* tau, float* B, char* smem) {                 \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            constexpr uint32_t BMAX = _GLASS_GELS_BMAX(M, N);                                    \
            float* As = reinterpret_cast<float*>(smem);                                          \
            float* Bs = As + (M * N);                                                            \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(A, M, As, M);              \
            cusolverdx::copy_2d<SOLVER, M, NRHS, cusolverdx::col_major>(B, BMAX, Bs, BMAX);     \
            __syncthreads();                                                                     \
            SOLVER().execute(As, M, tau, Bs, BMAX);                                              \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, BMAX, B, BMAX);     \
            if constexpr (TRAILING_SYNC) {                                                       \
                __syncthreads();                                                                 \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void gels<float, M, N, NRHS, 0, ARCH, true>(float* A, float* tau, float* B, char* smem) \
    { _nvidia_gels_impl_##M##x##N##x##NRHS##_bd0_sm##ARCH::template run<true>(A, tau, B, smem); }\
    template <>                                                                                  \
    __device__ inline void gels<float, M, N, NRHS, 0, ARCH, false>(float* A, float* tau, float* B, char* smem) \
    { _nvidia_gels_impl_##M##x##N##x##NRHS##_bd0_sm##ARCH::template run<false>(A, tau, B, smem); }\
    template <>                                                                                  \
    constexpr std::size_t gels_smem_size<float, M, N, NRHS, 0, ARCH>()                          \
    { return _nvidia_gels_impl_##M##x##N##x##NRHS##_bd0_sm##ARCH::smem_bytes; }                  \
    template <>                                                                                  \
    constexpr uint32_t gels_threads<float, M, N, NRHS, 0, ARCH>()                               \
    { return _nvidia_gels_impl_##M##x##N##x##NRHS##_bd0_sm##ARCH::block_threads; }

#define _GLASS_GELS_BD(M, N, NRHS, TC, ARCH)                                                     \
    namespace _nvidia_gels_impl_##M##x##N##x##NRHS##_bd##TC##_sm##ARCH {                         \
        using SOLVER = decltype(                                                                 \
            cusolverdx::Size<M, N, NRHS>()                                                       \
            + cusolverdx::Precision<float>()                                                     \
            + cusolverdx::Type<cusolverdx::type::real>()                                         \
            + cusolverdx::Function<cusolverdx::function::gels>()                                 \
            + cusolverdx::Arrangement<cusolverdx::col_major, cusolverdx::col_major>()            \
            + cusolverdx::SM<ARCH>()                                                             \
            + cusolverdx::Block()                                                                \
            + cusolverdx::BlockDim<TC>());                                                       \
        static constexpr uint32_t block_threads =                                                \
            static_cast<uint32_t>(SOLVER::block_dim.x *                                          \
                                  SOLVER::block_dim.y *                                          \
                                  SOLVER::block_dim.z);                                          \
        static constexpr std::size_t smem_bytes = SOLVER::shared_memory_size;                    \
        template <bool TRAILING_SYNC>                                                            \
        __device__ inline void run(float* A, float* tau, float* B, char* smem) {                 \
            _GLASS_ASSERT_BLOCKDIM_GEQ(SOLVER)                                                   \
            constexpr uint32_t BMAX = _GLASS_GELS_BMAX(M, N);                                    \
            float* As = reinterpret_cast<float*>(smem);                                          \
            float* Bs = As + (M * N);                                                            \
            cusolverdx::copy_2d<SOLVER, M, N, cusolverdx::col_major>(A, M, As, M);              \
            cusolverdx::copy_2d<SOLVER, M, NRHS, cusolverdx::col_major>(B, BMAX, Bs, BMAX);     \
            __syncthreads();                                                                     \
            SOLVER().execute(As, M, tau, Bs, BMAX);                                              \
            __syncthreads();                                                                     \
            cusolverdx::copy_2d<SOLVER, N, NRHS, cusolverdx::col_major>(Bs, BMAX, B, BMAX);     \
            if constexpr (TRAILING_SYNC) {                                                       \
                __syncthreads();                                                                 \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    template <>                                                                                  \
    __device__ inline void gels<float, M, N, NRHS, TC, ARCH, true>(float* A, float* tau, float* B, char* smem) \
    { _nvidia_gels_impl_##M##x##N##x##NRHS##_bd##TC##_sm##ARCH::template run<true>(A, tau, B, smem); }\
    template <>                                                                                  \
    __device__ inline void gels<float, M, N, NRHS, TC, ARCH, false>(float* A, float* tau, float* B, char* smem) \
    { _nvidia_gels_impl_##M##x##N##x##NRHS##_bd##TC##_sm##ARCH::template run<false>(A, tau, B, smem); }\
    template <>                                                                                  \
    constexpr std::size_t gels_smem_size<float, M, N, NRHS, TC, ARCH>()                         \
    { return _nvidia_gels_impl_##M##x##N##x##NRHS##_bd##TC##_sm##ARCH::smem_bytes; }             \
    template <>                                                                                  \
    constexpr uint32_t gels_threads<float, M, N, NRHS, TC, ARCH>()                              \
    { return _nvidia_gels_impl_##M##x##N##x##NRHS##_bd##TC##_sm##ARCH::block_threads; }

#define _GLASS_GELS_NO_BD_E(M, N, NRHS, ARCH)        _GLASS_GELS_NO_BD(M, N, NRHS, ARCH)
#define _GLASS_GELS_BD_E(M, N, NRHS, TC, ARCH)       _GLASS_GELS_BD(M, N, NRHS, TC, ARCH)

#define DEFINE_NVIDIA_GELS(M, N, NRHS)                          _GLASS_GELS_NO_BD_E(M, N, NRHS, SMS)
#define DEFINE_NVIDIA_GELS_BLOCKDIM(M, N, NRHS, TC)             _GLASS_GELS_BD_E(M, N, NRHS, TC, SMS)
#define DEFINE_NVIDIA_GELS_SM(M, N, NRHS, SM)                   _GLASS_GELS_NO_BD_E(M, N, NRHS, SM)
#define DEFINE_NVIDIA_GELS_BLOCKDIM_SM(M, N, NRHS, TC, SM)      _GLASS_GELS_BD_E(M, N, NRHS, TC, SM)
