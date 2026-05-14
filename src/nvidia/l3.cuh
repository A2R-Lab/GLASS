#pragma once
#include <cstdint>
#include <cublasdx.hpp>
// types.cuh provides the `layout` enum and the shared
// _GLASS_CUBLAS_LAYOUT / _GLASS_ASSERT_BLOCKDIM_GEQ helper macros.
#include "./types.cuh"

// glass::nvidia L3 — cuBLASDx-backed gemm
//
// All sizes are compile-time. Call one of the DEFINE_NVIDIA_GEMM* macros once per
// (M, N, K, BLOCK_THREADS, layouts, SM) combination you need, then call
// glass::nvidia::gemm<...>(alpha, A, B, beta, C, smem) inside your kernel.
//
// Backward-compatible defaults:
//   BLOCK_THREADS = 0           -> let cuBLASDx pick block_dim from its database
//   LA = LB = LC = col_major    -> standard column-major (no transpose)
//   SM_VAL = SMS                -> SMS macro (default 860)
//
// Example (basic):
//   DEFINE_NVIDIA_GEMM(6, 6, 6)
//   constexpr auto smem    = glass::nvidia::gemm_smem_size<float, 6, 6, 6>();
//   constexpr auto threads = glass::nvidia::gemm_threads<float, 6, 6, 6>();
//   kernel<<<1, threads, smem>>>(...);
//   glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, smem_ptr);
//
// Example (caller-controlled BlockDim — fixes the deadlock when launching with
// a thread count not chosen by cuBLASDx's database):
//   DEFINE_NVIDIA_GEMM_BLOCKDIM(6, 6, 6, 352)
//   kernel<<<1, 352, smem>>>(...);
//   glass::nvidia::gemm<float, 6, 6, 6, 352>(1.f, A, B, 0.f, C, smem_ptr);
//
// Example (transpose B — A * B^T):
//   DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB(6, 6, 6, 352)
//   glass::nvidia::gemm<float, 6, 6, 6, 352,
//                       glass::nvidia::layout::col_major,
//                       glass::nvidia::layout::row_major,
//                       glass::nvidia::layout::col_major>(...);
//
// Example (multi-arch dispatch with explicit SM):
//   DEFINE_NVIDIA_GEMM_BLOCKDIM_SM(6, 6, 6, 352, 890)
//   DEFINE_NVIDIA_GEMM_BLOCKDIM_SM(6, 6, 6, 352, 1200)
//   glass::nvidia::gemm<float, 6, 6, 6, 352,
//       glass::nvidia::layout::col_major,
//       glass::nvidia::layout::col_major,
//       glass::nvidia::layout::col_major,
//       #if __CUDA_ARCH__ >= 1200
//           1200
//       #else
//           890
//       #endif
//   >(...);

#ifndef SMS
#define SMS 860
#endif

// ---------------------------------------------------------------------------
// Primary templates — instantiated by the DEFINE_NVIDIA_GEMM* macros below.
// Default arguments make existing callers `gemm<float, M, N, K>(...)` resolve
// to `gemm<float, M, N, K, 0, col_major, col_major, col_major, SMS>(...)`.
//
// The primary template now AUTO-DISPATCHES based on
// `should_use_cublasdx<T, M, N, K, SM_VAL>()` (query.cuh / tuning_table.cuh):
//   - returns false  → routes to ::glass::gemm<T,M,N,K,TRANSPOSE_B,ROW_MAJOR>
//                      (pure-SIMT, no cuBLASDx scratch needed)
//   - returns true   → static_assert that a DEFINE_NVIDIA_GEMM* macro
//                      specialized this signature. Macro-emitted explicit
//                      specializations beat the primary template at lookup,
//                      so when present they are picked unconditionally.
//
// Layout flags map onto SIMT booleans:
//   TRANSPOSE_B = (LB == layout::row_major)            // A * B^T
//   ROW_MAJOR   = (LA == row_major) && (LC == row_major)
// SIMT does not support arbitrary mixed layouts; mixed combinations that
// can't be expressed via the two flags above force the static_assert path,
// requiring an explicit DEFINE_NVIDIA_GEMM_LAYOUT* specialization.
// ---------------------------------------------------------------------------

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
__device__ void gemm(T alpha, T* A, T* B, T beta, T* C, char* smem)
{
    if constexpr (!should_use_cublasdx<T, M, N, K, SM_VAL>()) {
        constexpr bool TRANSPOSE_B = (LB == layout::row_major);
        constexpr bool ROW_MAJOR   =
            (LA == layout::row_major) && (LC == layout::row_major);
        ::glass::gemm<T, M, N, K, TRANSPOSE_B, ROW_MAJOR>(alpha, A, B, beta, C);
    } else {
        static_assert(sizeof(T) == 0,
            "glass::nvidia::gemm<T,M,N,K,BLOCK_THREADS,LA,LB,LC,SM_VAL>: "
            "should_use_cublasdx<> returned true for this shape but no "
            "DEFINE_NVIDIA_GEMM* macro is in scope. Add one in your .cu, or "
            "override the dispatch via tuning_table.cuh / GLASS_TUNING_TABLE_LOCAL.");
    }
}

// Returns the cuBLASDx scratch size for this signature; 0 when the auto-
// dispatch would route to SIMT (which needs no scratch). DEFINE_NVIDIA_GEMM*
// macros emit explicit specializations that return the real cuBLASDx size,
// taking precedence over this primary.
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr std::size_t gemm_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr uint32_t gemm_threads() { return 256; }

// ---------------------------------------------------------------------------
// Private core macros — one for "no BlockDim" (cuBLASDx picks block_dim from
// its database) and one for "BlockDim<TC,1,1>" (caller pins it). They differ
// only in the namespace suffix and the optional `+ cublasdx::BlockDim<...>()`
// in the type expression.
// ---------------------------------------------------------------------------

// LA, LB, LC are integer literals 0 (col_major) or 1 (row_major).
// ARCH is an integer literal SM architecture (e.g. 860). The parameter is
// named ARCH (not SM) to avoid colliding with the cublasdx::SM<> token during
// macro substitution.
#define _GLASS_GEMM_NO_BD(M, N, K, LA, LB, LC, ARCH)                                            \
    namespace _nvidia_gemm_impl_##M##x##N##x##K##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH {     \
        using GEMM = decltype(                                                                  \
            cublasdx::Size<M, N, K>()                                                           \
            + cublasdx::Precision<float>()                                                      \
            + cublasdx::Type<cublasdx::type::real>()                                            \
            + cublasdx::Function<cublasdx::function::MM>()                                      \
            + cublasdx::Arrangement<                                                            \
                  _GLASS_CUBLAS_LAYOUT(LA),                                                     \
                  _GLASS_CUBLAS_LAYOUT(LB),                                                     \
                  _GLASS_CUBLAS_LAYOUT(LC)>()                                                   \
            + cublasdx::SM<ARCH>()                                                              \
            + cublasdx::Block());                                                               \
        static constexpr uint32_t block_threads =                                               \
            static_cast<uint32_t>(GEMM::block_dim.x *                                           \
                                  GEMM::block_dim.y *                                           \
                                  GEMM::block_dim.z);                                           \
        static constexpr std::size_t smem_bytes =                                               \
            cublasdx::get_shared_storage_size<GEMM>();                                          \
        __device__ inline void run(float alpha, float* A, float* B,                             \
                                   float beta,  float* C, char* smem)                           \
        {                                                                                       \
            _GLASS_ASSERT_BLOCKDIM_GEQ(GEMM)                                                    \
            using align = cublasdx::alignment_of<GEMM>;                                         \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);         \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());            \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());            \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());            \
            cublasdx::copy<GEMM, align::a>(                                                     \
                cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);                  \
            cublasdx::copy<GEMM, align::b>(                                                     \
                cublasdx::make_tensor(B, GEMM::get_layout_gmem_b()), b_smem);                  \
            cublasdx::copy<GEMM, align::c>(                                                     \
                cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()), c_smem);                  \
            cublasdx::copy_wait();                                                              \
            GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                               \
            __syncthreads();                                                                    \
            cublasdx::copy<GEMM, align::c>(                                                     \
                c_smem, cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()));                  \
            __syncthreads();                                                                    \
        }                                                                                       \
    }                                                                                           \
    template <>                                                                                 \
    __device__ inline void gemm<float, M, N, K, 0,                                              \
                                static_cast<layout>(LA),                                        \
                                static_cast<layout>(LB),                                        \
                                static_cast<layout>(LC), ARCH>                                  \
                               (float alpha, float* A, float* B,                                \
                                float beta,  float* C, char* smem)                              \
    {                                                                                           \
        _nvidia_gemm_impl_##M##x##N##x##K##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH::run(       \
            alpha, A, B, beta, C, smem);                                                        \
    }                                                                                           \
    template <>                                                                                 \
    constexpr std::size_t gemm_smem_size<float, M, N, K, 0,                                     \
                                          static_cast<layout>(LA),                              \
                                          static_cast<layout>(LB),                              \
                                          static_cast<layout>(LC), ARCH>()                      \
    {                                                                                           \
        return _nvidia_gemm_impl_##M##x##N##x##K##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH::smem_bytes; \
    }                                                                                           \
    template <>                                                                                 \
    constexpr uint32_t gemm_threads<float, M, N, K, 0,                                          \
                                     static_cast<layout>(LA),                                   \
                                     static_cast<layout>(LB),                                   \
                                     static_cast<layout>(LC), ARCH>()                           \
    {                                                                                           \
        return _nvidia_gemm_impl_##M##x##N##x##K##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH::block_threads; \
    }

// LA, LB, LC are integer literals 0 (col_major) or 1 (row_major).
// TC is the pinned BlockDim thread count (1D); ARCH is the SM architecture
// (parameter name avoids colliding with cublasdx::SM token).
#define _GLASS_GEMM_BD(M, N, K, TC, LA, LB, LC, ARCH)                                           \
    namespace _nvidia_gemm_impl_##M##x##N##x##K##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH { \
        using GEMM = decltype(                                                                  \
            cublasdx::Size<M, N, K>()                                                           \
            + cublasdx::Precision<float>()                                                      \
            + cublasdx::Type<cublasdx::type::real>()                                            \
            + cublasdx::Function<cublasdx::function::MM>()                                      \
            + cublasdx::Arrangement<                                                            \
                  _GLASS_CUBLAS_LAYOUT(LA),                                                     \
                  _GLASS_CUBLAS_LAYOUT(LB),                                                     \
                  _GLASS_CUBLAS_LAYOUT(LC)>()                                                   \
            + cublasdx::SM<ARCH>()                                                              \
            + cublasdx::Block()                                                                 \
            + cublasdx::BlockDim<TC, 1, 1>());                                                  \
        static constexpr uint32_t block_threads =                                               \
            static_cast<uint32_t>(GEMM::block_dim.x *                                           \
                                  GEMM::block_dim.y *                                           \
                                  GEMM::block_dim.z);                                           \
        static constexpr std::size_t smem_bytes =                                               \
            cublasdx::get_shared_storage_size<GEMM>();                                          \
        __device__ inline void run(float alpha, float* A, float* B,                             \
                                   float beta,  float* C, char* smem)                           \
        {                                                                                       \
            _GLASS_ASSERT_BLOCKDIM_GEQ(GEMM)                                                    \
            using align = cublasdx::alignment_of<GEMM>;                                         \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);         \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());            \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());            \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());            \
            cublasdx::copy<GEMM, align::a>(                                                     \
                cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);                  \
            cublasdx::copy<GEMM, align::b>(                                                     \
                cublasdx::make_tensor(B, GEMM::get_layout_gmem_b()), b_smem);                  \
            cublasdx::copy<GEMM, align::c>(                                                     \
                cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()), c_smem);                  \
            cublasdx::copy_wait();                                                              \
            GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                               \
            __syncthreads();                                                                    \
            cublasdx::copy<GEMM, align::c>(                                                     \
                c_smem, cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()));                  \
            __syncthreads();                                                                    \
        }                                                                                       \
    }                                                                                           \
    template <>                                                                                 \
    __device__ inline void gemm<float, M, N, K, TC,                                             \
                                static_cast<layout>(LA),                                        \
                                static_cast<layout>(LB),                                        \
                                static_cast<layout>(LC), ARCH>                                  \
                               (float alpha, float* A, float* B,                                \
                                float beta,  float* C, char* smem)                              \
    {                                                                                           \
        _nvidia_gemm_impl_##M##x##N##x##K##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::run(  \
            alpha, A, B, beta, C, smem);                                                        \
    }                                                                                           \
    template <>                                                                                 \
    constexpr std::size_t gemm_smem_size<float, M, N, K, TC,                                    \
                                          static_cast<layout>(LA),                              \
                                          static_cast<layout>(LB),                              \
                                          static_cast<layout>(LC), ARCH>()                     \
    {                                                                                           \
        return _nvidia_gemm_impl_##M##x##N##x##K##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::smem_bytes; \
    }                                                                                           \
    template <>                                                                                 \
    constexpr uint32_t gemm_threads<float, M, N, K, TC,                                         \
                                     static_cast<layout>(LA),                                   \
                                     static_cast<layout>(LB),                                   \
                                     static_cast<layout>(LC), ARCH>()                          \
    {                                                                                           \
        return _nvidia_gemm_impl_##M##x##N##x##K##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::block_threads; \
    }

// One-line indirection wrappers — exist solely to force the SMS macro (or any
// macro passed as ARCH) to be expanded at the call boundary, BEFORE token
// pasting in `_GLASS_GEMM_*` would freeze it as the literal text "SMS".
#define _GLASS_GEMM_NO_BD_E(M, N, K, LA, LB, LC, ARCH)                                          \
    _GLASS_GEMM_NO_BD(M, N, K, LA, LB, LC, ARCH)
#define _GLASS_GEMM_BD_E(M, N, K, TC, LA, LB, LC, ARCH)                                         \
    _GLASS_GEMM_BD(M, N, K, TC, LA, LB, LC, ARCH)

// ---------------------------------------------------------------------------
// Public DEFINE_NVIDIA_GEMM* convenience macros
// ---------------------------------------------------------------------------

// All public macros route through the _E indirection so SMS (or any macro
// passed for SM) is expanded before token pasting in the implementation macro.

// Default: cuBLASDx picks block_dim, all col_major, SM = SMS macro.
#define DEFINE_NVIDIA_GEMM(M, N, K) \
    _GLASS_GEMM_NO_BD_E(M, N, K, 0, 0, 0, SMS)

// Pinned BlockDim<TC,1,1>; all col_major; SM = SMS.
#define DEFINE_NVIDIA_GEMM_BLOCKDIM(M, N, K, TC) \
    _GLASS_GEMM_BD_E(M, N, K, TC, 0, 0, 0, SMS)

// Custom layouts; cuBLASDx picks block_dim; SM = SMS.
// LA, LB, LC are integer literals 0 (col_major) or 1 (row_major).
#define DEFINE_NVIDIA_GEMM_LAYOUT(M, N, K, LA, LB, LC) \
    _GLASS_GEMM_NO_BD_E(M, N, K, LA, LB, LC, SMS)

// Pinned BlockDim<TC,1,1>; custom layouts; SM = SMS.
#define DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT(M, N, K, TC, LA, LB, LC) \
    _GLASS_GEMM_BD_E(M, N, K, TC, LA, LB, LC, SMS)

// cuBLASDx picks block_dim; all col_major; explicit SM.
#define DEFINE_NVIDIA_GEMM_SM(M, N, K, SM) \
    _GLASS_GEMM_NO_BD_E(M, N, K, 0, 0, 0, SM)

// Pinned BlockDim<TC,1,1>; all col_major; explicit SM.
#define DEFINE_NVIDIA_GEMM_BLOCKDIM_SM(M, N, K, TC, SM) \
    _GLASS_GEMM_BD_E(M, N, K, TC, 0, 0, 0, SM)

// Custom layouts; cuBLASDx picks block_dim; explicit SM.
#define DEFINE_NVIDIA_GEMM_LAYOUT_SM(M, N, K, LA, LB, LC, SM) \
    _GLASS_GEMM_NO_BD_E(M, N, K, LA, LB, LC, SM)

// All explicit: pinned BlockDim<TC,1,1>; custom layouts; explicit SM.
#define DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT_SM(M, N, K, TC, LA, LB, LC, SM) \
    _GLASS_GEMM_BD_E(M, N, K, TC, LA, LB, LC, SM)

// ---------------------------------------------------------------------------
// GRiD-friendly aliases (matching the existing TRANSPOSE_B flag in the shim).
// "B transposed" maps to row_major B in cuBLASDx Arrangement.
// ---------------------------------------------------------------------------

#define DEFINE_NVIDIA_GEMM_TRANSB(M, N, K) \
    DEFINE_NVIDIA_GEMM_LAYOUT(M, N, K, 0, 1, 0)

#define DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB(M, N, K, TC) \
    DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT(M, N, K, TC, 0, 1, 0)

#define DEFINE_NVIDIA_GEMM_TRANSB_SM(M, N, K, SM) \
    DEFINE_NVIDIA_GEMM_LAYOUT_SM(M, N, K, 0, 1, 0, SM)

#define DEFINE_NVIDIA_GEMM_BLOCKDIM_TRANSB_SM(M, N, K, TC, SM) \
    DEFINE_NVIDIA_GEMM_BLOCKDIM_LAYOUT_SM(M, N, K, TC, 0, 1, 0, SM)

// ---------------------------------------------------------------------------
// row_strided_gemm: packs strided A and B into compact shared scratch, then
// delegates to the standard nvidia::gemm<...>. Forwards all template parameters
// (BLOCK_THREADS, layouts, SM) to the inner call so any DEFINE_NVIDIA_GEMM*
// variant works underneath.
//
// smem layout: [A_compact: M*N*sizeof(T)] [B_compact: N*K*sizeof(T)] [cuBLASDx smem]
// A_RS = leading dimension of column-major A: A[i][j] = A[i + j*A_RS].
// B_RS = leading dimension of column-major B: B[j][l] = B[j + l*B_RS].
// C is written as standard column-major with LDC=M (no strided output support).
// When A_RS==M and B_RS==N this degenerates to standard gemm with one pack pass.
// ---------------------------------------------------------------------------
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = N,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
__device__ void row_strided_gemm(T alpha, T* A, T* B, T beta, T* C, char* smem)
{
    if constexpr (!should_use_cublasdx_row_strided_gemm<T, M, N, K, A_RS, B_RS, SM_VAL>()) {
        // SIMT path: ::glass::row_strided_gemm uses the strides directly.
        // No packing, no scratch needed.
        ::glass::row_strided_gemm<T, M, N, K, A_RS, B_RS>(A, B, C, alpha, beta);
    } else {
        // cuBLASDx path: pack strided inputs into compact scratch, then
        // delegate to the cuBLASDx-specialized gemm<>.
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T* A_compact = reinterpret_cast<T*>(smem);
        T* B_compact = reinterpret_cast<T*>(smem + M * N * sizeof(T));
        char* cublas_smem = smem + (M * N + N * K) * sizeof(T);
        for (uint32_t i = rank; i < M * N; i += size) {
            uint32_t r = i % M, c = i / M;
            A_compact[r + c*M] = A[r + c*A_RS];
        }
        for (uint32_t i = rank; i < N * K; i += size) {
            uint32_t r = i % N, c = i / N;
            B_compact[r + c*N] = B[r + c*B_RS];
        }
        __syncthreads();
        gemm<T, M, N, K, BLOCK_THREADS, LA, LB, LC, SM_VAL>(
            alpha, A_compact, B_compact, beta, C, cublas_smem);
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = N,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr std::size_t row_strided_gemm_smem_size()
{
    if constexpr (!should_use_cublasdx_row_strided_gemm<T, M, N, K, A_RS, B_RS, SM_VAL>())
        return 0;
    else
        return (M * N + N * K) * sizeof(T)
             + gemm_smem_size<T, M, N, K, BLOCK_THREADS, LA, LB, LC, SM_VAL>();
}

// ---------------------------------------------------------------------------
// Batched GEMM (P2-7) — performs BATCH independent (M×N)·(N×K) GEMMs in a
// single CUDA block. Modeled after cuBLASDx example 05_gemm_batched: pin a 1D
// BlockDim<TC> per batch and launch with dim3(TC, BATCH). Threads with the
// same threadIdx.y participate in one batch's GEMM. Shared memory is laid out
// as BATCH copies of cuBLASDx's per-GEMM smem (a/b/c interleaved).
//
// The caller passes arrays of pointers (length BATCH) so each batch can be at
// an arbitrary global address. For contiguous batches, populate the array as
//   { base + 0*M*N, base + 1*M*N, ..., base + (BATCH-1)*M*N }.
//
// Required launch:  kernel<<<grid, dim3(TC, BATCH), gemm_batched_smem_size>>>
// Required smem:    glass::nvidia::gemm_batched_smem_size<T, M, N, K, BATCH, TC>()
// ---------------------------------------------------------------------------

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
__device__ void gemm_batched(T alpha, T* const* A, T* const* B,
                             T beta,  T* const* C, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::gemm_batched<T,M,N,K,BATCH,...> not available — "
        "add DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(M,N,K,BATCH,TC) in your .cu file.");
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr std::size_t gemm_batched_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr uint32_t gemm_batched_threads() { return 256; }

// Private core macro for batched GEMM. BlockDim<TC,1,1> + 2D launch dim3(TC, BATCH).
// ARCH (not SM) avoids token collision with cublasdx::SM<>.
#define _GLASS_GEMM_BATCHED_BD(M, N, K, BATCH, TC, LA, LB, LC, ARCH)                            \
    namespace _nvidia_gemm_batched_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH { \
        using GEMM = decltype(                                                                  \
            cublasdx::Size<M, N, K>()                                                           \
            + cublasdx::Precision<float>()                                                      \
            + cublasdx::Type<cublasdx::type::real>()                                            \
            + cublasdx::Function<cublasdx::function::MM>()                                      \
            + cublasdx::Arrangement<                                                            \
                  _GLASS_CUBLAS_LAYOUT(LA),                                                     \
                  _GLASS_CUBLAS_LAYOUT(LB),                                                     \
                  _GLASS_CUBLAS_LAYOUT(LC)>()                                                   \
            + cublasdx::SM<ARCH>()                                                              \
            + cublasdx::Block()                                                                 \
            + cublasdx::BlockDim<TC, 1, 1>());                                                  \
        static constexpr uint32_t per_batch_threads =                                           \
            static_cast<uint32_t>(GEMM::block_dim.x);                                           \
        static constexpr uint32_t total_threads = per_batch_threads * BATCH;                    \
        static constexpr std::size_t per_batch_smem =                                           \
            cublasdx::get_shared_storage_size<GEMM>();                                          \
        static constexpr std::size_t total_smem = per_batch_smem * BATCH;                       \
        __device__ inline void run(float alpha, float* const* A, float* const* B,               \
                                   float beta,  float* const* C, char* smem)                   \
        {                                                                                       \
            assert(blockDim.x >= per_batch_threads && blockDim.y >= BATCH &&                    \
                   "glass::nvidia::gemm_batched: launch dim3(>=TC, >=BATCH) required");         \
            const uint32_t b = threadIdx.y;                                                     \
            char* my_smem = smem + b * per_batch_smem;                                          \
            float* a = A[b];                                                                    \
            float* bp = B[b];                                                                   \
            float* c = C[b];                                                                    \
            using align = cublasdx::alignment_of<GEMM>;                                         \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(my_smem);      \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());            \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());            \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());            \
            cublasdx::copy<GEMM, align::a>(                                                     \
                cublasdx::make_tensor(a, GEMM::get_layout_gmem_a()), a_smem);                  \
            cublasdx::copy<GEMM, align::b>(                                                     \
                cublasdx::make_tensor(bp, GEMM::get_layout_gmem_b()), b_smem);                 \
            cublasdx::copy<GEMM, align::c>(                                                     \
                cublasdx::make_tensor(c, GEMM::get_layout_gmem_c()), c_smem);                  \
            cublasdx::copy_wait();                                                              \
            GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                               \
            __syncthreads();                                                                    \
            cublasdx::copy<GEMM, align::c>(                                                     \
                c_smem, cublasdx::make_tensor(c, GEMM::get_layout_gmem_c()));                  \
            __syncthreads();                                                                    \
        }                                                                                       \
    }                                                                                           \
    template <>                                                                                 \
    __device__ inline void gemm_batched<float, M, N, K, BATCH, TC,                             \
                                         static_cast<layout>(LA),                               \
                                         static_cast<layout>(LB),                               \
                                         static_cast<layout>(LC), ARCH>                        \
                                        (float alpha, float* const* A, float* const* B,        \
                                         float beta,  float* const* C, char* smem)             \
    {                                                                                           \
        _nvidia_gemm_batched_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::run( \
            alpha, A, B, beta, C, smem);                                                        \
    }                                                                                           \
    template <>                                                                                 \
    constexpr std::size_t gemm_batched_smem_size<float, M, N, K, BATCH, TC,                    \
                                                  static_cast<layout>(LA),                      \
                                                  static_cast<layout>(LB),                      \
                                                  static_cast<layout>(LC), ARCH>()             \
    {                                                                                           \
        return _nvidia_gemm_batched_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::total_smem; \
    }                                                                                           \
    template <>                                                                                 \
    constexpr uint32_t gemm_batched_threads<float, M, N, K, BATCH, TC,                         \
                                             static_cast<layout>(LA),                           \
                                             static_cast<layout>(LB),                           \
                                             static_cast<layout>(LC), ARCH>()                  \
    {                                                                                           \
        return _nvidia_gemm_batched_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::total_threads; \
    }

#define _GLASS_GEMM_BATCHED_BD_E(M, N, K, BATCH, TC, LA, LB, LC, ARCH)                          \
    _GLASS_GEMM_BATCHED_BD(M, N, K, BATCH, TC, LA, LB, LC, ARCH)

// Public batched GEMM convenience macros. BlockDim is required for batching, so
// we don't expose a "no-BlockDim" variant.
#define DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM(M, N, K, BATCH, TC) \
    _GLASS_GEMM_BATCHED_BD_E(M, N, K, BATCH, TC, 0, 0, 0, SMS)

#define DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM_LAYOUT(M, N, K, BATCH, TC, LA, LB, LC) \
    _GLASS_GEMM_BATCHED_BD_E(M, N, K, BATCH, TC, LA, LB, LC, SMS)

#define DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM_SM(M, N, K, BATCH, TC, SM) \
    _GLASS_GEMM_BATCHED_BD_E(M, N, K, BATCH, TC, 0, 0, 0, SM)

#define DEFINE_NVIDIA_GEMM_BATCHED_BLOCKDIM_LAYOUT_SM(M, N, K, BATCH, TC, LA, LB, LC, SM) \
    _GLASS_GEMM_BATCHED_BD_E(M, N, K, BATCH, TC, LA, LB, LC, SM)

// ---------------------------------------------------------------------------
// gemm_batched_1d — 1D-launch batched GEMM (auto-dispatch).
//
// Distinct from gemm_batched above (which requires a 2D launch dim3(TC,BATCH)
// and runs all BATCH gemms in parallel via threadIdx.y). The "_1d" variant
// takes a flat 1D launch (TC threads) and EITHER:
//   - SIMT route: strides every thread across BATCH*M*K output elements, no
//                 scratch needed. ::glass::gemm_batched_1d.
//   - cuBLASDx route: loops sequentially over batches, reusing one cuBLASDx
//                     scratch tile per iteration. Pinned BlockDim<TC,1,1>.
//
// Two input forms:
//   gemm_batched_1d<...>(alpha, A, B, beta, C, smem)
//        A, B, C are arrays of pointers (length BATCH). Each batch's matrices
//        can live at arbitrary addresses.
//   gemm_strided_batched_1d<...>(alpha, A, B, beta, C, smem)
//        A, B, C are single base pointers; batch b lives at base+b*STRIDE.
// ---------------------------------------------------------------------------

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
__device__ void gemm_batched_1d(T alpha, T* const* A, T* const* B,
                                T beta,  T* const* C, char* smem)
{
    if constexpr (!should_use_cublasdx_batched<T, M, N, K, BATCH, SM_VAL>()) {
        constexpr bool TRANSPOSE_B = (LB == layout::row_major);
        ::glass::gemm_batched_1d<T, M, N, K, BATCH, TRANSPOSE_B>(
            alpha, A, B, beta, C);
    } else {
        static_assert(sizeof(T) == 0,
            "glass::nvidia::gemm_batched_1d<T,M,N,K,BATCH,...>: "
            "should_use_cublasdx_batched<> returned true but no "
            "DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM* macro is in scope.");
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t A_STRIDE = M * N,
          uint32_t B_STRIDE = N * K,
          uint32_t C_STRIDE = M * K,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
__device__ void gemm_strided_batched_1d(T alpha, T* A, T* B,
                                        T beta,  T* C, char* smem)
{
    if constexpr (!should_use_cublasdx_batched<T, M, N, K, BATCH, SM_VAL>()) {
        constexpr bool TRANSPOSE_B = (LB == layout::row_major);
        ::glass::gemm_strided_batched_1d<T, M, N, K, BATCH,
                                         A_STRIDE, B_STRIDE, C_STRIDE,
                                         TRANSPOSE_B>(alpha, A, B, beta, C);
    } else {
        static_assert(sizeof(T) == 0,
            "glass::nvidia::gemm_strided_batched_1d<T,M,N,K,BATCH,...>: "
            "should_use_cublasdx_batched<> returned true but no "
            "DEFINE_NVIDIA_GEMM_STRIDED_BATCHED_1D_BLOCKDIM* macro is in scope.");
    }
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr std::size_t gemm_batched_1d_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t A_STRIDE = M * N,
          uint32_t B_STRIDE = N * K,
          uint32_t C_STRIDE = M * K,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr std::size_t gemm_strided_batched_1d_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N, uint32_t K, uint32_t BATCH,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr uint32_t gemm_batched_1d_threads() { return 256; }

// ---------------------------------------------------------------------------
// Private core macro for cuBLASDx-backed gemm_batched_1d. Sequential outer
// loop over BATCH; each iteration reuses the same cuBLASDx scratch tile.
// Single block, 1D launch dim3(TC,1,1). The smem requirement is per-batch
// (NOT scaled by BATCH).
// ---------------------------------------------------------------------------

#define _GLASS_GEMM_BATCHED_1D_BD(M, N, K, BATCH, TC, LA, LB, LC, ARCH)                       \
    namespace _nvidia_gemm_batched_1d_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH { \
        using GEMM = decltype(                                                                \
            cublasdx::Size<M, N, K>()                                                         \
            + cublasdx::Precision<float>()                                                    \
            + cublasdx::Type<cublasdx::type::real>()                                          \
            + cublasdx::Function<cublasdx::function::MM>()                                    \
            + cublasdx::Arrangement<                                                          \
                  _GLASS_CUBLAS_LAYOUT(LA),                                                   \
                  _GLASS_CUBLAS_LAYOUT(LB),                                                   \
                  _GLASS_CUBLAS_LAYOUT(LC)>()                                                 \
            + cublasdx::SM<ARCH>()                                                            \
            + cublasdx::Block()                                                               \
            + cublasdx::BlockDim<TC, 1, 1>());                                                \
        static constexpr uint32_t block_threads =                                             \
            static_cast<uint32_t>(GEMM::block_dim.x);                                         \
        static constexpr std::size_t smem_bytes =                                             \
            cublasdx::get_shared_storage_size<GEMM>();                                        \
        __device__ inline void run(float alpha, float* const* A, float* const* B,             \
                                   float beta,  float* const* C, char* smem)                  \
        {                                                                                     \
            _GLASS_ASSERT_BLOCKDIM_GEQ(GEMM)                                                  \
            using align = cublasdx::alignment_of<GEMM>;                                       \
            for (uint32_t b = 0; b < BATCH; ++b) {                                            \
                float* Ab = A[b];                                                             \
                float* Bb = B[b];                                                             \
                float* Cb = C[b];                                                             \
                auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);    \
                auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());       \
                auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());       \
                auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());       \
                cublasdx::copy<GEMM, align::a>(                                               \
                    cublasdx::make_tensor(Ab, GEMM::get_layout_gmem_a()), a_smem);            \
                cublasdx::copy<GEMM, align::b>(                                               \
                    cublasdx::make_tensor(Bb, GEMM::get_layout_gmem_b()), b_smem);            \
                cublasdx::copy<GEMM, align::c>(                                               \
                    cublasdx::make_tensor(Cb, GEMM::get_layout_gmem_c()), c_smem);            \
                cublasdx::copy_wait();                                                        \
                GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                          \
                __syncthreads();                                                              \
                cublasdx::copy<GEMM, align::c>(                                               \
                    c_smem, cublasdx::make_tensor(Cb, GEMM::get_layout_gmem_c()));            \
                __syncthreads();                                                              \
            }                                                                                 \
        }                                                                                     \
        __device__ inline void run_strided(float alpha, float* A, float* B,                   \
                                           float beta,  float* C, char* smem,                 \
                                           uint32_t A_STRIDE, uint32_t B_STRIDE,              \
                                           uint32_t C_STRIDE)                                 \
        {                                                                                     \
            _GLASS_ASSERT_BLOCKDIM_GEQ(GEMM)                                                  \
            using align = cublasdx::alignment_of<GEMM>;                                       \
            for (uint32_t b = 0; b < BATCH; ++b) {                                            \
                float* Ab = A + b * A_STRIDE;                                                 \
                float* Bb = B + b * B_STRIDE;                                                 \
                float* Cb = C + b * C_STRIDE;                                                 \
                auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);    \
                auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());       \
                auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());       \
                auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());       \
                cublasdx::copy<GEMM, align::a>(                                               \
                    cublasdx::make_tensor(Ab, GEMM::get_layout_gmem_a()), a_smem);            \
                cublasdx::copy<GEMM, align::b>(                                               \
                    cublasdx::make_tensor(Bb, GEMM::get_layout_gmem_b()), b_smem);            \
                cublasdx::copy<GEMM, align::c>(                                               \
                    cublasdx::make_tensor(Cb, GEMM::get_layout_gmem_c()), c_smem);            \
                cublasdx::copy_wait();                                                        \
                GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                          \
                __syncthreads();                                                              \
                cublasdx::copy<GEMM, align::c>(                                               \
                    c_smem, cublasdx::make_tensor(Cb, GEMM::get_layout_gmem_c()));            \
                __syncthreads();                                                              \
            }                                                                                 \
        }                                                                                     \
    }                                                                                         \
    template <>                                                                               \
    __device__ inline void gemm_batched_1d<float, M, N, K, BATCH, TC,                        \
                                            static_cast<layout>(LA),                          \
                                            static_cast<layout>(LB),                          \
                                            static_cast<layout>(LC), ARCH>                   \
                                           (float alpha, float* const* A, float* const* B,    \
                                            float beta,  float* const* C, char* smem)         \
    {                                                                                         \
        _nvidia_gemm_batched_1d_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::run( \
            alpha, A, B, beta, C, smem);                                                      \
    }                                                                                         \
    template <>                                                                               \
    constexpr std::size_t gemm_batched_1d_smem_size<float, M, N, K, BATCH, TC,                \
                                                     static_cast<layout>(LA),                  \
                                                     static_cast<layout>(LB),                  \
                                                     static_cast<layout>(LC), ARCH>()         \
    {                                                                                         \
        return _nvidia_gemm_batched_1d_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::smem_bytes; \
    }                                                                                         \
    template <>                                                                               \
    constexpr uint32_t gemm_batched_1d_threads<float, M, N, K, BATCH, TC,                    \
                                                static_cast<layout>(LA),                       \
                                                static_cast<layout>(LB),                       \
                                                static_cast<layout>(LC), ARCH>()              \
    {                                                                                         \
        return _nvidia_gemm_batched_1d_impl_##M##x##N##x##K##_b##BATCH##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::block_threads; \
    }

#define _GLASS_GEMM_BATCHED_1D_BD_E(M, N, K, BATCH, TC, LA, LB, LC, ARCH) \
    _GLASS_GEMM_BATCHED_1D_BD(M, N, K, BATCH, TC, LA, LB, LC, ARCH)

// Public DEFINE macros for the 1D-launch batched cuBLASDx variant. BlockDim
// is required because batched cuBLASDx needs a pinned thread count.
#define DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM(M, N, K, BATCH, TC) \
    _GLASS_GEMM_BATCHED_1D_BD_E(M, N, K, BATCH, TC, 0, 0, 0, SMS)

#define DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM_LAYOUT(M, N, K, BATCH, TC, LA, LB, LC) \
    _GLASS_GEMM_BATCHED_1D_BD_E(M, N, K, BATCH, TC, LA, LB, LC, SMS)

#define DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM_SM(M, N, K, BATCH, TC, SM) \
    _GLASS_GEMM_BATCHED_1D_BD_E(M, N, K, BATCH, TC, 0, 0, 0, SM)

#define DEFINE_NVIDIA_GEMM_BATCHED_1D_BLOCKDIM_LAYOUT_SM(M, N, K, BATCH, TC, LA, LB, LC, SM) \
    _GLASS_GEMM_BATCHED_1D_BD_E(M, N, K, BATCH, TC, LA, LB, LC, SM)
