#pragma once
#include <cstdint>
#include <cublasdx.hpp>
// types.cuh provides the `layout` enum and the shared
// _GLASS_CUBLAS_LAYOUT / _GLASS_ASSERT_BLOCKDIM_GEQ helper macros.
#include "./types.cuh"

// glass::nvidia L2 — cuBLASDx-backed gemv (implemented as GEMM with N=1)
//
// All sizes are compile-time. Call one of the DEFINE_NVIDIA_GEMV* macros once
// per (M, N, BLOCK_THREADS, layouts, SM) combination you need, then call
// glass::nvidia::gemv<...>(alpha, A, x, beta, y, smem) inside your kernel.
//
// Backward-compatible defaults:
//   BLOCK_THREADS = 0           -> let cuBLASDx pick block_dim from its database
//   LA = LB = LC = col_major    -> standard column-major
//   SM_VAL = SMS                -> SMS macro (default 860)
//
// Example (basic):
//   DEFINE_NVIDIA_GEMV(6, 6)
//   constexpr auto smem    = glass::nvidia::gemv_smem_size<float, 6, 6>();
//   constexpr auto threads = glass::nvidia::gemv_threads<float, 6, 6>();
//   kernel<<<1, threads, smem>>>(...);
//   glass::nvidia::gemv<float, 6, 6>(1.f, A, x, 0.f, y, smem_ptr);
//
// Example (caller-controlled BlockDim):
//   DEFINE_NVIDIA_GEMV_BLOCKDIM(6, 6, 352)
//   kernel<<<1, 352, smem>>>(...);
//   glass::nvidia::gemv<float, 6, 6, 352>(1.f, A, x, 0.f, y, smem_ptr);

#ifndef SMS
#define SMS 860
#endif

// ---------------------------------------------------------------------------
// Primary templates — instantiated by the DEFINE_NVIDIA_GEMV* macros below.
//
// AUTO-DISPATCH: like glass::nvidia::gemm<>, the gemv<> primary template now
// consults should_use_cublasdx_gemv<T, M, N, SM_VAL>():
//   - returns false → routes to ::glass::gemv<T, M, N, false, ROW_MAJOR>
//                     (SIMT, no scratch). Layout LA maps to SIMT's ROW_MAJOR
//                     flag; LB/LC are degenerate for vectors and ignored.
//   - returns true  → static_assert that a DEFINE_NVIDIA_GEMV* macro was used.
// ---------------------------------------------------------------------------

/**
 * @brief Block-level GEMV that auto-dispatches between SIMT and cuBLASDx.
 *
 * Computes `y = alpha*A*x + beta*y` for an M×N matrix A. The primary template
 * consults `should_use_cublasdx_gemv<T,M,N,SM_VAL>()` at compile time: small
 * shapes fall through to the dependency-free SIMT path `%glass::gemv` (no
 * scratch, LA maps to its ROW_MAJOR flag); shapes the heuristic flags for the
 * vendor backend require a `DEFINE_NVIDIA_GEMV*` macro in scope (cuBLASDx,
 * needs the MathDx headers / MATHDX_ROOT) or a static_assert fires.
 * Compile-time sizes only. NumPy equivalent: `y = alpha*A@x + beta*y`.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A / length of y.
 * @tparam N             Columns of A / length of x.
 * @tparam BLOCK_THREADS Pinned cuBLASDx BlockDim (0 = vendor picks).
 * @tparam LA            Memory layout of A.
 * @tparam LB            Memory layout of B (degenerate for a vector; ignored).
 * @tparam LC            Memory layout of C (degenerate for a vector; ignored).
 * @tparam SM_VAL        Target SM architecture (default = SMS).
 * @tparam TRAILING_SYNC Emit a trailing __syncthreads() before return (default true).
 * @param  alpha         Scaling factor for A*x.
 * @param  A             Pointer to the M×N matrix.
 * @param  x             Pointer to the input vector (length N).
 * @param  beta          Scaling factor for the incoming y.
 * @param  y             Pointer to the output vector (length M).
 * @param  smem          Shared scratch (cuBLASDx route only; unused for SIMT).
 */
template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS,
          bool TRAILING_SYNC = true>
__device__ void gemv(T alpha, T* A, T* x, T beta, T* y, char* smem)
{
    if constexpr (!should_use_cublasdx_gemv<T, M, N, SM_VAL>()) {
        constexpr bool ROW_MAJOR = (LA == layout::row_major);
        ::glass::gemv<T, M, N, /*TRANSPOSE=*/false, ROW_MAJOR>(
            alpha, A, x, beta, y);
    } else {
        static_assert(sizeof(T) == 0,
            "glass::nvidia::gemv<T,M,N,BLOCK_THREADS,LA,LB,LC,SM_VAL>: "
            "should_use_cublasdx_gemv<> returned true for this shape but no "
            "DEFINE_NVIDIA_GEMV* macro is in scope. Add one in your .cu, or "
            "override the dispatch via tuning_table.cuh / GLASS_TUNING_TABLE_LOCAL.");
    }
}

/**
 * @brief Shared-memory bytes needed by `gemv<...>` (host-callable).
 *
 * Returns the cuBLASDx scratch size for this signature, or 0 when the
 * auto-dispatch routes to SIMT (which needs no scratch). Template parameters
 * match gemv<>. constexpr.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A.
 * @tparam N             Columns of A.
 * @tparam BLOCK_THREADS Pinned cuBLASDx BlockDim (0 = vendor picks).
 * @tparam LA            Memory layout of A.
 * @tparam LB            Memory layout of B.
 * @tparam LC            Memory layout of C.
 * @tparam SM_VAL        Target SM architecture.
 */
template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr std::size_t gemv_smem_size() { return 0; }

/**
 * @brief Thread count cuBLASDx wants for `gemv<...>` (host-callable).
 *
 * Returns the block thread count to launch with for the cuBLASDx route (the
 * default 256 is overridden by the matching `DEFINE_NVIDIA_GEMV*`
 * specialization). Template parameters match gemv<>. constexpr.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A.
 * @tparam N             Columns of A.
 * @tparam BLOCK_THREADS Pinned cuBLASDx BlockDim (0 = vendor picks).
 * @tparam LA            Memory layout of A.
 * @tparam LB            Memory layout of B.
 * @tparam LC            Memory layout of C.
 * @tparam SM_VAL        Target SM architecture.
 */
template <typename T, uint32_t M, uint32_t N,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr uint32_t gemv_threads() { return 256; }

// ---------------------------------------------------------------------------
// Private core macros — gemv = GEMM with N=1.  cuBLASDx Size<M,1,N> maps to
// A(M×N) * x(N×1) = y(M×1).
// LA, LB, LC are integer literals 0 (col_major) or 1 (row_major).
// SM is an integer literal SM architecture (e.g. 860).
// ---------------------------------------------------------------------------

// ARCH (not SM) avoids token collision with cublasdx::SM<>.
#define _GLASS_GEMV_NO_BD(M, N, LA, LB, LC, ARCH)                                               \
    namespace _nvidia_gemv_impl_##M##x##N##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH {           \
        using GEMM = decltype(                                                                  \
            cublasdx::Size<M, 1, N>()                                                           \
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
        template <bool TRAILING_SYNC>                                                           \
        __device__ inline void run(float alpha, float* A, float* x,                             \
                                   float beta,  float* y, char* smem)                           \
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
                cublasdx::make_tensor(x, GEMM::get_layout_gmem_b()), b_smem);                  \
            cublasdx::copy<GEMM, align::c>(                                                     \
                cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()), c_smem);                  \
            cublasdx::copy_wait();                                                              \
            GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                               \
            __syncthreads();                                                                    \
            cublasdx::copy<GEMM, align::c>(                                                     \
                c_smem, cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()));                  \
            if constexpr (TRAILING_SYNC) {                                                      \
                __syncthreads();                                                                \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    template <>                                                                                 \
    __device__ inline void gemv<float, M, N, 0,                                                 \
                                static_cast<layout>(LA),                                        \
                                static_cast<layout>(LB),                                        \
                                static_cast<layout>(LC), ARCH, true>                            \
                               (float alpha, float* A, float* x,                                \
                                float beta,  float* y, char* smem)                              \
    {                                                                                           \
        _nvidia_gemv_impl_##M##x##N##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH                   \
            ::template run<true>(alpha, A, x, beta, y, smem);                                   \
    }                                                                                           \
    template <>                                                                                 \
    __device__ inline void gemv<float, M, N, 0,                                                 \
                                static_cast<layout>(LA),                                        \
                                static_cast<layout>(LB),                                        \
                                static_cast<layout>(LC), ARCH, false>                           \
                               (float alpha, float* A, float* x,                                \
                                float beta,  float* y, char* smem)                              \
    {                                                                                           \
        _nvidia_gemv_impl_##M##x##N##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH                   \
            ::template run<false>(alpha, A, x, beta, y, smem);                                  \
    }                                                                                           \
    template <>                                                                                 \
    constexpr std::size_t gemv_smem_size<float, M, N, 0,                                        \
                                          static_cast<layout>(LA),                              \
                                          static_cast<layout>(LB),                              \
                                          static_cast<layout>(LC), ARCH>()                      \
    {                                                                                           \
        return _nvidia_gemv_impl_##M##x##N##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH::smem_bytes; \
    }                                                                                           \
    template <>                                                                                 \
    constexpr uint32_t gemv_threads<float, M, N, 0,                                             \
                                     static_cast<layout>(LA),                                   \
                                     static_cast<layout>(LB),                                   \
                                     static_cast<layout>(LC), ARCH>()                           \
    {                                                                                           \
        return _nvidia_gemv_impl_##M##x##N##_bd0_la##LA##_lb##LB##_lc##LC##_sm##ARCH::block_threads; \
    }

#define _GLASS_GEMV_BD(M, N, TC, LA, LB, LC, ARCH)                                              \
    namespace _nvidia_gemv_impl_##M##x##N##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH {      \
        using GEMM = decltype(                                                                  \
            cublasdx::Size<M, 1, N>()                                                           \
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
        template <bool TRAILING_SYNC>                                                           \
        __device__ inline void run(float alpha, float* A, float* x,                             \
                                   float beta,  float* y, char* smem)                           \
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
                cublasdx::make_tensor(x, GEMM::get_layout_gmem_b()), b_smem);                  \
            cublasdx::copy<GEMM, align::c>(                                                     \
                cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()), c_smem);                  \
            cublasdx::copy_wait();                                                              \
            GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                               \
            __syncthreads();                                                                    \
            cublasdx::copy<GEMM, align::c>(                                                     \
                c_smem, cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()));                  \
            if constexpr (TRAILING_SYNC) {                                                      \
                __syncthreads();                                                                \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    template <>                                                                                 \
    __device__ inline void gemv<float, M, N, TC,                                                \
                                static_cast<layout>(LA),                                        \
                                static_cast<layout>(LB),                                        \
                                static_cast<layout>(LC), ARCH, true>                            \
                               (float alpha, float* A, float* x,                                \
                                float beta,  float* y, char* smem)                              \
    {                                                                                           \
        _nvidia_gemv_impl_##M##x##N##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH              \
            ::template run<true>(alpha, A, x, beta, y, smem);                                   \
    }                                                                                           \
    template <>                                                                                 \
    __device__ inline void gemv<float, M, N, TC,                                                \
                                static_cast<layout>(LA),                                        \
                                static_cast<layout>(LB),                                        \
                                static_cast<layout>(LC), ARCH, false>                           \
                               (float alpha, float* A, float* x,                                \
                                float beta,  float* y, char* smem)                              \
    {                                                                                           \
        _nvidia_gemv_impl_##M##x##N##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH              \
            ::template run<false>(alpha, A, x, beta, y, smem);                                  \
    }                                                                                           \
    template <>                                                                                 \
    constexpr std::size_t gemv_smem_size<float, M, N, TC,                                       \
                                          static_cast<layout>(LA),                              \
                                          static_cast<layout>(LB),                              \
                                          static_cast<layout>(LC), ARCH>()                      \
    {                                                                                           \
        return _nvidia_gemv_impl_##M##x##N##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::smem_bytes; \
    }                                                                                           \
    template <>                                                                                 \
    constexpr uint32_t gemv_threads<float, M, N, TC,                                            \
                                     static_cast<layout>(LA),                                   \
                                     static_cast<layout>(LB),                                   \
                                     static_cast<layout>(LC), ARCH>()                           \
    {                                                                                           \
        return _nvidia_gemv_impl_##M##x##N##_bd##TC##_la##LA##_lb##LB##_lc##LC##_sm##ARCH::block_threads; \
    }

// One-line indirection wrappers — force SMS (or any macro passed as ARCH) to
// be expanded at the call boundary, BEFORE token pasting freezes it as text.
#define _GLASS_GEMV_NO_BD_E(M, N, LA, LB, LC, ARCH)        _GLASS_GEMV_NO_BD(M, N, LA, LB, LC, ARCH)
#define _GLASS_GEMV_BD_E(M, N, TC, LA, LB, LC, ARCH)       _GLASS_GEMV_BD(M, N, TC, LA, LB, LC, ARCH)

// ---------------------------------------------------------------------------
// Public DEFINE_NVIDIA_GEMV* convenience macros (route through _E indirection)
// ---------------------------------------------------------------------------

#define DEFINE_NVIDIA_GEMV(M, N) \
    _GLASS_GEMV_NO_BD_E(M, N, 0, 0, 0, SMS)

#define DEFINE_NVIDIA_GEMV_BLOCKDIM(M, N, TC) \
    _GLASS_GEMV_BD_E(M, N, TC, 0, 0, 0, SMS)

#define DEFINE_NVIDIA_GEMV_LAYOUT(M, N, LA, LB, LC) \
    _GLASS_GEMV_NO_BD_E(M, N, LA, LB, LC, SMS)

#define DEFINE_NVIDIA_GEMV_BLOCKDIM_LAYOUT(M, N, TC, LA, LB, LC) \
    _GLASS_GEMV_BD_E(M, N, TC, LA, LB, LC, SMS)

#define DEFINE_NVIDIA_GEMV_SM(M, N, SM) \
    _GLASS_GEMV_NO_BD_E(M, N, 0, 0, 0, SM)

#define DEFINE_NVIDIA_GEMV_BLOCKDIM_SM(M, N, TC, SM) \
    _GLASS_GEMV_BD_E(M, N, TC, 0, 0, 0, SM)

#define DEFINE_NVIDIA_GEMV_LAYOUT_SM(M, N, LA, LB, LC, SM) \
    _GLASS_GEMV_NO_BD_E(M, N, LA, LB, LC, SM)

#define DEFINE_NVIDIA_GEMV_BLOCKDIM_LAYOUT_SM(M, N, TC, LA, LB, LC, SM) \
    _GLASS_GEMV_BD_E(M, N, TC, LA, LB, LC, SM)

// ---------------------------------------------------------------------------
// row_strided_gemv: packs strided A into compact shared scratch, then delegates
// to the standard nvidia::gemv<...>. Forwards all template parameters
// (BLOCK_THREADS, layouts, SM) to the inner call.
//
// smem layout: [A_compact: M*N*sizeof(T)] [cuBLASDx smem for gemv<...>]
// ROW_STRIDE = leading dimension of column-major A: A[i][j] = A[i + j*ROW_STRIDE].
// When ROW_STRIDE == M this degenerates to standard gemv with no overhead.
//
// 1D-launch compatible (P1-5): the rank/size flattening below makes this
// kernel work with any launch geometry — 1D <<<grid, dim3(N,1,1)>>>, 2D, or
// 3D — as long as the total thread count is the BLOCK_THREADS value passed
// to the inner cuBLASDx gemv. No 2D-only batched-style assumption.
// ---------------------------------------------------------------------------
/**
 * @brief Strided-A GEMV that auto-dispatches between SIMT and cuBLASDx.
 *
 * Computes `y = alpha*A*x + beta*y` where A is stored with an arbitrary
 * leading dimension ROW_STRIDE (A[i + j*ROW_STRIDE]). On the SIMT route the
 * stride is used directly (no packing, no scratch); on the cuBLASDx route the
 * strided A is packed into compact shared scratch and forwarded to gemv<>.
 * When ROW_STRIDE == M this degenerates to standard gemv. Works under any
 * launch geometry (1D/2D/3D) as long as the total thread count matches.
 * cuBLASDx route needs the MathDx headers / MATHDX_ROOT.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A / length of y.
 * @tparam N             Columns of A / length of x.
 * @tparam ROW_STRIDE    Leading dimension (row stride) of A.
 * @tparam BLOCK_THREADS Pinned cuBLASDx BlockDim (0 = vendor picks).
 * @tparam LA            Memory layout of A.
 * @tparam LB            Memory layout of B.
 * @tparam LC            Memory layout of C.
 * @tparam SM_VAL        Target SM architecture.
 * @tparam TRAILING_SYNC Emit a trailing __syncthreads() before return (default true).
 * @param  alpha         Scaling factor for A*x.
 * @param  A             Pointer to the strided M×N matrix.
 * @param  x             Pointer to the input vector (length N).
 * @param  beta          Scaling factor for the incoming y.
 * @param  y             Pointer to the output vector (length M).
 * @param  smem          Shared scratch (cuBLASDx route only; unused for SIMT).
 */
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS,
          bool TRAILING_SYNC = true>
__device__ void row_strided_gemv(T alpha, T* A, T* x, T beta, T* y, char* smem)
{
    if constexpr (!should_use_cublasdx_row_strided_gemv<T, M, N, ROW_STRIDE, SM_VAL>()) {
        // SIMT path: ::glass::row_strided_gemv uses the stride directly,
        // no packing required. Arg order shuffles to match SIMT convention.
        ::glass::row_strided_gemv<T, M, N, ROW_STRIDE>(A, x, y, alpha, beta);
    } else {
        // cuBLASDx path: pack strided A into compact scratch, then delegate
        // to the cuBLASDx-specialized gemv<>.
        uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        T* A_compact = reinterpret_cast<T*>(smem);
        char* cublas_smem = smem + M * N * sizeof(T);
        for (uint32_t i = rank; i < M * N; i += size) {
            uint32_t r = i % M, c = i / M;
            A_compact[r + c*M] = A[r + c*ROW_STRIDE];
        }
        __syncthreads();
        gemv<T, M, N, BLOCK_THREADS, LA, LB, LC, SM_VAL, TRAILING_SYNC>(
            alpha, A_compact, x, beta, y, cublas_smem);
    }
}

/**
 * @brief Shared-memory bytes needed by `row_strided_gemv<...>` (host-callable).
 *
 * Returns 0 when the auto-dispatch routes to SIMT (no packing), or the
 * A-packing scratch plus the inner cuBLASDx gemv scratch otherwise. constexpr.
 *
 * @tparam T             Scalar type.
 * @tparam M             Rows of A.
 * @tparam N             Columns of A.
 * @tparam ROW_STRIDE    Leading dimension (row stride) of A.
 * @tparam BLOCK_THREADS Pinned cuBLASDx BlockDim (0 = vendor picks).
 * @tparam LA            Memory layout of A.
 * @tparam LB            Memory layout of B.
 * @tparam LC            Memory layout of C.
 * @tparam SM_VAL        Target SM architecture.
 */
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M,
          uint32_t BLOCK_THREADS = 0,
          layout LA = layout::col_major,
          layout LB = layout::col_major,
          layout LC = layout::col_major,
          uint32_t SM_VAL = SMS>
constexpr std::size_t row_strided_gemv_smem_size()
{
    if constexpr (!should_use_cublasdx_row_strided_gemv<T, M, N, ROW_STRIDE, SM_VAL>())
        return 0;
    else
        return M * N * sizeof(T)
             + gemv_smem_size<T, M, N, BLOCK_THREADS, LA, LB, LC, SM_VAL>();
}
