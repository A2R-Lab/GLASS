#pragma once
#include <cstdint>
#include <cublasdx.hpp>

// glass::nvidia L2 — cuBLASDx-backed gemv
// All sizes are compile-time; call the DEFINE_NVIDIA_GEMV(M, N) macro once per
// (M, N) pair you need, then use glass::nvidia::gemv<float, M, N>(...).
//
// The caller is responsible for providing an smem pointer with at least
// glass::nvidia::gemv_smem_size<T, M, N>() bytes.
//
// Example:
//   constexpr auto smem = glass::nvidia::gemv_smem_size<float, 6, 6>();
//   kernel<<<1, glass::nvidia::gemv_threads<float, 6, 6>(), smem>>>(A, x, y);
//   // inside kernel:
//   glass::nvidia::gemv<float, 6, 6>(1.f, A, x, 0.f, y, smem_ptr);

#ifndef SMS
#define SMS 860
#endif

// GEMV via cuBLASDx: compute y = alpha * A * x + beta * y
// cuBLASDx has no native GEMV; we implement it as GEMM with N=1.
// The macro creates a specialization in the glass::nvidia namespace.
#define DEFINE_NVIDIA_GEMV(M, N)                                                        \
    namespace _nvidia_gemv_impl_##M##x##N {                                             \
        using GEMM = decltype(                                                           \
            cublasdx::Size<M, 1, N>()                                                   \
            + cublasdx::Precision<float>()                                               \
            + cublasdx::Type<cublasdx::type::real>()                                     \
            + cublasdx::Function<cublasdx::function::MM>()                               \
            + cublasdx::SM<SMS>()                                                         \
            + cublasdx::Block());                                                         \
        static constexpr uint32_t block_threads =                                        \
            static_cast<uint32_t>(GEMM::block_dim.x *                                   \
                                  GEMM::block_dim.y *                                   \
                                  GEMM::block_dim.z);                                   \
        static constexpr std::size_t smem_bytes =                                        \
            cublasdx::get_shared_storage_size<GEMM>();                                   \
        __device__ inline void run(float alpha, float* A, float* x,                     \
                                   float beta,  float* y, char* smem)                   \
        {                                                                                \
            using align = cublasdx::alignment_of<GEMM>;                                 \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem); \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());    \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());    \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());    \
            cublasdx::copy<GEMM, align::a>(                                             \
                cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);          \
            cublasdx::copy<GEMM, align::b>(                                             \
                cublasdx::make_tensor(x, GEMM::get_layout_gmem_b()), b_smem);          \
            cublasdx::copy<GEMM, align::c>(                                             \
                cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()), c_smem);          \
            cublasdx::copy_wait();                                                      \
            GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                       \
            __syncthreads();                                                             \
            cublasdx::copy<GEMM, align::c>(                                             \
                c_smem, cublasdx::make_tensor(y, GEMM::get_layout_gmem_c()));          \
            __syncthreads();                                                             \
        }                                                                                \
    }                                                                                    \
    template <>                                                                          \
    __device__ inline void gemv<float, M, N>(float alpha, float* A, float* x,          \
                                              float beta,  float* y, char* smem)        \
    {                                                                                    \
        _nvidia_gemv_impl_##M##x##N::run(alpha, A, x, beta, y, smem);                  \
    }                                                                                    \
    template <>                                                                          \
    constexpr std::size_t gemv_smem_size<float, M, N>()                                 \
    {                                                                                    \
        return _nvidia_gemv_impl_##M##x##N::smem_bytes;                                 \
    }                                                                                    \
    template <>                                                                          \
    constexpr uint32_t gemv_threads<float, M, N>()                                      \
    {                                                                                    \
        return _nvidia_gemv_impl_##M##x##N::block_threads;                              \
    }

// Primary templates — instantiated by DEFINE_NVIDIA_GEMV
template <typename T, uint32_t M, uint32_t N>
__device__ void gemv(T alpha, T* A, T* x, T beta, T* y, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::gemv<T,M,N> is not available for this (T,M,N). "
        "Add DEFINE_NVIDIA_GEMV(M,N) in your .cu file or use a pre-defined size.");
}

template <typename T, uint32_t M, uint32_t N>
constexpr std::size_t gemv_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N>
constexpr uint32_t gemv_threads() { return 256; }

// row_strided_gemv: packs strided A into compact shared scratch, then delegates
// to the standard nvidia::gemv<T,M,N> (which requires DEFINE_NVIDIA_GEMV(M,N)).
//
// smem layout: [A_compact: M*N*sizeof(T)] [cuBLASDx smem for gemv<T,M,N>]
// Total bytes: row_strided_gemv_smem_size<T,M,N>()
//
// ROW_STRIDE = leading dimension of column-major A: A[i][j] = A[i + j*ROW_STRIDE].
// When ROW_STRIDE == M this degenerates to standard gemv with no overhead.
template <typename T, uint32_t M, uint32_t N, uint32_t ROW_STRIDE = M>
__device__ void row_strided_gemv(T alpha, T* A, T* x, T beta, T* y, char* smem)
{
    uint32_t rank = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    uint32_t size = blockDim.x * blockDim.y * blockDim.z;
    T* A_compact = reinterpret_cast<T*>(smem);
    char* cublas_smem = smem + M * N * sizeof(T);
    for (uint32_t i = rank; i < M * N; i += size) {
        uint32_t r = i % M, c = i / M;
        A_compact[r + c*M] = A[r + c*ROW_STRIDE];
    }
    __syncthreads();
    gemv<T, M, N>(alpha, A_compact, x, beta, y, cublas_smem);
}

template <typename T, uint32_t M, uint32_t N>
constexpr std::size_t row_strided_gemv_smem_size()
{
    return M * N * sizeof(T) + gemv_smem_size<T, M, N>();
}
