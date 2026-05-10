#pragma once
#include <cstdint>
#include <cublasdx.hpp>

// glass::nvidia L3 — cuBLASDx-backed gemm
// All sizes are compile-time; call the DEFINE_NVIDIA_GEMM(M, N, K) macro once per
// (M, N, K) triple you need, then use glass::nvidia::gemm<float, M, N, K>(...).
//
// The caller is responsible for providing an smem pointer with at least
// glass::nvidia::gemm_smem_size<T, M, N, K>() bytes.
//
// Example:
//   constexpr auto smem = glass::nvidia::gemm_smem_size<float, 6, 6, 6>();
//   constexpr auto threads = glass::nvidia::gemm_threads<float, 6, 6, 6>();
//   kernel<<<1, threads, smem>>>(A, B, C);
//   // inside kernel:
//   glass::nvidia::gemm<float, 6, 6, 6>(1.f, A, B, 0.f, C, smem_ptr);

#ifndef SMS
#define SMS 860
#endif

// GEMM via cuBLASDx: C = alpha * A * B + beta * C
// cuBLASDx Size<M, K, N> → computes (M×K) * (K×N) = (M×N); our GEMM is (M×N) * (N×K) = (M×K)
// so we pass Size<M, N, K> which maps to A(M×N) * B(N×K) = C(M×K).
#define DEFINE_NVIDIA_GEMM(M, N, K)                                                     \
    namespace _nvidia_gemm_impl_##M##x##N##x##K {                                       \
        using GEMM = decltype(                                                           \
            cublasdx::Size<M, N, K>()                                                   \
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
        __device__ inline void run(float alpha, float* A, float* B,                     \
                                   float beta,  float* C, char* smem)                   \
        {                                                                                \
            using align = cublasdx::alignment_of<GEMM>;                                 \
            auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem); \
            auto a_smem = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());    \
            auto b_smem = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());    \
            auto c_smem = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());    \
            cublasdx::copy<GEMM, align::a>(                                             \
                cublasdx::make_tensor(A, GEMM::get_layout_gmem_a()), a_smem);          \
            cublasdx::copy<GEMM, align::b>(                                             \
                cublasdx::make_tensor(B, GEMM::get_layout_gmem_b()), b_smem);          \
            cublasdx::copy<GEMM, align::c>(                                             \
                cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()), c_smem);          \
            cublasdx::copy_wait();                                                      \
            GEMM().execute(alpha, a_smem, b_smem, beta, c_smem);                       \
            __syncthreads();                                                             \
            cublasdx::copy<GEMM, align::c>(                                             \
                c_smem, cublasdx::make_tensor(C, GEMM::get_layout_gmem_c()));          \
            __syncthreads();                                                             \
        }                                                                                \
    }                                                                                    \
    template <>                                                                          \
    __device__ inline void gemm<float, M, N, K>(float alpha, float* A, float* B,       \
                                                  float beta,  float* C, char* smem)    \
    {                                                                                    \
        _nvidia_gemm_impl_##M##x##N##x##K::run(alpha, A, B, beta, C, smem);            \
    }                                                                                    \
    template <>                                                                          \
    constexpr std::size_t gemm_smem_size<float, M, N, K>()                              \
    {                                                                                    \
        return _nvidia_gemm_impl_##M##x##N##x##K::smem_bytes;                           \
    }                                                                                    \
    template <>                                                                          \
    constexpr uint32_t gemm_threads<float, M, N, K>()                                   \
    {                                                                                    \
        return _nvidia_gemm_impl_##M##x##N##x##K::block_threads;                        \
    }

// Primary templates — instantiated by DEFINE_NVIDIA_GEMM
template <typename T, uint32_t M, uint32_t N, uint32_t K>
__device__ void gemm(T alpha, T* A, T* B, T beta, T* C, char* smem)
{
    static_assert(sizeof(T) == 0,
        "glass::nvidia::gemm<T,M,N,K> is not available for this (T,M,N,K). "
        "Add DEFINE_NVIDIA_GEMM(M,N,K) in your .cu file or use a pre-defined size.");
}

template <typename T, uint32_t M, uint32_t N, uint32_t K>
constexpr std::size_t gemm_smem_size() { return 0; }

template <typename T, uint32_t M, uint32_t N, uint32_t K>
constexpr uint32_t gemm_threads() { return 256; }

// row_strided_gemm: packs strided A and B into compact shared scratch, then
// delegates to the standard nvidia::gemm<T,M,N,K> (requires DEFINE_NVIDIA_GEMM(M,N,K)).
//
// smem layout: [A_compact: M*N*sizeof(T)] [B_compact: N*K*sizeof(T)] [cuBLASDx smem]
// Total bytes: row_strided_gemm_smem_size<T,M,N,K>()
//
// A_RS = leading dimension of column-major A: A[i][j] = A[i + j*A_RS].
// B_RS = leading dimension of column-major B: B[j][l] = B[j + l*B_RS].
// C is written as standard column-major with LDC=M (no strided output support).
// When A_RS==M and B_RS==N this degenerates to standard gemm with one pack pass.
template <typename T, uint32_t M, uint32_t N, uint32_t K,
          uint32_t A_RS = M, uint32_t B_RS = N>
__device__ void row_strided_gemm(T alpha, T* A, T* B, T beta, T* C, char* smem)
{
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
    gemm<T, M, N, K>(alpha, A_compact, B_compact, beta, C, cublas_smem);
}

template <typename T, uint32_t M, uint32_t N, uint32_t K>
constexpr std::size_t row_strided_gemm_smem_size()
{
    return (M * N + N * K) * sizeof(T) + gemm_smem_size<T, M, N, K>();
}
