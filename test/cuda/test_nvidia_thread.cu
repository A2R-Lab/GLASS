// Correctness runner for the cuSOLVERDx 0.4+ per-thread LAPACK surface.
// Every CUDA thread owns one packed column-major problem; pytest reconstructs
// the deterministic inputs and checks the emitted factors/solutions in NumPy.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "glass-nvidia.cuh"
#include "glass-defaults.cuh"

static_assert(GLASS_HAVE_CUSOLVERDX_THREAD == 1,
              "test_nvidia_thread requires cuSOLVERDx thread execution");
static_assert(glass::defaults::have_nv_thread,
              "backend picker must see the included cuSOLVERDx thread surface");
static_assert(glass::defaults::nv_thread_available(glass::op::chol) &&
              !glass::defaults::nv_thread_available(glass::op::gemm),
              "NVIDIA thread availability is limited to the LAPACK ladder ops");
static_assert(glass::suggested_backend<glass::op::chol, 8, float, 1200>() ==
              glass::backend::nvidia_thread,
              "sm_120 measured picker reaches NVIDIA thread");
static_assert(glass::suggested_backend<glass::op::trsv, 16, float, 870>() ==
              glass::backend::nvidia_thread,
              "sm_87 measured picker reaches NVIDIA thread");

namespace gnt = glass::nvidia::thread;
constexpr int BATCHES = 7;
constexpr int THREADS = 32;

template <typename T, int N>
__global__ void k_potrf(T* A)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::potrf<T, N>(A + b * N * N);
}

template <typename T, int N>
__global__ void k_trsm(const T* L, T* B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::trsm<T, N, 2>(T(0.7), L + b * N * N,
                                        B + b * N * 2);
}

template <typename T, int N>
__global__ void k_posv(T* A, T* B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::posv<T, N, 2>(A + b * N * N,
                                        B + b * N * 2);
}

template <typename T, int N>
__global__ void k_potrs(const T* L, T* B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::potrs<T, N, 2>(L + b * N * N,
                                         B + b * N * 2);
}

template <typename T, int N>
__global__ void k_getrf(T* A)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::getrf_no_pivot<T, N>(A + b * N * N);
}

template <typename T, int N>
__global__ void k_getrs(T* A, T* B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) {
        T* Ab = A + b * N * N;
        gnt::getrf_no_pivot<T, N>(Ab);
        gnt::getrs_no_pivot<T, N, 2>(Ab, B + b * N * 2);
    }
}

template <typename T, int N>
__global__ void k_gesv(T* A, T* B)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::gesv_no_pivot<T, N, 2>(A + b * N * N,
                                                  B + b * N * 2);
}

template <typename T, int N>
__global__ void k_geqrf(T* A, T* tau)
{
    constexpr int M = N + 2;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::geqrf<T, M, N>(A + b * M * N, tau + b * N);
}

template <typename T, int N>
__global__ void k_gels(T* A, T* tau, T* B)
{
    constexpr int M = N + 2;
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < BATCHES) gnt::gels<T, M, N, 2>(A + b * M * N, tau + b * N,
                                           B + b * M * 2);
}

template <typename T>
void fill_spd(int n, std::vector<T>& A)
{
    for (int b = 0; b < BATCHES; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                T sum = T(0);
                for (int k = 0; k < n; ++k) {
                    const T ri = T(0.03) * T(1 + ((i + 2 * k + b) % 5));
                    const T rj = T(0.03) * T(1 + ((j + 2 * k + b) % 5));
                    sum += ri * rj;
                }
                if (i == j) sum += T(n) + T(0.2) * T(b);
                A[b * n * n + i + j * n] = sum;
            }
        }
    }
}

template <typename T>
void fill_general(int n, std::vector<T>& A)
{
    for (int b = 0; b < BATCHES; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                T v = T(0.02) * T((i + 2 * j + 3 * b) % 7);
                if (i == j) v += T(n) + T(0.25) * T(b);
                A[b * n * n + i + j * n] = v;
            }
        }
    }
}

template <typename T>
void fill_lower(int n, std::vector<T>& L)
{
    for (int b = 0; b < BATCHES; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                T v = T(0);
                if (i == j) v = T(1.5) + T(0.1) * T(i + b);
                else if (i > j) v = T(0.03) * T(1 + ((i + j + b) % 5));
                L[b * n * n + i + j * n] = v;
            }
        }
    }
}

template <typename T>
void fill_rhs(int rows, std::vector<T>& B)
{
    for (int b = 0; b < BATCHES; ++b)
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < rows; ++i)
                B[b * rows * 2 + i + j * rows] =
                    T(0.2) + T(0.04) * T((2 * i + 3 * j + b) % 6);
}

template <typename T>
void fill_rect(int m, int n, std::vector<T>& A)
{
    for (int b = 0; b < BATCHES; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                T v = T(0.04) * T(1 + ((i + 3 * j + b) % 7));
                if (i == j) v += T(1.5) + T(0.05) * T(b);
                A[b * m * n + i + j * m] = v;
            }
        }
    }
}

template <typename T, int N, bool FULL_SURFACE>
int run(const char* op)
{
    constexpr int M = N + 2;
    std::vector<T> hA(BATCHES * M * M, T(0));
    std::vector<T> hB(BATCHES * M * 2, T(0));
    std::vector<T> htau(BATCHES * N, T(0));

    bool rectangular = !std::strcmp(op, "geqrf") || !std::strcmp(op, "gels");
    if (rectangular) fill_rect(M, N, hA);
    else if (!std::strcmp(op, "potrf") || !std::strcmp(op, "posv")) fill_spd(N, hA);
    else if (!std::strcmp(op, "trsm") || !std::strcmp(op, "potrs")) fill_lower(N, hA);
    else fill_general(N, hA);
    fill_rhs(rectangular ? M : N, hB);

    T *dA = nullptr, *dB = nullptr, *dtau = nullptr;
    cudaError_t err = cudaMalloc(&dA, hA.size() * sizeof(T));
    if (err == cudaSuccess) err = cudaMalloc(&dB, hB.size() * sizeof(T));
    if (err == cudaSuccess) err = cudaMalloc(&dtau, htau.size() * sizeof(T));
    if (err == cudaSuccess) err = cudaMemcpy(dA, hA.data(), hA.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dB, hB.data(), hB.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dtau, htau.data(), htau.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA setup failed: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dtau);
        return 1;
    }

    bool launched = true;
    if (!std::strcmp(op, "potrf")) k_potrf<T, N><<<1, THREADS>>>(dA);
    else if (!std::strcmp(op, "trsm")) k_trsm<T, N><<<1, THREADS>>>(dA, dB);
    else if (!std::strcmp(op, "posv")) k_posv<T, N><<<1, THREADS>>>(dA, dB);
    else if constexpr (FULL_SURFACE) {
        if (!std::strcmp(op, "potrs")) k_potrs<T, N><<<1, THREADS>>>(dA, dB);
        else if (!std::strcmp(op, "getrf")) k_getrf<T, N><<<1, THREADS>>>(dA);
        else if (!std::strcmp(op, "getrs")) k_getrs<T, N><<<1, THREADS>>>(dA, dB);
        else if (!std::strcmp(op, "gesv")) k_gesv<T, N><<<1, THREADS>>>(dA, dB);
        else if (!std::strcmp(op, "geqrf")) k_geqrf<T, N><<<1, THREADS>>>(dA, dtau);
        else if (!std::strcmp(op, "gels")) k_gels<T, N><<<1, THREADS>>>(dA, dtau, dB);
        else launched = false;
    } else {
        launched = false;
    }
    if (!launched) {
        std::fprintf(stderr, "unknown operation: %s\n", op);
        cudaFree(dA); cudaFree(dB); cudaFree(dtau);
        return 2;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA operation failed: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dtau);
        return 1;
    }

    const bool emit_a = !std::strcmp(op, "potrf") || !std::strcmp(op, "getrf") ||
                        !std::strcmp(op, "geqrf");
    const int count = emit_a ? BATCHES * (rectangular ? M * N : N * N)
                             : BATCHES * (rectangular ? M * 2 : N * 2);
    std::vector<T>& out = emit_a ? hA : hB;
    err = cudaMemcpy(out.data(), emit_a ? dA : dB, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(dA); cudaFree(dB); cudaFree(dtau);
        return 1;
    }
    for (int i = 0; i < count; ++i) std::printf("%.17g ", static_cast<double>(out[i]));
    std::printf("\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dtau);
    return 0;
}

template <typename T>
int dispatch_size(const char* op, int n)
{
    if (n == 4) return run<T, 4, true>(op);
    if (n == 6) return run<T, 6, false>(op);
    if (n == 8) return run<T, 8, true>(op);
    if (n == 12) return run<T, 12, false>(op);
    if (n == 16) return run<T, 16, false>(op);
    if (n == 24) return run<T, 24, false>(op);
    if (n == 32) return run<T, 32, false>(op);
    std::fprintf(stderr, "N must be one of 4,6,8,12,16,24,32\n");
    return 2;
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s OP <f32|f64> <4|6|8|12|16|24|32>\n", argv[0]);
        return 2;
    }
    const int n = std::atoi(argv[3]);
    if (!std::strcmp(argv[2], "f32")) return dispatch_size<float>(argv[1], n);
    if (!std::strcmp(argv[2], "f64")) return dispatch_size<double>(argv[1], n);
    std::fprintf(stderr, "dtype must be f32 or f64\n");
    return 2;
}
