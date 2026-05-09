#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void transpose(uint32_t N, uint32_t M, T* a, T* b, cgrps::thread_group g = cgrps::this_thread_block()) {
    // transpose matrix a, an NxM matrix, and store the result in b
    // matrix a is assumed to be in column-major order
    // resulting matrix b will also be in column-major order
    for (uint32_t i = g.thread_rank(); i < N * M; i += g.size()) {
        uint32_t col = i / N;
        uint32_t row = i % N;
        b[col + M * row] = a[row + N * col];
    }

    g.sync();
}

// Transpose in place; assumes matrix A is size N x N and is stored in column major order
template <typename T>
__device__
void transpose(uint32_t N, T* a, cgrps::thread_group g = cgrps::this_thread_block()) {
    for (uint32_t index = g.thread_rank(); index < N * N; index += g.size()) {
        uint32_t i = index % N;
        uint32_t j = index / N;

        if (i < j) {
            uint32_t swapIndex = i * N + j;

            T temp = a[index];
            a[index] = a[swapIndex];
            a[swapIndex] = temp;
        }
    }

    g.sync();
}

// === glass::simple variants ===
namespace simple {
    // Transpose NxM matrix a (column-major) into b (column-major).
    template <typename T>
    __device__ void transpose(uint32_t N, uint32_t M, T* a, T* b)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t i = rank; i < N * M; i += size) {
            uint32_t col = i / N;
            uint32_t row = i % N;
            b[col + M * row] = a[row + N * col];
        }
        __syncthreads();
    }

    // In-place transpose of N x N matrix (column-major).
    template <typename T>
    __device__ void transpose(uint32_t N, T* a)
    {
        uint32_t rank = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        uint32_t size = blockDim.x * blockDim.y * blockDim.z;
        for (uint32_t index = rank; index < N * N; index += size) {
            uint32_t i = index % N;
            uint32_t j = index / N;
            if (i < j) {
                uint32_t swapIndex = i * N + j;
                T temp = a[index];
                a[index] = a[swapIndex];
                a[swapIndex] = temp;
            }
        }
        __syncthreads();
    }
}
// ===