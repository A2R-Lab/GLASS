#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

template <typename T>
__device__
void transpose(uint32_t N, uint32_t M, T* a, T* b, cgrps::thread_group g = cgrps::this_thread_block()) {
    // transpose matrix a, and NxM matrix, and store the result in b
    // matrix a is assumed to be in column-major order
    // resulting matrix b will also be in column-major order
    uint32_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = ind; i < N * M; i += stride) {
        uint32_t col = i / N;
        uint32_t row = i % N;
        b[col + M * row] = a[row + N * col];
    }

    g.sync();
}