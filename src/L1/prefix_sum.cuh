#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

/*
To be called from within a block
Assumes that the size of input/output is <= block size
Handles odd size arrays
Could use some more optimizations
*/
template <typename T> 
__device__ void prefix_sum_exclusive(T* s_input, T* s_output, int n) {
    int tid = threadIdx.x;

    if (tid < n) {
        s_output[tid] = (tid > 0) ? s_input[tid - 1] : 0;
    } else {
        // Handle case where tid >= n for completeness, though not expected to execute
        s_output[tid] = 0;
    }
    __syncthreads();

    // Perform the scan on s_output for an exclusive result
    T temp;
    for (int d = 1; d < n; d *= 2) {
        __syncthreads();
        if (tid < n && tid >= d) {
            temp = s_output[tid] + s_output[tid - d];
        }
        __syncthreads();
        if (tid < n && tid >= d) {
            s_output[tid] = temp;
        }
    }
}
