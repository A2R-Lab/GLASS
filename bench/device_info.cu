// device_info.cu — provenance probe for benchmark captures.
// Prints every cudaDeviceProp field a paper table cares about (name, CC,
// SM count, clocks, memory) so a results tarball is self-describing.
//   nvcc -std=c++17 -O2 device_info.cu -o build/device_info
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess || n == 0) {
        std::printf("no CUDA device: %s\n", cudaGetErrorString(err));
        return 1;
    }
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        std::printf("device=%d name=\"%s\"\n", d, p.name);
        std::printf("  compute_capability=%d.%d (sm_%d%d)\n",
                    p.major, p.minor, p.major, p.minor);
        std::printf("  multiprocessors=%d\n", p.multiProcessorCount);
        std::printf("  clock_khz=%d mem_clock_khz=%d\n",
                    p.clockRate, p.memoryClockRate);
        std::printf("  global_mem_mb=%zu\n", p.totalGlobalMem >> 20);
        std::printf("  shared_per_block_kb=%zu shared_per_sm_kb=%zu\n",
                    p.sharedMemPerBlock >> 10, p.sharedMemPerMultiprocessor >> 10);
        std::printf("  regs_per_block=%d regs_per_sm=%d\n",
                    p.regsPerBlock, p.regsPerMultiprocessor);
        std::printf("  l2_cache_kb=%d mem_bus_bits=%d\n",
                    p.l2CacheSize >> 10, p.memoryBusWidth);
        std::printf("  integrated=%d ecc=%d\n", p.integrated, p.ECCEnabled);
    }
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    std::printf("cuda_runtime=%d cuda_driver=%d\n", rt, drv);
    return 0;
}
