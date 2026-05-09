#pragma once
#include "glass.cuh"

// System headers must be included before namespace wrapping so their internal
// declarations land at global scope. The re-includes inside the namespace below
// are no-ops due to #pragma once.
#include <cstdint>
#include <cub/cub.cuh>

#ifdef __CUBLASDX_HPP__
#define GLASS_HAVE_CUBLASDX 1
#else
// cublasdx.hpp not yet included; try to include it now.
// Set MATHDX_ROOT and pass -I$MATHDX_ROOT/include -I$MATHDX_ROOT/external/cutlass/include
#ifdef GLASS_BENCH_CUBLASDX
#include <cublasdx.hpp>
#define GLASS_HAVE_CUBLASDX 1
#else
#define GLASS_HAVE_CUBLASDX 0
#endif
#endif

namespace glass {
namespace nvidia {

    // L1: CUB-backed reduce / dot / l2norm
    #include "./src/nvidia/l1.cuh"

#if GLASS_HAVE_CUBLASDX
    // L2: cuBLASDx-backed gemv  (primary templates + DEFINE_NVIDIA_GEMV macro)
    #include "./src/nvidia/l2.cuh"

    // L3: cuBLASDx-backed gemm  (primary templates + DEFINE_NVIDIA_GEMM macro)
    #include "./src/nvidia/l3.cuh"

    // Pre-instantiated standard sizes (matches benchmark suite)
    DEFINE_NVIDIA_GEMV(4,   4)
    DEFINE_NVIDIA_GEMV(6,   6)
    DEFINE_NVIDIA_GEMV(8,   8)
    DEFINE_NVIDIA_GEMV(12, 12)
    DEFINE_NVIDIA_GEMV(14, 14)
    DEFINE_NVIDIA_GEMV(24, 24)
    DEFINE_NVIDIA_GEMV(64, 64)

    DEFINE_NVIDIA_GEMM(4,   4,  4)
    DEFINE_NVIDIA_GEMM(6,   6,  6)
    DEFINE_NVIDIA_GEMM(8,   8,  8)
    DEFINE_NVIDIA_GEMM(12, 12, 12)
    DEFINE_NVIDIA_GEMM(14, 14, 14)
    DEFINE_NVIDIA_GEMM(24, 24, 24)
    DEFINE_NVIDIA_GEMM(64, 64, 64)
#endif // GLASS_HAVE_CUBLASDX

} // namespace nvidia
} // namespace glass
