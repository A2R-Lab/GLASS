#pragma once
// Pre-include system headers at global scope so they are not pulled into the
// namespace glass { } block when the base files include them via #pragma once.
#include <cstdint>
#include <math.h>

namespace glass {
    /*      L1      */
    #include "./src/base/L1/reduce.cuh"
    #include "./src/base/L1/axpy.cuh"
    #include "./src/base/L1/copy.cuh"
    #include "./src/base/L1/dot.cuh"
    #include "./src/base/L1/ident.cuh"
    #include "./src/base/L1/scal.cuh"
    #include "./src/base/L1/swap.cuh"
    #include "./src/base/L1/elementwise_logic.cuh"
    #include "./src/base/L1/transpose.cuh"
    #include "./src/base/L1/prefix_sum.cuh"
    #include "./src/base/L1/norm.cuh"
    #include "./src/base/L1/l2norm.cuh"
    #include "./src/base/L1/infnorm.cuh"
    #include "./src/base/L1/clip.cuh"
    #include "./src/base/L1/set_const.cuh"
    #include "./src/base/L1/asum.cuh"

    /*      L2      */
    #include "./src/base/L2/gemv.cuh"
    #include "./src/base/L2/ger.cuh"

    /*      L3      */
    #include "./src/base/L3/gemm.cuh"
    #include "./src/base/L3/inv.cuh"
    #include "./src/base/L3/chol_InPlace.cuh"
    #include "./src/base/L3/trsm.cuh"
}

// Host helper: shared-memory bytes needed for glass::gemm_dispatch (tiled path).
// Returns 0 if tiling is not warranted (m >= 32 or m*k > block_threads).
// Usage:
//   size_t smem = glass_gemm_dispatch_smem<float>(m, k);
//   kernel<<<grid, 256, smem>>>(m, n, k, A, B, C);
// Inside the kernel:
//   extern __shared__ T scratch[];
//   glass::gemm_dispatch(m, n, k, alpha, A, B, beta, C,
//       (smem > 0) ? scratch : nullptr,
//       (smem > 0) ? scratch + m * 8 : nullptr);
template <typename T = float>
inline std::size_t glass_gemm_dispatch_smem(int m, int k,
                                            int block_threads = 256, int tile = 8)
{
    if (m < 32 && m * k <= block_threads)
        return static_cast<std::size_t>(m * tile + tile * k) * sizeof(T);
    return 0;
}
