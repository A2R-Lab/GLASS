#pragma once
#include <cooperative_groups.h>
namespace glass{
    /*      L1      */
    #include "./src/L1/reduce.cuh"
    #include "./src/L1/axpy.cuh"
    #include "./src/L1/copy.cuh"
    #include "./src/L1/dot.cuh"
    #include "./src/L1/ident.cuh"
    #include "./src/L1/scal.cuh"
    #include "./src/L1/swap.cuh"
    #include "./src/L1/elementwise_logic.cuh"
    #include "./src/L1/transpose.cuh"
    #include "./src/L1/prefix_sum.cuh"
    #include "./src/L1/norm.cuh"
    #include "./src/L1/l2norm.cuh"
    #include "./src/L1/infnorm.cuh"
    #include "./src/L1/clip.cuh"
    #include "./src/L1/set_const.cuh"
    #include "./src/L1/asum.cuh"

    /*      L2      */
    #include "./src/L2/gemv.cuh"
    #include "./src/L2/ger.cuh"

    /*      L3      */
    #include "./src/L3/gemm.cuh"
    #include "./src/L3/inv.cuh"
    #include "./src/L3/chol_InPlace.cuh"
    #include "./src/L3/trsm.cuh"
    #include "./src/L3/cpqp.cuh"
}

// ─── Host helper: shared-memory bytes needed for glass::simple::gemm_dispatch ─
// Returns the scratch size to pass as the third argument of <<<grid, block, smem>>>.
// Returns 0 if tiling is not warranted (m >= 32 or m*k > block_threads).
// Usage:
//   size_t smem = glass_gemm_dispatch_smem<float>(m, k);
//   kernel<<<grid, 256, smem>>>(m, n, k, A, B, C);
// Inside the kernel:
//   extern __shared__ T scratch[];
//   glass::simple::gemm_dispatch(m, n, k, alpha, A, B, beta, C,
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