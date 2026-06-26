#pragma once
/**
 * @file glass.cuh
 * @brief Umbrella header for the hand-rolled SIMT `glass::` namespace (no deps).
 *
 * Pulls in the full pure-SIMT, single-block BLAS/LAPACK surface — L1 vector
 * ops, L2 matrix-vector (gemv/ger), and L3 matrix ops (gemm, inv, Cholesky,
 * trsm) — all as `__device__` helpers that cooperate across one CUDA block
 * using `threadIdx` / `blockDim` directly (no cooperative-groups dependency).
 * Every op offers runtime-size and compile-time-size (`<T, N, ...>`) overloads.
 *
 * Include glass-cgrps.cuh for the cooperative-groups variants, or
 * glass-nvidia.cuh for the CUB / cuBLASDx / cuSOLVERDx-accelerated paths. Also
 * defines the host helper ::glass_gemm_dispatch_smem below.
 */
// Pre-include system headers at global scope so they are not pulled into the
// namespace glass { } block when the base files include them via #pragma once.
#include <cstdint>
#include <cstddef>
#include <math.h>

namespace glass {
    /*  barrier policy (shared *_impl bodies; BlockBarrier = threadIdx + __syncthreads)  */
    #include "./src/base/barrier.cuh"

    /*      L1      */
    #include "./src/base/L1/reduce.cuh"
    #include "./src/base/L1/axpy.cuh"
    #include "./src/base/L1/copy.cuh"
    #include "./src/base/L1/dot.cuh"
    #include "./src/base/L1/dot_strided.cuh"
    #include "./src/base/L1/dot_strided_coalesced.cuh"
    #include "./src/base/L1/ident.cuh"
    #include "./src/base/L1/scal.cuh"
    #include "./src/base/L1/swap.cuh"
    #include "./src/base/L1/elementwise_logic.cuh"
    #include "./src/base/L1/transpose.cuh"
    #include "./src/base/L1/prefix_sum.cuh"
    #include "./src/base/L1/norm.cuh"
    #include "./src/base/L1/nrm2.cuh"
    #include "./src/base/L1/infnorm.cuh"
    #include "./src/base/L1/iamax.cuh"
    #include "./src/base/L1/clip.cuh"
    #include "./src/base/L1/set_const.cuh"
    #include "./src/base/L1/asum.cuh"
    #include "./src/base/L1/nrm1_diff.cuh"
    #include "./src/base/L1/axpy_strided.cuh"
    #include "./src/base/L1/copy_strided.cuh"

    /*      L2      */
    #include "./src/base/L2/gemv.cuh"
    #include "./src/base/L2/gemv_reduced.cuh"
    #include "./src/base/L2/trsv.cuh"
    #include "./src/base/L2/ger.cuh"
    #include "./src/base/L2/gemv_strided.cuh"
    #include "./src/base/L2/gemv_segmented.cuh"

    /*      L3      */
    #include "./src/base/L3/gemm.cuh"
    #include "./src/base/L3/gemm_reduced.cuh"
    #include "./src/base/L3/syrk_reduced.cuh"
    #include "./src/base/L3/tensor_contract.cuh"
    #include "./src/base/L3/congruence.cuh"
    #include "./src/base/L3/syrk.cuh"
    #include "./src/base/L3/gemm_strided.cuh"
    #include "./src/base/L3/gemm_batched_indexed.cuh"
    #include "./src/base/L3/inv.cuh"
    #include "./src/base/L3/chol_InPlace.cuh"
    #include "./src/base/L3/trsm.cuh"
    #include "./src/base/L3/ldlt.cuh"
    #include "./src/base/L3/posv.cuh"
    #include "./src/base/L3/riccati.cuh"

    /*  block-tridiagonal: glass::bdmv (matvec) + glass::pcg (solver)  */
    #include "./src/base/banded/bdmv.cuh"
    #include "./src/base/banded/block_access.cuh"
    #include "./src/base/pcg/solve.cuh"
}

/**
 * @brief Host helper: shared-memory bytes needed for glass::gemm_dispatch (tiled path).
 *
 * Compute on the host at launch time and pass as the kernel's dynamic-smem
 * argument. Returns 0 when tiling is not warranted (m >= 32 or
 * m*n > block_threads), in which case glass::gemm_dispatch runs the plain
 * (non-tiled) path. Host-callable. Standard convention: C is m×n, contraction k;
 * the tiled path stages an `m×TILE` A-tile and a `TILE×n` B-tile.
 *
 * Usage:
 *   size_t smem = glass_gemm_dispatch_smem<float>(m, n);
 *   kernel<<<grid, 256, smem>>>(m, n, k, A, B, C);
 *   // inside the kernel:
 *   extern __shared__ T scratch[];
 *   glass::gemm_dispatch(m, n, k, alpha, A, B, beta, C,
 *       (smem > 0) ? scratch : nullptr,
 *       (smem > 0) ? scratch + m * 8 : nullptr);
 *
 * @tparam T            Scalar type (defaults to float).
 * @param  m            Rows of A / C.
 * @param  n            Columns of B / C.
 * @param  block_threads Launch thread count used for the tiling heuristic.
 * @param  tile         Tile width (must match the gemm_tiled<T, TILE> used).
 * @return Bytes of dynamic shared memory to allocate, or 0 for the plain path.
 */
template <typename T = float>
inline std::size_t glass_gemm_dispatch_smem(int m, int n,
                                            int block_threads = 256, int tile = 8)
{
    if (m < 32 && m * n <= block_threads)
        return static_cast<std::size_t>(m * tile + tile * n) * sizeof(T);
    return 0;
}
