#pragma once
/**
 * @file glass.cuh
 * @brief Umbrella header for the hand-rolled SIMT surface (no deps).
 *
 * Pulls in the full pure-SIMT, single-block BLAS/LAPACK surface — L1 vector
 * ops, L2 matrix-vector (gemv/ger), and L3 matrix ops (gemm, inv, Cholesky,
 * trsm) — all as `__device__` helpers that cooperate across one CUDA block
 * using `threadIdx` / `blockDim` directly (no cooperative-groups dependency).
 * Every op offers runtime-size and compile-time-size (`<T, N, ...>`) overloads.
 *
 * NAMESPACE CONTRACT (2026-07-30 restructure):
 *   - `glass::block::`  — the explicit block-scope SIMT tier. CONTRACT tier:
 *                         bit-exact, thread-count invariant, never re-dispatched.
 *   - `glass::warp::`   — one problem per warp (alias of block::warp — the
 *                         warp mirrors live inline in the same base headers).
 *   - `glass::thread::` — one problem per thread (alias of block::thread).
 *   - `glass::op` BARE  — the measured-DEFAULT face: block-scope calling
 *                         contract, implementation chosen per (op, size, dtype)
 *                         from the shipped dispatch table. Phase 1 pins every
 *                         cell to the block body (`glass::dispatch_body()` in
 *                         glass-defaults.cuh), so today the bare names ARE the
 *                         block tier via the using-directive below — identical
 *                         symbols, bit-identical results. A future measured
 *                         retune may add shadowing wrappers that route specific
 *                         cells to a warp- or thread-body executed under the
 *                         same block-scope contract (all threads enter, result
 *                         valid after return); such a retune is an attested
 *                         event, and determinism-sensitive consumers should pin
 *                         `glass::block::` explicitly.
 *
 * Include glass-cgrps.cuh for the cooperative-groups variants, or
 * glass-nvidia.cuh for the CUB / cuBLASDx / cuSOLVERDx-accelerated paths
 * (`glass::nvidia::block::` / `glass::nvidia::warp::`, with the same
 * bare-name re-export inside `glass::nvidia::`). Also defines the host
 * helper ::glass_gemm_dispatch_smem below.
 */
// Pre-include system headers at global scope so they are not pulled into the
// namespace glass { } block when the base files include them via #pragma once.
#include <cstdint>
#include <cstddef>
#include <math.h>

namespace glass {
namespace block {
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
    #include "./src/base/L1/rot.cuh"
    #include "./src/base/L1/elementwise_logic.cuh"
    #include "./src/base/L1/symmetrize.cuh"
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
    #include "./src/base/L1/softmax.cuh"
    #include "./src/base/L1/argreduce.cuh"

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
    #include "./src/base/L3/symm.cuh"
    #include "./src/base/L3/gemm_batched_indexed.cuh"
    #include "./src/base/L3/inv.cuh"
    #include "./src/base/L3/potrf.cuh"
    #include "./src/base/L3/trsm.cuh"
    #include "./src/base/L3/getrf.cuh"
    #include "./src/base/L3/ldlt.cuh"
    #include "./src/base/L3/posv.cuh"
    #include "./src/base/L3/syev.cuh"
    #include "./src/base/L3/eigh.cuh"
    #include "./src/base/L3/riccati.cuh"
    #include "./src/base/L3/gn_step.cuh"

    /*  block-tridiagonal: glass::bdmv (matvec), glass::bdsv (direct solve),
        glass::pcg (iterative solver)  */
    #include "./src/base/banded/bdmv.cuh"
    #include "./src/base/banded/block_access.cuh"
    #include "./src/base/banded/bdsv.cuh"
    #include "./src/base/pcg/solve.cuh"

    /*  robotics-specialized operators (see docs: robotics_conventions) —
        Lie/quaternion family (angle → quat → so3 → se3 → pose, dependency
        order), Featherstone spatial 6-D ops (cross products, coordinate
        transforms, 10-parameter inertia), projections/cones/AL scalars,
        geometry distance primitives, and the 3x3 estimation kit
        (eig3/svd3/closest_rotation). All array-shaped ops span the block/
        warp/thread interfaces; scalar ops are tier-free.  */
    #include "./src/base/lie/angle.cuh"
    #include "./src/base/lie/quat.cuh"
    #include "./src/base/lie/so3.cuh"
    #include "./src/base/lie/se3.cuh"
    #include "./src/base/lie/pose.cuh"
    #include "./src/base/spatial/cross.cuh"
    #include "./src/base/spatial/transform.cuh"
    #include "./src/base/spatial/inertia.cuh"
    #include "./src/base/proj/cone.cuh"
    #include "./src/base/proj/interval.cuh"
    #include "./src/base/geom/sphere.cuh"
    #include "./src/base/geom/frame.cuh"
    #include "./src/base/geom/segment.cuh"
    #include "./src/base/est/svd3.cuh"
}  // namespace block

/*  The bare glass:: face. Phase 1 of the dispatch contract: every cell of
    dispatch_body() (glass-defaults.cuh) pins to the block body, so the bare
    names resolve to the SAME entities as glass::block:: via this directive
    (function-pointer identical — pinned by static_asserts in test_defaults).
    The warp/thread sub-namespaces are hoisted back to glass:: scope: they
    are written inline in the base headers, so under the block wrap they land
    at block::warp / block::thread; the aliases keep the public spellings. */
using namespace block;
namespace warp   = block::warp;
namespace thread = block::thread;
}  // namespace glass

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
