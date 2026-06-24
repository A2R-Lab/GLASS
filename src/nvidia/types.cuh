#pragma once
/**
 * @file types.cuh
 * @brief Public types and shared helper macros for the `glass::nvidia::` wrappers.
 *
 * Defines the user-facing `glass::nvidia::layout` enum (per-matrix memory
 * order, used as the LA/LB/LC template arguments of gemm/gemv/...) plus the
 * private helper macros (_GLASS_CUBLAS_LAYOUT, _GLASS_ASSERT_BLOCKDIM_GEQ)
 * shared by l2.cuh and l3.cuh. Included before l1/l2/l3/lapack so `layout` is
 * in scope inside the `glass::nvidia` namespace.
 */
#include <cstdint>
#include <cassert>

/**
 * @brief Matrix memory layout for the cuBLASDx-backed `glass::nvidia::` wrappers.
 *
 * Maps directly to `cublasdx::Arrangement<>`: `col_major` (Fortran/cuBLAS
 * default) and `row_major` (C-style). Passed per matrix as the LA/LB/LC
 * template arguments of gemm/gemv/row_strided_*. The numeric values (0/1) are
 * part of the public ABI: the `DEFINE_NVIDIA_*_LAYOUT*` macros take integer
 * literals and static_cast them back to this enum in their specializations.
 */
namespace glass {
namespace nvidia {

    enum class layout : uint8_t {
        col_major = 0,
        row_major = 1,
    };

} // namespace nvidia
} // namespace glass

// ---------------------------------------------------------------------------
// Shared private helper macros used by both l2.cuh (gemv) and l3.cuh (gemm).
// The cublasdx:: names referenced here are resolved at the expansion site, so
// this header does NOT include <cublasdx.hpp>.
// ---------------------------------------------------------------------------

// Convert integer macro arg (0 or 1) to a cublasdx::arrangement enum value.
#define _GLASS_CUBLAS_LAYOUT(L) ((L) ? cublasdx::row_major : cublasdx::col_major)

// Debug-only assertion that the launched blockDim has at least as many threads
// (in each rank and in total) as the GEMM/GEMV's required block_dim. With
// BlockDim<TC,1,1>() pinned via the operator, "extra" threads in the same rank
// are explicitly allowed by cuBLASDx — see
//   /opt/nvidia/mathdx/25.12/example/cublasdx/04_gemm_blockdim/blockdim_gemm_fp16.cu
#ifdef NDEBUG
#define _GLASS_ASSERT_BLOCKDIM_GEQ(GEMM_T) /* nothing */
#else
#define _GLASS_ASSERT_BLOCKDIM_GEQ(GEMM_T)                                      \
    assert(blockDim.x >= GEMM_T::block_dim.x &&                                 \
           blockDim.y >= GEMM_T::block_dim.y &&                                 \
           blockDim.z >= GEMM_T::block_dim.z &&                                 \
           "glass::nvidia: launched blockDim < GEMM::block_dim "                \
           "(see the Batched-1D concept guide in docs)");                       \
    assert((blockDim.x * blockDim.y * blockDim.z) >=                            \
           (GEMM_T::block_dim.x * GEMM_T::block_dim.y * GEMM_T::block_dim.z) && \
           "glass::nvidia: total threads < GEMM::block_dim product");
#endif
