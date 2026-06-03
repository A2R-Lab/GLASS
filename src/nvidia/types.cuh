#pragma once
#include <cstdint>
#include <cassert>

// glass::nvidia::layout — matrix memory layout for cuBLASDx-backed wrappers.
// Maps directly to cublasdx::Arrangement<>: col_major <-> cublasdx::col_major,
// row_major <-> cublasdx::row_major. Numeric values are part of the public ABI
// because the DEFINE_NVIDIA_GEMM*_LAYOUT* macros take integer literals (0/1)
// and static_cast them back to this enum in the explicit specialization.
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
           "(see glass-rfc-batched-1d.md)");                                    \
    assert((blockDim.x * blockDim.y * blockDim.z) >=                            \
           (GEMM_T::block_dim.x * GEMM_T::block_dim.y * GEMM_T::block_dim.z) && \
           "glass::nvidia: total threads < GEMM::block_dim product");
#endif
