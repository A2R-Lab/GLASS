#pragma once
#include <cstdint>

/**
 * @file barrier.cuh
 * @brief Barrier policy for the shared `*_impl` bodies (cgrps dedup + TRAILING_SYNC).
 *
 * Every public single-block op is written ONCE as a `*_impl(Bar bar, ...)` body
 * templated on a *barrier policy* that carries the thread `rank()`/`size()` and a
 * `sync()`. The plain `glass::` surface passes `BlockBarrier` (threadIdx/blockDim
 * + `__syncthreads()`); the `glass::cgrps::` surface passes a `GroupBarrier`
 * (cooperative-groups handle + `cooperative_groups::sync`, defined in
 * `src/cgrps/`). Routing BOTH the internal and the trailing barrier through
 * `bar.sync()` is what lets one body serve both surfaces — that barrier
 * primitive is the only thing that ever differed between them.
 *
 * `BlockBarrier` names no cooperative-groups type, so `glass.cuh` stays
 * dependency-free; the `GroupBarrier` twin is only compiled when a caller
 * includes `<cooperative_groups.h>` via `glass-cgrps.cuh`.
 *
 * Uniformity rule (project-wide): every public op takes `bool TRAILING_SYNC=true`
 * and ends on `if constexpr (TRAILING_SYNC) bar.sync();`, so the result is valid
 * for ALL threads by default. Callers that own the following barrier pass
 * `false` to elide it. For ops that already ended on a barrier this is
 * byte-identical; for the elementwise/reduce-tail ops it adds a strictly-safer
 * trailing barrier.
 */
struct BlockBarrier {
    __device__ __forceinline__ uint32_t rank() const {
        return threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    }
    __device__ __forceinline__ uint32_t size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
    __device__ __forceinline__ void sync() const { __syncthreads(); }
};
