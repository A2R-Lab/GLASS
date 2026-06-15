#pragma once
/**
 * @file glass-cgrps.cuh
 * @brief Umbrella header for the cooperative-groups SIMT `glass::cgrps::` namespace.
 *
 * Pulls in the L1/L2/L3 hand-rolled SIMT surface in a variant that drives
 * cooperation through a cooperative_groups handle (`g.thread_rank()` /
 * `g.size()`) rather than raw `threadIdx` / `blockDim`. This lets the same
 * primitives run over a sub-block tile (e.g. a warp via
 * `cgrps::tiled_partition<32>(...)`) as well as the whole block. The default
 * group is the full thread block, matching the plain `glass::` behaviour.
 *
 * Requires <cooperative_groups.h>. Include glass.cuh for the dependency-free
 * threadIdx-based variants, or glass-nvidia.cuh for the vendor-accelerated path.
 */
#include "glass.cuh"
#include <cooperative_groups.h>

namespace glass {
namespace cgrps {
    #include "./src/cgrps/l1.cuh"
    #include "./src/cgrps/l2.cuh"
    #include "./src/cgrps/l3.cuh"
} // namespace cgrps
} // namespace glass
