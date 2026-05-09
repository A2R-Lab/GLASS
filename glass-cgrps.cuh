#pragma once
#include "glass.cuh"
#include <cooperative_groups.h>

namespace glass {
namespace cgrps {
    #include "./src/cgrps/l1.cuh"
    #include "./src/cgrps/l2.cuh"
    #include "./src/cgrps/l3.cuh"
} // namespace cgrps
} // namespace glass
