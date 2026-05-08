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