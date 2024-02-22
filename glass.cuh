#pragma once

namespace glass{
    namespace column_major {
        /*      L1      */
        #include "./src/L1/axpy.cuh"
        #include "./src/L1/copy.cuh"
        #include "./src/L1/dot.cuh"
        #include "./src/L1/ident.cuh"
        #include "./src/L1/scal.cuh"
        #include "./src/L1/set_const.cuh"
        #include "./src/L1/swap.cuh"
        #include "./src/L1/l2norm.cuh"
        #include "./src/L1/infnorm.cuh"
        #include "./src/L1/clip.cuh"

        /*      L2      */
        #include "./src/column_major/L2/gemv.cuh"

        /*      L3      */
        #include "./src/column_major/L3/chol_InPlace.cuh"
        #include "./src/column_major/L3/gemm.cuh"
        #include "./src/column_major/L3/inv.cuh"
    }
}
