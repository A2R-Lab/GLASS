#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;


///TODO: error checking
///TODO: transpose A

template <typename T, bool TRANSPOSE_B>
__device__
void gemm(std::uint32_t m,
          std::uint32_t n,
          std::uint32_t k,
          T alpha, 
          T *A, 
          T *B,
          T beta,
          T *C, 
          cgrps::thread_group g)
{
    if(TRANSPOSE_B){
        const unsigned max = m*n;
        uint32_t element, ind, row, col;
        T res;

        for(element = g.thread_rank(); element < max; element += g.size()){
            res = static_cast<T>(0);
            row = element % m;
            col = element / m;

            for(ind = 0; ind < n; ind++){
                res += A[ind*m + row] * B[ind*n + col];
            }

            C[col*m + row] = res;
        }
    }
    else{
        const unsigned max = m*k;
        uint32_t element, ind, row, col;
        T res;

        for(element = g.thread_rank(); element < max; element += g.size()){
            res = static_cast<T>(0);
            row = element % m;
            col = element / m;

            for(ind = 0; ind < n; ind++){
                res += A[ind*m + row] * B[col*n + ind];
            }

            C[col*m + row] = res;
        }
    }
}