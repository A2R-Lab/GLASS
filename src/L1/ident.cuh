#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

/**
 * Loads the identity matrix into a specified memory region.
 *
 * This function appends an identity matrix to a square matrix `A` of type `T`.
 * So the result will b [A | I] where `A` is a square matrix of size `dimA` and `I` is the identity matrix of size `dimA`.
 * The matrix `A` must be stored in device memory
 *
 * @tparam T         The type of elements in the matrix `A`.
 * @param  dimA      The dimension of the square matrix `A`.
 * @param  A         Pointer next memory address after the memory region representing matrix `A`.
 * @param  g         (Optional) Thread group specifying the thread block to use for parallel execution.
 *                   Defaults to the current thread block obtained using `cgrps::this_thread_block()`.
 */
template <typename T>
__device__
void loadIdentity(uint32_t dimA, 
                  T *A,
                  cgrps::thread_group g = cgrps::this_thread_block()){
    for (unsigned ind = g.thread_rank(); ind < dimA*dimA; ind += g.size()){
        unsigned r, c;
        r = ind % dimA; 
        c = ind / dimA;
        A[ind] = static_cast<T>(r == c);
    }
}

// load identity in so memory is [V | I]
template <typename T>
__device__
void loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B,
                  cgrps::thread_group g = cgrps::this_thread_block()){
    for (unsigned ind = g.thread_rank(); ind < dimA*dimA+dimB*dimB; ind += g.size()){
        unsigned r, c, indAdj; T *V;
        if (ind < dimA*dimA){
            indAdj = ind;
            r = indAdj % dimA; c = indAdj/dimA; V = A;
        }
        else {
            indAdj = ind - dimA*dimA;
            r = indAdj % dimB; c = indAdj/dimB; V = B;
        }
        V[indAdj] = static_cast<T>(r == c);
    }
}

// load identity in so memory is [V | I]
template <typename T>
__device__
void loadIdentity(uint32_t dimA, 
                  T *A, 
                  uint32_t dimB, 
                  T *B, 
                  uint32_t dimC, 
                  T *C,
                  cgrps::thread_group g = cgrps::this_thread_block()){
    for (unsigned ind = g.thread_rank(); ind < dimA*dimA+dimB*dimB+dimC*dimC; ind += g.size()){
        unsigned r, c, indAdj; T *V;
        if (ind < dimA*dimA){
            indAdj = ind;
            r = indAdj % dimA; c = indAdj/dimA; V = A;
        }
        else if (ind < dimA*dimA+dimB*dimB){
            indAdj = ind - dimA*dimA;
            r = indAdj % dimB; c = indAdj/dimB; V = B;
        }
        else{
            indAdj = ind - dimA*dimA - dimB*dimB;
            r = indAdj % dimC; c = indAdj/dimC; V = C;
        }
        V[indAdj] = static_cast<T>(r == c);
    }
}


template <typename T>
__device__
void addI(uint32_t n,
          T *A,
          T alpha,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(uint32_t i = g.thread_rank(); i < n * n; i += g.size()){
        if(i % n == i / n){ A[i] += alpha; }
    }
}