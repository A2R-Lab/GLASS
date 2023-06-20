#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;

// load identity in so memory is [A | I]
template <typename T>
__device__
void loadIdentity(uint32_t dimA, 
                  T *A){
    for (unsigned ind = threadIdx.x; ind < dimA*dimA; ind += blockDim.x){
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
                  T *B){
    for (unsigned ind = threadIdx.x; ind < dimA*dimA+dimB*dimB; ind += blockDim.x){
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
                  T *C){
    for (unsigned ind = threadIdx.x; ind < dimA*dimA+dimB*dimB+dimC*dimC; ind += blockDim.x){
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
          T alpha)
{
    for(uint32_t i = threadIdx.x; i < n * n; i += blockDim.x){
        if(i % n == i / n){ A[i] += alpha; }
    }
}