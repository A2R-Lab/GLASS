#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;


template <typename T>
__device__
void invertMatrix(uint32_t dimA, T *A, T *s_temp, cgrps::thread_group g){ 
// we are going to guassian elimination walking down the matrix (assuming no leading 0s)
// we therefore use the columns in order as the pivot column for each pivot we need to rescale 
// that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
// of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
// pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++){
        unsigned pivColOffset = pivRC*dimA;
        // save the pivot and pivot column and row
        T pvInv = static_cast<T>(1)/A[pivRC + pivColOffset];
        for (unsigned ind = GATO_THREAD_ID; ind < 2*dimA+1; ind++){
            unsigned AInd;
            if (ind < dimA){AInd = ind + pivColOffset;}
            else{AInd = pivRC + pivColOffset + (ind-dimA)*dimA;}
            s_temp[ind] = A[AInd];
        }
        g.sync();
        // make the pivot update
        for (unsigned ind = GATO_THREAD_ID; ind < dimA*(dimA+1); ind += GATO_THREADS_PER_BLOCK){
            unsigned row = ind % dimA; unsigned col = ind / dimA; unsigned colOffset = ind - row;
            // s_temp = orpcvs|prvOld
            if (row == pivRC){A[row + pivColOffset + colOffset] *= pvInv;}
            else{A[row + pivColOffset + colOffset] -= s_temp[row]*pvInv*s_temp[dimA+col];}
        }
        g.sync();
    }
}


template <typename T>
__device__
void invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, uint32_t dimMax, T *s_temp, cgrps::thread_group g){

    // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
    // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
    // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
    // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
    // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    T *s_memA = s_temp; T *s_memB = &s_memA[2*dimA+1];
    for (unsigned pivRC = 0; pivRC < dimMax; pivRC++){
        bool AActive = pivRC < dimA; bool BActive = pivRC < dimB;
        unsigned pivColOffsetA = pivRC*dimA; unsigned pivColOffsetB = pivRC*dimB;
        // save the pivot column and row
        for (unsigned ind = GATO_THREAD_ID; ind < dimMax; ind++){
            if (AActive && ind < dimA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < dimB){s_memB[ind] = B[ind + pivColOffsetB];}
        }
        for (unsigned ind = GATO_THREAD_ID; ind < dimMax+1; ind++){
            if (AActive && ind < dimA+1){s_memA[ind + dimA] = A[ind*dimA + pivRC + pivColOffsetA];}
            if (BActive && ind < dimB+1){s_memB[ind + dimB] = B[ind*dimB + pivRC + pivColOffsetB];}
        }
        g.sync();
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = GATO_THREAD_ID; ind < dimMax*(dimMax+1); ind += GATO_THREADS_PER_BLOCK){
            if (AActive && ind < dimA*(dimA+1)){
                unsigned row = ind % dimA; unsigned col = ind / dimA;
                if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[dimA+col];}
            }
            if (BActive && ind < dimB*(dimB+1)){
                unsigned row = ind % dimB; unsigned col = ind / dimB; 
                if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[dimB+col];}
            }
        }
        g.sync();
    }
}

// invert A,B,C assume memory for all is [V | VInv] where both are DIMxDIM and continguous
// relies on s_temp of size [2*dimA + 2*dimB + 2*dimC + 3]
template <typename T>
__device__
void invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, uint32_t dimC, T *C, uint32_t dimMax, T *s_temp, cgrps::thread_group g){

    // now we are going to guassian elimination walking down the matrix (assuming no leading 0s)
    // we therefore use the columns in order as the pivot column for each pivot we need to rescale 
    // that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
    // of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
    // pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    T *s_memA = s_temp; T *s_memB = &s_memA[2*dimA+1]; T *s_memC = &s_memB[2*dimB+1];
    for (unsigned pivRC = 0; pivRC < dimMax; pivRC++){
        bool AActive = pivRC < dimA; bool BActive = pivRC < dimB; bool CActive = pivRC < dimC;
        unsigned pivColOffsetA = pivRC*dimA; unsigned pivColOffsetB = pivRC*dimB; unsigned pivColOffsetC = pivRC*dimC;
        // save the pivot column and row
        for (unsigned ind = GATO_THREAD_ID; ind < dimMax; ind++){
            if (AActive && ind < dimA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < dimB){s_memB[ind] = B[ind + pivColOffsetB];}
            if (CActive && ind < dimC){s_memC[ind] = C[ind + pivColOffsetC];}
        }
        for (unsigned ind = GATO_THREAD_ID; ind < dimMax+1; ind++){
            if (AActive && ind < dimA+1){s_memA[ind + dimA] = A[ind*dimA + pivRC + pivColOffsetA];}
            if (BActive && ind < dimB+1){s_memB[ind + dimB] = B[ind*dimB + pivRC + pivColOffsetB];}
            if (CActive && ind < dimC+1){s_memC[ind + dimC] = C[ind*dimC + pivRC + pivColOffsetC];}
        }
        g.sync();
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = GATO_THREAD_ID; ind < dimMax*(dimMax+1); ind += GATO_THREADS_PER_BLOCK){
            if (AActive && ind < dimA*(dimA+1)){
                unsigned row = ind % dimA; unsigned col = ind / dimA;
                if (row == pivRC){A[pivColOffsetA + ind] /= s_memA[pivRC];}
                else{A[pivColOffsetA + ind] -= s_memA[row]/s_memA[pivRC]*s_memA[dimA+col];}
            }
            if (BActive && ind < dimB*(dimB+1)){
                unsigned row = ind % dimB; unsigned col = ind / dimB; 
                if (row == pivRC){B[pivColOffsetB + ind] /= s_memB[pivRC];}
                else{B[pivColOffsetB + ind] -= s_memB[row]/s_memB[pivRC]*s_memB[dimB+col];}
            }
            if (CActive && ind < dimC*(dimC+1)){
                unsigned row = ind % dimC; unsigned col = ind / dimC;
                if (row == pivRC){C[pivColOffsetC + ind] /= s_memC[pivRC];}
                else{C[pivColOffsetC + ind] -= s_memC[row]/s_memC[pivRC]*s_memC[dimC+col];}
            }
        }
        g.sync();
    }
}
