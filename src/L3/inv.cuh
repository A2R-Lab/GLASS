#pragma once

#include <cstdint>
#include <cooperative_groups.h>
namespace cgrps = cooperative_groups;


template <typename T>
__device__
void invertMatrix(uint32_t dimA, T *A, T *s_temp, cgrps::thread_group g = cgrps::this_thread_block()){ 
// we are going to guassian elimination walking down the matrix (assuming no leading 0s)
// we therefore use the columns in order as the pivot column for each pivot we need to rescale 
// that row so that the pivot value (pv) is 1 THEN for all other row values (orv) we need to add a multiple 
// of the NEW pivot row value (prv) such that we transorm the other row pivot column value (orpcv) to 0
// pr *= 1/pv   orv -= orpcv*prv == orv -= orpcv*1/pv*prvOld
    for (unsigned pivRC = 0; pivRC < dimA; pivRC++){
        unsigned pivColOffset = pivRC*dimA;
        // save the pivot and pivot column and row
        T pvInv = static_cast<T>(1)/A[pivRC + pivColOffset];
        for (unsigned ind = g.thread_rank(); ind < 2*dimA+1; ind++){
            unsigned AInd;
            if (ind < dimA){AInd = ind + pivColOffset;}
            else{AInd = pivRC + pivColOffset + (ind-dimA)*dimA;}
            s_temp[ind] = A[AInd];
        }
        g.sync();
        // make the pivot update
        for (unsigned ind = g.thread_rank(); ind < dimA*(dimA+1); ind += g.size()){
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
void invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, T *s_temp, cgrps::thread_group g = cgrps::this_thread_block()){

    uint32_t dimMax = max(dimA, dimB);
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
        for (unsigned ind = g.thread_rank(); ind < dimMax; ind++){
            if (AActive && ind < dimA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < dimB){s_memB[ind] = B[ind + pivColOffsetB];}
        }
        for (unsigned ind = g.thread_rank(); ind < dimMax+1; ind++){
            if (AActive && ind < dimA+1){s_memA[ind + dimA] = A[ind*dimA + pivRC + pivColOffsetA];}
            if (BActive && ind < dimB+1){s_memB[ind + dimB] = B[ind*dimB + pivRC + pivColOffsetB];}
        }
        g.sync();
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = g.thread_rank(); ind < dimMax*(dimMax+1); ind += g.size()){
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
void invertMatrix(uint32_t dimA, T *A, uint32_t dimB, T *B, uint32_t dimC, T *C, T *s_temp, cgrps::thread_group g = cgrps::this_thread_block()){
    
    uint32_t dimMax = max(dimA, dimB);
    dimMax = max(dimMax, dimC);
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
        for (unsigned ind = g.thread_rank(); ind < dimMax; ind++){
            if (AActive && ind < dimA){s_memA[ind] = A[ind + pivColOffsetA];}
            if (BActive && ind < dimB){s_memB[ind] = B[ind + pivColOffsetB];}
            if (CActive && ind < dimC){s_memC[ind] = C[ind + pivColOffsetC];}
        }
        for (unsigned ind = g.thread_rank(); ind < dimMax+1; ind++){
            if (AActive && ind < dimA+1){s_memA[ind + dimA] = A[ind*dimA + pivRC + pivColOffsetA];}
            if (BActive && ind < dimB+1){s_memB[ind + dimB] = B[ind*dimB + pivRC + pivColOffsetB];}
            if (CActive && ind < dimC+1){s_memC[ind + dimC] = C[ind*dimC + pivRC + pivColOffsetC];}
        }
        g.sync();
        // make the pivot update with s_mem = [colA,rowA,colB,rowB,colC,rowC]
        for (unsigned ind = g.thread_rank(); ind < dimMax*(dimMax+1); ind += g.size()){
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

/*
Working but untested; generated code
*/
template <typename T>
__device__
void invertSubMatrixColumnMajor2(uint32_t dimA, uint32_t dimSub, T *A, T *s_temp){
    for (unsigned pivRC = 0; pivRC < dimSub; pivRC++) {
        // Compute the inverse of the pivot element correctly for column-major storage
        T pvInv = static_cast<T>(1) / A[pivRC + pivRC * dimA]; // Correct indexing for pivot in column-major
        
        // Load pivot row and column into shared memory with correct indexing for column-major
        for (unsigned ind = threadIdx.x; ind < 2*dimSub; ind += blockDim.x) {
            if (ind < dimSub) { // Loading pivot column
                s_temp[ind] = A[ind * dimA + pivRC]; // Correct for column-major
            } else { // Loading pivot row
                s_temp[ind] = A[pivRC * dimA + (ind - dimSub)]; // Correct for column-major
            }
        }
        __syncthreads();
        
        // Update matrix elements except for pivot row and column
        for (unsigned ind = threadIdx.x; ind < dimSub * dimSub; ind += blockDim.x) {
            unsigned row = ind / dimSub; // Correct calculation for column-major
            unsigned col = ind % dimSub; // Correct calculation for column-major
            
            if (row == pivRC || col == pivRC) continue; // Skip pivot row and column
            
            T element = A[row * dimA + col]; // Accessing element for column-major
            T pivotColElement = s_temp[row]; // Pivot column element for the current row
            T pivotRowElement = s_temp[dimSub + col]; // Pivot row element for the current column
            
            // Update the element using Gaussian elimination logic
            A[row * dimA + col] = element - pivotColElement * pivotRowElement * pvInv;
        }
        __syncthreads();
        
        // Update the pivot row for column-major, applying the inverse of the pivot value
        for (unsigned ind = threadIdx.x; ind < dimSub; ind += blockDim.x) {
            if (ind != pivRC) { // Exclude the pivot itself from this update
                A[pivRC * dimA + ind] = s_temp[dimSub + ind] * pvInv;
            }
        }
        __syncthreads();
        
        // Set the pivot element to its inverse in the matrix
        A[pivRC + pivRC * dimA] = pvInv;
        
        // Update pivot column elements except for the pivot itself
        for (unsigned ind = threadIdx.x; ind < dimSub; ind += blockDim.x) {
            if (ind != pivRC) { // Exclude the pivot itself
                A[ind * dimA + pivRC] = -s_temp[ind] * pvInv;
            }
        }
        __syncthreads();
    }
}
