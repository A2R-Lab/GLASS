#include "L1/copy.cuh"
#include "L1/elementwise_logic.cuh"
#include "L1/norm.cuh"
#include "L3/gemm.cuh"
#include <cooperative_groups.h>
#include <cstdint>

namespace cgrps = cooperative_groups;

// all of these can be played with
#define TOL 1e-5
#define MAX_ITER 2

/*
All inputs are assumed to be in column major order
Parameters can be made more general in terms of dimensions in the future.
Decrease alpha if doesn't converge

Assumes P is dim * dim
q is dim
A is dim
l is dim
u is dim
tmps are all dim
res is dim


See test_cpqp.cu for test usage

For use in forward pass of box constraints
*/


template <typename T> __device__ void printMatrixColumnMajor(T *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[j * rows + i]);
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T> __device__ void project(std::uint32_t dim, T *x, T *u, T *l, T *t1, T *t2)
{
    elementwise_min(dim, u, x, t1);
    __syncthreads();
    elementwise_max(dim, l, t1, t2);
    __syncthreads();
}

template <typename T>
__device__ bool cpqp(std::uint32_t dim, T *P, T *q, T *A, T *l, T *u, T *res, T *x, T *tmp1, T *tmp2, T *tmp3, T *tmp4,
                     T *tmp5, T *tmp6, T alpha, cgrps::thread_group g = cgrps::this_thread_block())

{
    printMatrixColumnMajor(l, 1, dim);
    printMatrixColumnMajor(u, 1, dim);
    project(dim, x, u, l, tmp1, tmp2);
    printMatrixColumnMajor(tmp2, 1, dim);
    // ***x is tmp2; up to here is fine

#pragma unroll
    for (int i = 0; i < MAX_ITER; i++)
    {
        // tmp3 is P @ x
        printMatrixColumnMajor(tmp2, 1, dim);
        gemm(dim, dim, 1, (T)1, P, tmp2, tmp3);
        __syncthreads();
        printMatrixColumnMajor(tmp3, 1, dim);
        __syncthreads();
        matrixAlphaAdd((T)1, tmp3, q, tmp1, 1, dim);
        __syncthreads();
        //  // ****grad is tmp1
        //  // can use tmp3
        printMatrixColumnMajor(tmp1, 1, dim); // up to here is fine
        elementwise_mult_scalar(dim, tmp1, alpha, tmp3);
        __syncthreads();
        elementwise_sub(dim, tmp2, tmp3, tmp4); // x_c is tmp4
        __syncthreads();
        project(dim, tmp4, u, l, tmp3, tmp5); // x_c projected is tmp5; up to here is fine
        __syncthreads();
        printMatrixColumnMajor(tmp5, 1, dim);
        // // result is in tmp5
        // // i still need tmp1, tmp5,
        gemm(dim, dim, 1, (T)1, A, tmp5, tmp2);
        __syncthreads();
        printMatrixColumnMajor(tmp2, 1, dim); // Ax_C
        printf("\n");
        __syncthreads();
        elementwise_less_than_or_eq(dim, l, tmp2, tmp3);
        elementwise_less_than_or_eq(dim, tmp2, u, tmp4);
        reduce(dim, tmp3, g);
        reduce(dim, tmp4, g);
        vector_norm(dim, tmp1, tmp6);
        __syncthreads();
        printMatrixColumnMajor(tmp3, 1, dim);
        printMatrixColumnMajor(tmp4, 1, dim);
        printMatrixColumnMajor(tmp6, 1, dim);
        if (tmp3[0] == dim && tmp4[0] == dim && tmp6[1] <= TOL)
            return true;
        copy(dim, tmp5, tmp2);
        __syncthreads();
    }

    return false;
}