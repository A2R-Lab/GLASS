#include "L1/copy.cuh"
#include "L1/dot.cuh"
#include "L1/elementwise_logic.cuh"
#include "L1/norm.cuh"
#include "L3/gemm.cuh"
#include <cooperative_groups.h>
#include <cstdint>

namespace cgrps = cooperative_groups;

// all of these can be played with
#define TOL 1e-5
#define MAX_ITER 1000

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

// beware that this assumes q and x are 1D, which it is for DDP use case
template <typename T>
__device__ T objective_function(std::uint32_t dim, T *x, T *P, T *q, T *obj_tmp1, T *obj_tmp2, T *res, T *s_tmp,
                                cgrps::thread_group g = cgrps::this_thread_block())
{
    gemm_v2<T, true, false>(dim, 1, dim, dim, x, dim, P, dim, obj_tmp1, s_tmp);
    __syncthreads();
    dot<T>(res, dim, obj_tmp1, x, g);
    dot<T>(obj_tmp2, dim, q, x, g);
    __syncthreads();

    return 0.5 * res[0] + obj_tmp2[0];
}

template <typename T> __device__ void project(std::uint32_t dim, T *x, T *u, T *l, T *t1, T *t2)
{
    elementwise_min(dim, u, x, t1);
    __syncthreads();
    elementwise_max(dim, l, t1, t2);
    __syncthreads();
}

template <typename T>
__device__ T line_search_armijo(std::uint32_t dim, T *x, T *x_new, T *grad, T *P, T *q, T *l, T *u, T *s_tmp,
                                T *obj_tmp1, T *obj_tmp2, T *obj_res, T *dot_grad, T alpha = 1.0, T c = 1e-4,
                                T beta = 0.5)
{
    T f_xk = objective_function(dim, x, P, q, obj_tmp1, obj_tmp2, obj_res, s_tmp);
    dot(dot_grad, dim, grad, grad);
    for (int i = 0; i < MAX_ITER; i++)
    {
        // obj_tmp1 = alpha * grad
        elementwise_mult_scalar(dim, grad, alpha, obj_tmp1);
        __syncthreads();
        // obj_tmp2 = x - alpha * grad
        elementwise_sub(dim, x, obj_tmp1, obj_tmp2);
        __syncthreads();
        // x_new = project_onto_constraints(x - alpha * grad, l, u)
        project(dim, obj_tmp2, u, l, obj_tmp1, x_new);

        T f_xk_apk = objective_function(dim, x_new, P, q, obj_tmp1, obj_tmp2, obj_res, s_tmp);

        // printf("%f %f %e\n", f_xk_apk, f_xk, c * alpha * dot_grad[0]);
        if (f_xk_apk <= f_xk + c * alpha * dot_grad[0])
            break;

        alpha *= beta;
    }
    return alpha;
}

template <typename T>
__device__ bool cpqp(std::uint32_t dim, T *P, T *q, T *A, T *l, T *u, T *x, T *tmp1, T *res, T *tmp3, T *tmp4, T *tmp5,
                     T *tmp6, T *s_tmp, T *obj_tmp1, T *obj_tmp2, T *obj_res, T *x_new, T *dot_grad,
                     cgrps::thread_group g = cgrps::this_thread_block())

{
    T alpha;
    project(dim, x, u, l, tmp1, res);
    // ***x is res; up to here is fine

    for (int i = 0; i < MAX_ITER; i++)
    {
        // tmp3 is P @ x
        // grad = P @ x + q
        // printMatrixColumnMajor(res, 1, dim);
        gemm(dim, dim, 1, (T)1, P, res, tmp3);
        __syncthreads();
        matrixAlphaAdd((T)1, tmp3, q, tmp1, 1, dim);
        __syncthreads();
        // printMatrixColumnMajor(tmp1, 1, dim);

        alpha = line_search_armijo(dim, res, x_new, tmp1, P, q, l, u, s_tmp, obj_tmp1, obj_tmp2, obj_res, dot_grad);

        //  ****grad is tmp1
        //  x_c = x - alpha * grad
        elementwise_mult_scalar(dim, tmp1, alpha, tmp3);
        __syncthreads();
        elementwise_sub(dim, res, tmp3, tmp4); // x_c is tmp4
        __syncthreads();
        // printMatrixColumnMajor(tmp4, 1, dim);

        // x_c = project_onto_constraints(x_c, l, u)
        project(dim, tmp4, u, l, tmp3, tmp5); // x_c projected is tmp5; up to here is fine
        // printMatrixColumnMajor(tmp5, 1, dim);

        // Ax_c = A @ x_c
        gemm(dim, dim, 1, (T)1, A, tmp5, res);
        __syncthreads();
        // printMatrixColumnMajor(res, 1, dim);

        // np.all(l <= Ax_c) and np.all(Ax_c <= u) and np.linalg.norm(grad) <= tol
        elementwise_less_than_or_eq(dim, l, res, tmp3);
        elementwise_less_than_or_eq(dim, res, u, tmp4);
        reduce(dim, tmp3, g);
        reduce(dim, tmp4, g);
        vector_norm(dim, tmp1, tmp6);
        __syncthreads();

        if (tmp3[0] == dim && tmp4[0] == dim && tmp6[1] <= TOL)
        {
            // printf("iter: %d\n", i);
            return true;
        }

        copy(dim, tmp5, res);
        __syncthreads();
        // printf("----------------------------------------\n");
    }
    return false;
}