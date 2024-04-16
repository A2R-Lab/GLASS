#include <cstdint>
#include <cooperative_groups.h>
#include "L1/elementwise_logic.cuh"
#include "L3/gemm.cuh"
#include "L1/norm.cuh"
// #include "L1/reduce.cuh"

namespace cgrps = cooperative_groups;

// all of these can be played with
#define TOL 1e-5
#define MAX_ITER 10

/*
All inputs are assumed to be in column major order
Parameters can be made more general in terms of dimensions in the future.
Decrease alpha if doesn't converge

Assumes P is dim * dim 
q is dim
A is dim
l is dim
u is dim
tmp is dim
res is dim


See test_cpqp.cu for test usage

For use in forward pass of box constraints

def qp_gradient_projection(P: np.array, q: np.array, A: np.array, l: np.array, u: np.array,  alpha, x0=0, tol=1e-5, max_iter=100):
    x = project_onto_constraints(x0, l, u)
    # tmp1
    
    for k in range(max_iter):
        grad = P @ x + q
        #tmp2
        x_c = x - alpha * grad
        #tmp3
        x_c = project_onto_constraints(x_c, l, u)
        #tmp1

        Ax_c = A @ x_c
        #tmp2
        # check constraints + feasibility
        if np.all(l <= Ax_c) and np.all(Ax_c <= u) and np.linalg.norm(grad) <= tol:
            print(f"Converged to a feasible solution in {k} iterations.")
            return x_c, True
        
        x = x_c
    
    print("Maximum number of iterations reached without convergence.")
    return x, False
*/

/*
#Project x onto the feasible region defined by l and u
def project_onto_constraints(x, l, u):
    return np.maximum(l, np.minimum(u, x))

template <typename T>
__device__
void elementwise_max_scalar(uint32_t N, T* a, T b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = max(a[i], b);
    }
}

template <typename T>
__device__
void elementwise_min_scalar(uint32_t N, T* a, T b, T* c,
                cgrps::thread_group g = cgrps::this_thread_block()) {
    for (int i = g.thread_rank(); i < N; i += g.size()) {
        c[i] = min(a[i], b);
    }
}
*/

template <typename T>
__device__
void printMatrixColumnMajor(T * matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%f ", matrix[j*rows + i]);
        }
        printf("\n");
    }
    printf("\n");
}

template <typename T> __device__ void project(std::uint32_t dim, T *x, T *u, T *l, T *t1, T *t2)
{
    elementwise_min(dim, u, x, t1);
    __syncthreads();
    // printMatrixColumnMajor(t1, dim, 1);
    elementwise_max(dim, l, t1, t2);
    // printMatrixColumnMajor(t2, dim, 1);
    __syncthreads();
}

// template <typename T>
// __device__
// void printMatrixColumnMajor(T * matrix, int rows, int cols) {
//     for(int i=0; i<rows; i++) {
//         for(int j=0; j<cols; j++) {
//             printf("%f ", matrix[j*rows + i]);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }

template <typename T>
__device__ void cpqp(std::uint32_t dim, T *P, T *q, T *A, T *l, T *u, T *res, T *x, T *tmp1, T *tmp2, T *tmp3,
                     T *tmp4, T *tmp5, T *tmp6, T alpha, cgrps::thread_group g = cgrps::this_thread_block())

{
    printMatrixColumnMajor(x, 1, dim);
    printMatrixColumnMajor(l, 1, dim);
    printMatrixColumnMajor(u, 1, dim);
    project(dim, x, u, l, tmp1, tmp2);
    printMatrixColumnMajor(tmp2, 1, dim);
    // ***x is tmp2

    // #pragma unroll
    // for (int i = 0; i < MAX_ITER; i++)
    // {
        // tmp3 is P @ x
        gemm(dim, dim, 1, (T) 1, P, tmp2, tmp3);
        __syncthreads();
        printMatrixColumnMajor(tmp3, 1, dim);
         __syncthreads();
        matrixAlphaAdd((T) 1, tmp3, q, tmp1, 1, dim);
         __syncthreads();
         // ****grad is tmp1
         // can use tmp3
        printMatrixColumnMajor(tmp1, 1, dim);
        elementwise_mult_scalar(dim, tmp1, alpha, tmp3);
         __syncthreads();
        elementwise_sub(dim, tmp2, tmp3, tmp4);//x_c is tmp4
         __syncthreads();
        project(dim, tmp4, u, l, tmp3, tmp5); //x_c projected is tmp5
         __syncthreads();
        printMatrixColumnMajor(tmp5, 1, dim);
        // result is in tmp5
        // i still need tmp1, tmp5
        gemm(dim, dim, 1, (T) 1, A, tmp5, tmp2);
         __syncthreads();
        printMatrixColumnMajor(tmp2, 1, dim); // Ax_C
        printf("\n");
        __syncthreads();
        elementwise_less_than_or_eq(dim, l, tmp2, tmp3);
        elementwise_less_than_or_eq(dim, tmp2, u, tmp4);
        reduce(dim, tmp3, g);
        reduce(dim, tmp4, g);
        // normalize here 
        vector_norm(dim, tmp1, tmp6);
        __syncthreads();
        printMatrixColumnMajor(tmp3, 1, dim); 
        printMatrixColumnMajor(tmp4, 1, dim); 
        printMatrixColumnMajor(tmp6, 1, dim); 
        if (tmp3[0] == dim && tmp4[0] == dim && tmp1[1] <= TOL)
            printf("hello");

        // __syncthreads();
        // gemm(dim, dim, dim, 1, P, A, tmp_grad);
    // }
}