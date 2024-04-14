#include <cstdint>
#include <cooperative_groups.h>
#include "L1/elementwise_logic.cuh"
#include "L3/gemm.cuh"

namespace cgrps = cooperative_groups;

// all of these can be played with
#define TOL 1e-5
#define MAX_ITER 100

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
template <typename T> __device__ void project(std::uint32_t dim, T *x, T *u, T *l, T *t1, T *t2)
{
    elementwise_less_than(dim, x, u, t1);
    __syncthreads();
    elementwise_more_than(dim, t1, l, t2);
    __syncthreads();
}

template <typename T>
__device__
void printMatrixColumnMajor(T * matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%f ", matrix[j*rows + i]);
        }
        printf("\n");
    }
}

template <typename T>
__device__ void cpqp(std::uint32_t dim, T *P, T *q, T *A, T *l, T *u, T *res, T *x_0, T *tmp_element_1, T *tmp_element_2, T *tmp_grad, T alpha,
                     cgrps::thread_group g = cgrps::this_thread_block())

{
    printf("HELLOLLOO??\n");
    project(dim, x_0, u, l, tmp_element_1, tmp_element_2);

    // for (int i = 0; i < MAX_ITER; i++)
    // {
        printMatrixColumnMajor(P, dim, dim);
        printf("\n");
        printMatrixColumnMajor(A, dim, dim);
        gemm(dim, dim, dim, (float) 1, P, A, tmp_grad);
        __syncthreads();
        printMatrixColumnMajor(tmp_grad, dim, dim);
        __syncthreads();
        // gemm(dim, dim, dim, 1, P, A, tmp_grad);
    // }
}