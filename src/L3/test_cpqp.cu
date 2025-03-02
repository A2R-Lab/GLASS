#include <cpqp.cuh>
#include <cuda_runtime.h>
#include <iostream>

#define FORWARDPASS_THREADS 32


/*
This is a real QP that we solve in forward pass

P_test
array([[12.6191,  0.0136, -0.3856,  0.1301,  0.0613,  0.0454,  0.0173],
       [ 0.0136, 12.5569,  0.0865,  0.1415, -0.0619,  0.0701, -0.004 ],
       [-0.3856,  0.0865, 14.13  , -0.1305, -0.6172, -0.1599, -0.0509],
       [ 0.1301,  0.1415, -0.1305, 12.9557, -0.0594,  0.3747,  0.0042],
       [ 0.0613, -0.0619, -0.6172, -0.0594, 15.9793,  0.0845, -0.1174],
       [ 0.0454,  0.0701, -0.1599,  0.3747,  0.0845, 15.9542, -0.1119],
       [ 0.0173, -0.004 , -0.0509,  0.0042, -0.1174, -0.1119, 16.5841]])

q
array([-0.3501,  0.0418,  1.1519, -0.2435,  0.0982, -0.5519,  0.3218])

A_
array([[1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 1.]])

lb
array([-4.9239, -4.9239, -4.9239, -4.9239, -4.9239, -4.9239, -4.9239])

ub
array([4.9239, 4.9239, 4.9239, 4.9239, 4.9239, 4.9239, 4.9239])
*/

template <typename T>
__global__ void test_cpqp(std::uint32_t dim, T *P, T *q, T *A, T *lb, T *ub, T *tmp1, T *res, T *tmp3, T *tmp4, T *tmp5,
                          T *tmp6, T *x_0, T *s_tmp, T *obj_tmp1, T *obj_tmp2, T *obj_res, T *x_new, T *dot_grad)
{
    cpqp<T>(dim, P, q, A, lb, ub, x_0, tmp1, res, tmp3, tmp4, tmp5, tmp6, s_tmp, obj_tmp1, obj_tmp2, obj_res, x_new,
            dot_grad);
    __syncthreads();
}

void cpqp_test_1()
{
    std::uint32_t num_control_dims = 7;

    double P[num_control_dims * num_control_dims] = {
        13.022784015920074, 0.1602830744821,     -0.975087407086252, 0.719470814817327,  0.269016915363258,
        0.174968209396238,  0.007070640863051,   0.160283074489815,  12.942923517222942, 0.312299783306563,
        0.965498329766035,  -0.231794957423362,  0.14404541886428,   0.270994416522598,  -0.975087407036581,
        0.312299783326669,  17.424384018629073,  -0.245860105289312, -3.189800608997873, -0.889226479786205,
        2.516164062142545,  0.719470814835794,   0.965498329758143,  -0.245860105387552, 15.562910964201624,
        -0.108626275482834, 1.080643826931862,   0.518484978850113,  0.269016914920422,  -0.231794957300408,
        -3.189800607052198, -0.108626275633404,  32.41501689914927,  1.365188996242705,  -24.624232216732917,
        0.17496820944846,   0.144045418847257,   -0.889226480018947, 1.080643826944995,  1.365188996237579,
        24.545707872269453, -0.918225780237453,  0.0070706420625,    0.270994415507298,  2.516164052600129,
        0.51848497757645,   -24.624232178925777, -0.918225785793746, 100.23596981903623};

    double q[num_control_dims] = {-4.959635017022945, 1.501537023540699,  1.465996601597597, -2.950586284512987,
                                  1.0058699524367,    -1.770490656955261, 5.591515635147498};

    double A[num_control_dims * num_control_dims] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                     0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

    double lb[num_control_dims] = {-45, -45, -45, -45, -45, -45, -45};
    double ub[num_control_dims] = {45, 45, 45, 45, 45, 45, 45};
    double x_0[num_control_dims] = {0, 0, 0, 0, 0, 0, 0};

    double *d_P, *d_q, *d_A, *d_lb, *d_ub, *d_tmp1, *d_res, *d_tmp3, *d_tmp4, *d_tmp5, *d_tmp6, *d_x_0;
    double *d_s_tmp, *d_obj_tmp1, *d_obj_tmp2, *d_obj_res;
    double *d_x_new, *d_dot_grad;

    cudaMalloc(&d_P, num_control_dims * num_control_dims * sizeof(double));
    cudaMalloc(&d_q, num_control_dims * sizeof(double));
    cudaMalloc(&d_A, num_control_dims * num_control_dims * sizeof(double));
    cudaMalloc(&d_lb, num_control_dims * sizeof(double));
    cudaMalloc(&d_ub, num_control_dims * sizeof(double));
    cudaMalloc(&d_tmp1, num_control_dims * sizeof(double));
    cudaMalloc(&d_res, num_control_dims * sizeof(double));
    cudaMalloc(&d_tmp3, num_control_dims * sizeof(double));
    cudaMalloc(&d_tmp4, num_control_dims * sizeof(double));
    cudaMalloc(&d_tmp5, num_control_dims * sizeof(double));
    cudaMalloc(&d_tmp6, num_control_dims * sizeof(double));
    cudaMalloc(&d_x_0, num_control_dims * sizeof(double));
    cudaMalloc(&d_res, num_control_dims * sizeof(double));
    cudaMalloc(&d_s_tmp, FORWARDPASS_THREADS * sizeof(double));
    cudaMalloc(&d_obj_tmp1, num_control_dims * sizeof(double));
    cudaMalloc(&d_obj_tmp2, num_control_dims * sizeof(double));
    cudaMalloc(&d_obj_res, num_control_dims * sizeof(double));
    cudaMalloc(&d_x_new, num_control_dims * sizeof(double));
    cudaMalloc(&d_dot_grad, num_control_dims * sizeof(double));

    cudaMemcpy(d_P, P, num_control_dims * num_control_dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, num_control_dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, num_control_dims * num_control_dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lb, lb, num_control_dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub, ub, num_control_dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_0, x_0, num_control_dims * sizeof(double), cudaMemcpyHostToDevice);

    dim3 blockSize(1);
    dim3 gridSize(1);

    test_cpqp<<<gridSize, blockSize, FORWARDPASS_THREADS>>>(num_control_dims, d_P, d_q, d_A, d_lb, d_ub, d_tmp1, d_res,
                                                            d_tmp3, d_tmp4, d_tmp5, d_tmp6, d_x_0, d_s_tmp, d_obj_tmp1,
                                                            d_obj_tmp2, d_obj_res, d_x_new, d_dot_grad);
    cudaDeviceSynchronize();

    double h_res[num_control_dims];
    cudaMemcpy(h_res, d_res, num_control_dims * sizeof(double), cudaMemcpyDeviceToHost);
    for (std::uint32_t i = 0; i < num_control_dims; i++)
        printf("%f ", h_res[i]);
    printf("\n");

    cudaFree(d_P);
    cudaFree(d_q);
    cudaFree(d_A);
    cudaFree(d_lb);
    cudaFree(d_ub);
    cudaFree(d_tmp1);
    cudaFree(d_res);
    cudaFree(d_tmp3);
    cudaFree(d_tmp4);
    cudaFree(d_tmp5);
    cudaFree(d_tmp6);
    cudaFree(d_x_0);
}

int main()
{
    cpqp_test_1();
    return 0;
}