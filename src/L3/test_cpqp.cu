#include <iostream>
#include <cuda_runtime.h>
#include <cpqp.cuh>

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

__global__ void test_cpqp(std::uint32_t dim, float *P, float *q, float *A, float *lb, float *ub, float *tmp1,
                          float *tmp2, float *tmp3, float *x_0, float *res, float alpha = 0.9)
{
    cpqp<float>(dim, P, q, A, lb, ub, res, x_0, tmp1, tmp2, tmp3, alpha);
    __syncthreads();
}

void cpqp_test_1() {
    std::uint32_t num_control_dims = 7;

    float P[num_control_dims *num_control_dims] = {12.6191, 0.0136,  -0.3856, 0.1301,  0.0613,  0.0454,  0.0173,  0.0136,  12.5569, 0.0865,
                 0.1415,  -0.0619, 0.0701,  -0.004,  -0.3856, 0.0865,  14.13,   -0.1305, -0.6172, -0.1599,
                 -0.0509, 0.1301,  0.1415,  -0.1305, 12.9557, -0.0594, 0.3747,  0.0042,  0.0613,  -0.0619,
                 -0.6172, -0.0594, 15.9793, 0.0845,  -0.1174, 0.0454,  0.0701,  -0.1599, 0.3747,  0.0845,
                 15.9542, -0.1119, 0.0173,  -0.004,  -0.0509, 0.0042,  -0.1174, -0.1119, 16.5841};

    float q[num_control_dims] = {-0.3501, 0.0418, 1.1519, -0.2435, 0.0982, -0.5519, 0.3218};

    float A[num_control_dims * num_control_dims] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

    float lb[num_control_dims] = {-4.9239, -4.9239, -4.9239, -4.9239, -4.9239, -4.9239, -4.9239};
    float ub[num_control_dims] = {4.9239, 4.9239, 4.9239, 4.9239, 4.9239, 4.9239, 4.9239};
    float x_0[num_control_dims] = {0, 0, 0, 0, 0, 0, 0};

    float *d_P, *d_q, *d_A, *d_lb, *d_ub, *d_tmp1, *d_tmp2, *d_tmp_grad_1, *d_x_0, *d_res;
    cudaMalloc(&d_P, num_control_dims * num_control_dims * sizeof(float));
    cudaMalloc(&d_q, num_control_dims * sizeof(float));
    cudaMalloc(&d_A, num_control_dims * num_control_dims * sizeof(float));
    cudaMalloc(&d_lb, num_control_dims * sizeof(float));
    cudaMalloc(&d_ub, num_control_dims * sizeof(float));
    cudaMalloc(&d_tmp1, num_control_dims * sizeof(float));
    cudaMalloc(&d_tmp2, num_control_dims * sizeof(float));
    cudaMalloc(&d_tmp_grad_1, num_control_dims * num_control_dims * sizeof(float));
    cudaMalloc(&d_x_0, num_control_dims * sizeof(float));
    cudaMalloc(&d_res, num_control_dims * sizeof(float));

    cudaMemcpy(d_P, P, num_control_dims * num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, num_control_dims * num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lb, lb, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub, ub, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_0, x_0, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_res, num_control_dims * sizeof(float));
    // cudaMemcpy(d_tmp1, ub, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_tmp2, ub, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_tmp3, ub, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);

    // float res[num_control_dims] = {0}; 
    // float *d_res;
    // cudaMalloc(&d_res, num_control_dims * sizeof(float));
    // cudaMemcpy(d_res, res, num_control_dims * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockSize(1); // Choose suitable block size
    dim3 gridSize(1); // small test, only need 1 block

    test_cpqp<<<gridSize, blockSize>>>(num_control_dims, d_P, d_q, d_A, d_lb, d_ub, d_tmp1, d_tmp2, d_tmp_grad_1, d_x_0,
                                       d_res);
    cudaDeviceSynchronize();
}

int main() {
    cpqp_test_1() ;
    return 0;
}