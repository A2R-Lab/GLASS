#pragma once
/*
    Compute the scaled sum of two vectors
    alpha * x + y
    store the result back in y
*/
template <typename T>
__device__
void axpy(uint32_t n, 
          T alpha, 
          T *x, 
          T *y)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind<n; ind+=stride){
        y[ind] = alpha * x[ind] + y[ind];
    }
}

/*
    Compute the scaled sum of two vectors
    alpha * x + y
    store the result in z
*/
template <typename T>
__device__
void axpy(uint32_t n, 
          T alpha, 
          T *x, 
          T *y, 
          T *z)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind<n; ind+=stride){
        z[ind] = alpha * x[ind] + y[ind];
    }
}

/*
    Compute the scaled sum of two vectors
    alpha * x + beta * y
    store the result in z
*/
template <typename T>
__device__
void axpby(uint32_t n, 
          T alpha, 
          T *x,
		  T beta, 
          T *y, 
          T *z)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind<n; ind+=stride){
        z[ind] = alpha * x[ind] + beta * y[ind];
    }
}