#pragma once



/*
    *  copy
    *  =====
    *
    *  Copies the contents of x into y.
    *
    *  Parameters
    *  ----------
    *  n : uint32_t
    *      The number of elements to copy.
    *  x : T*
    *      The array to copy from.
    *  y : T*
    *      The array to copy to.
    *  g : cgrps::thread_group
    *      The thread group to use.
    */
template <typename T>
__device__
void copy(const uint32_t n, 
          T *x, 
          T *y)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    for(; ind < n; ind += stride){
        y[ind] = x[ind];
    }
}

/*
    * copy
    * ====
    * 
    *  Copies the scaled contents of x into y.
    *  Scale each element in x by alpha
    * 
    * Parameters
    * ----------
    * n : uint32_t
    *    The number of elements to copy.
    * alpha : T
    *   The scaling factor
    * x : T*
    *  The array to copy from.
    * y : T*
    * The array to copy to.
    * g : cgrps::thread_group
    * The thread group to use.
*/
template <typename T>
__device__
void copy(uint32_t n,
          T alpha,
          T *x, 
          T *y)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind < n; ind += stride){
        y[ind] = alpha * x[ind];
    }
}