#pragma once

template <typename T>
__device__
void scal(uint32_t n, 
          T alpha, 
          T *x)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind < n; ind += stride){
        x[ind] = alpha * x[ind];
    }
}
template <typename T>
__device__
void scal(std::uint32_t n, 
          T alpha, 
          T *x,
          T *y,
          cgrps::thread_group g = cgrps::this_thread_block())
{
    for(std::uint32_t ind = g.thread_rank(); ind < n; ind += g.size()){
        y[ind] = alpha * x[ind];
    }
}