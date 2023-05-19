#pragma once
/*
    Clip z between l and u
*/
template <typename T>
__device__
void clip(uint32_t n, 
          T *x,
          T *l, 
          T *u)
{
    uint32_t ind = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;

    for(; ind<n; ind+=stride){
		if ( x[ind]  < l[ind]){
			x[ind] = l[ind];
		}
		else if (x[ind] > u[ind]) {
			x[ind] = u[ind];
		}
    }
}