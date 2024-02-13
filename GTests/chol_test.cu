#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../glass.cuh"

template <typename T> 
__global__ 
void global_cholDecomp_InPlace_c (uint32_t n,
                        T *s_A)
{
    glass::chol_InPlace<T>(n, s_A);
}

template <typename T> 
__global__ 
void global_cholDecomp_InPlace_vec (uint32_t n,
                        T *s_A)
{
    glass::chol_InPlace_vec<T>(n, s_A);
}

int main(int argc, char *argv[]) {
	double *d_a, *d_a_vec;
	double h_a[] = {10, 5, 2, 5, 3, 2, 2, 2, 3};
	double h_a_vec[] = {10, 5, 2, 5, 3, 2, 2, 2, 3};
	double res[] = {pow(10,0.5), 5/pow(10,0.5), 2/pow(10,0.5), 5, 
					1/pow(2,0.5), pow(2,0.5), 2, 2, pow(3,0.5)/pow(5,0.5)};
	int len = 3;

	cudaMalloc(&d_a, pow(len,2) * sizeof(double));
	cudaMalloc(&d_a_vec, pow(len,2) * sizeof(double));
	cudaMemcpy(d_a, h_a, pow(len,2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_a_vec, h_a_vec, pow(len,2) * sizeof(double), cudaMemcpyHostToDevice);

	global_cholDecomp_InPlace_c<<<1,len>>>(len, d_a);
	cudaDeviceSynchronize();
	global_cholDecomp_InPlace_vec<<<1,len>>>(len, d_a_vec);
	cudaDeviceSynchronize();
	cudaMemcpy(h_a, d_a, pow(len,2)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_a_vec, d_a_vec, pow(len,2)*sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < pow(len,2); i++) {
		printf("%f %f %f\n", res[i], h_a[i], h_a_vec[i]);
	}
	printf("\n");

	cudaFree(d_a);
	cudaFree(d_a_vec);
	return 0;
}
