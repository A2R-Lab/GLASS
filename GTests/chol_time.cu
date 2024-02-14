#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../glass.cuh"

template <typename T> 
__global__ 
void global_cholDecomp_InPlace (int n,
                        T *s_A, int reps)
{
	for (int i = 0; i < reps; i++) {
		glass::chol_InPlace<T>(n, s_A);
		__syncthreads();
	}
}

template <typename T> 
__global__ 
void global_cholDecomp_InPlace_vec (uint32_t n,
                        T *s_A, int reps)
{
    for (int i = 0; i < reps; i++) {
		glass::chol_InPlace_vec<T>(n, s_A);
		__syncthreads();
	}
}

int main(int argc, char *argv[]) {
	double total_time, avg_time, total_time_vec, avg_time_vec;
	struct timespec start, end;
	double *d_d;
	long i;

	long len = atol(argv[1]);
	long reps = atol(argv[2]);
	double *h_d =(double*)malloc(sizeof(double)*pow(len,2));
    double *res = (double*)malloc(sizeof(double)*pow(len,2));

	for (i = 0; i < pow(len,2); i++) {
		if (i % (len+1) == 0) h_d[i] = 1;
		else h_d[i] = 0;
	}

	cudaMalloc(&d_d, pow(len,2) * sizeof(double));
	cudaMemcpy(d_d, h_d, pow(len,2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// the glass implementation
	clock_gettime(CLOCK_MONOTONIC, &start);
	global_cholDecomp_InPlace<<<1,len>>>(len, d_d, reps);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);
	total_time = (double)end.tv_sec + (double)end.tv_nsec*1e-9
				- (double)start.tv_sec - (double)start.tv_nsec*1e-9;

	// seyoung's implementation
	clock_gettime(CLOCK_MONOTONIC, &start);
	global_cholDecomp_InPlace_vec<<<1,len>>>(len, d_d, reps);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end);
	total_time_vec = (double)end.tv_sec + (double)end.tv_nsec*1e-9
							- (double)start.tv_sec - (double)start.tv_nsec*1e-9;
	cudaFree(d_d);
	free(h_d);

	// print average time
	avg_time = total_time/reps;
	avg_time_vec= total_time_vec/reps;
	printf("avg time (glass):   %.10fs\n", avg_time);
	printf("avg time (seyoung): %.10fs\n", avg_time_vec);
	return 0;
}
