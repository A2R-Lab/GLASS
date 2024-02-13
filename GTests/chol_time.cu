#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#include "../glass.cuh"

template <typename T> 
__global__ 
void global_cholDecomp_InPlace (uint32_t n,
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
	double time_taken, total_time, avg_time, total_time_vec, avg_time_vec;
	long len = atol(argv[1]);
	long reps = atol(argv[2]);
	struct timespec start, end;
	double *d_d;
	long i;

	double *h_d =(double*)malloc(sizeof(double)*pow(len,2));
    double *res = (double*)malloc(sizeof(double)*pow(len,2));
	cudaMalloc(&d_d, pow(len,2) * sizeof(double));

	for (i = 0; i < pow(len,2); i++) {
		if (i % (len+1) == 0) {
			h_d[i] = 1; res[i] = 1;
		}
		else {
			h_d[i] = 0; res[i] = 0;
		}
	}

	for (i = 0; i < reps; i++) {
		// the glass implementation
		cudaMemcpy(d_d, h_d, pow(len,2) * sizeof(double), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &start);
		global_cholDecomp_InPlace<<<1,pow(len,2)>>>(len, d_d);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &end);
		time_taken = (double)end.tv_sec + (double)end.tv_nsec*1e-9
                                - (double)start.tv_sec - (double)start.tv_nsec*1e-9;
		if (i >= reps/2) {
				total_time += time_taken;
		}

		// seyoung's implementation
		cudaMemcpy(d_d, h_d, pow(len,2) * sizeof(double), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &start);
		global_cholDecomp_InPlace_vec<<<1,pow(len,2)>>>(len, d_d);
		cudaDeviceSynchronize();
		clock_gettime(CLOCK_MONOTONIC, &end);
		time_taken = (double)end.tv_sec + (double)end.tv_nsec*1e-9
                                - (double)start.tv_sec - (double)start.tv_nsec*1e-9;
		if (i >= reps/2) {
				total_time_vec += time_taken;
		}
	}
	
	cudaFree(d_d);
	free(h_d); free(res);

	// print average time
	avg_time = total_time/reps*2;
	avg_time_vec= total_time_vec/reps*2;
	printf("avg time (glass):   %.10fs\n", avg_time);
	printf("avg time (seyoung): %.10fs\n", avg_time_vec);
	return 0;
}
