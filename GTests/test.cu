#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <cooperative_groups.h>
#include "../glass.cuh"
#include "./global_glass.cuh"
#include "gtest/gtest.h"

__host__
void printArray(int n, int * arr){
	std::cout<<"{ " << arr[0]; 
	for(int i = 1; i < n; i++){
		std::cout<<", "<<arr[i];
	}
	std::cout<<"}"<<std::endl;
}

class DeviceTest : public ::testing::Test{

        protected:
                void SetUp() override {
                        n = 10;
                        h_a = new int[n];
                        h_b = new int[n];
			h_c = new int;
                        for(int i = 0; i < n; i++){
                                h_a[i] = i;
                                h_b[i] = 2 * i;
                        }
			cudaMalloc(&d_a, n * sizeof(int));
    			cudaMalloc(&d_b, n * sizeof(int));
    			cudaMalloc(&d_c, sizeof(int));
                	cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    			cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
    			cudaDeviceSynchronize();
		}
		void TearDown() override {
     			// Code here will be called immediately after each test (right
     			// before the destructor).
  			cudaFree(d_a);
			cudaFree(d_b);
			cudaFree(d_c);
			delete h_a;
			delete h_b;
			delete h_c;
		}

        int n;
        int * h_a;
        int * h_b;
	int * h_c;
	int * d_a, *d_b, *d_c;
};

TEST_F(DeviceTest, DotProduct){
	global_dot<<<1, n>>>(d_c, n, d_a, d_b);
	cudaDeviceSynchronize();

    	// copy the memory back
    	cudaMemcpy(h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    	cudaDeviceSynchronize();

	EXPECT_EQ(*h_c, 570);
	
}

int main(){
        ::testing::InitGoogleTest();
        return RUN_ALL_TESTS();
}




