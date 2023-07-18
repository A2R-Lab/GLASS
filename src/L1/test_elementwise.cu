#include <iostream>
#include <cuda_runtime.h>
#include "elementwise_logic.cuh"  // Replace with the actual filename containing the function

template <typename T>
__global__
void elementwiseLessThanKernel(uint32_t N, T* a, T* b, T* c) {
    elementwise_less_than(N, a, b, c);
}

template <typename T>
void testElementwiseLessThan(uint32_t N, T* a, T* b) {
    T* d_a;
    T* d_b;
    T* d_c;

    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(T));
    cudaMalloc(&d_b, N * sizeof(T));
    cudaMalloc(&d_c, N * sizeof(T));

    // Copy input data from host to device
    cudaMemcpy(d_a, a, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(T), cudaMemcpyHostToDevice);

    // Call the device function
    elementwiseLessThanKernel<<<1, 256>>>(N, d_a, d_b, d_c);

    // Copy the result back to the host
    T* result = new T[N];
    cudaMemcpy(result, d_c, N * sizeof(T), cudaMemcpyDeviceToHost);

    // Print the output
    std::cout << "Output: ";
    for (uint32_t i = 0; i < N; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] result;
}

template <typename T>
__global__
void elementwiseAndKernel(uint32_t N, T* a, T* b, T* c) {
    elementwise_and(N, a, b, c);
}

template <typename T>
void testElementwiseAnd(uint32_t N, T* a, T* b) {
    T* d_a;
    T* d_b;
    T* d_c;

    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(T));
    cudaMalloc(&d_b, N * sizeof(T));
    cudaMalloc(&d_c, N * sizeof(T));

    // Copy input data from host to device
    cudaMemcpy(d_a, a, N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(T), cudaMemcpyHostToDevice);

    // Launch the kernel
    elementwiseAndKernel<<<1, 256>>>(N, d_a, d_b, d_c);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    T* result = new T[N];
    cudaMemcpy(result, d_c, N * sizeof(T), cudaMemcpyDeviceToHost);

    // Print the output
    std::cout << "Output: ";
    for (uint32_t i = 0; i < N; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] result;
}

template <typename T>
__global__
void elementwiseNotKernel(uint32_t N, T* a, T* c) {
    elementwise_not(N, a, c);
}

template <typename T>
void testElementwiseNot(uint32_t N, T* a) {
    T* d_a;
    T* d_c;

    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(T));
    cudaMalloc(&d_c, N * sizeof(T));

    // Copy input data from host to device
    cudaMemcpy(d_a, a, N * sizeof(T), cudaMemcpyHostToDevice);

    // Launch the kernel
    elementwiseNotKernel<<<1, 256>>>(N, d_a, d_c);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    T* result = new T[N];
    cudaMemcpy(result, d_c, N * sizeof(T), cudaMemcpyDeviceToHost);

    // Print the output
    std::cout << "Output: ";
    for (uint32_t i = 0; i < N; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaFree(d_a);
    cudaFree(d_c);
    delete[] result;
}

int main() {
    uint32_t N = 5;
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {3, 2, 1, 5, 4};

    std::cout << "Testing elementwise_less_than" << std::endl;
    std::cout << "Input A: ";
    for (uint32_t i = 0; i < N; ++i) {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Input B: ";
    for (uint32_t i = 0; i < N; ++i) {
        std::cout << b[i] << " ";
    }
    std::cout << std::endl;

    testElementwiseLessThan(N, a, b);

    uint32_t M = 5;
    int c[] = {1, 0, 1, 0, 1};
    int d[] = {1, 1, 0, 0, 1};

    // Test elementwise_and
    std::cout << "Testing elementwise_and:" << std::endl;
    std::cout << "Input A: ";
    for (uint32_t i = 0; i < M; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Input B: ";
    for (uint32_t i = 0; i < M; ++i) {
        std::cout << d[i] << " ";
    }
    std::cout << std::endl;
    testElementwiseAnd(M, c, d);

    // Test elementwise_not
    std::cout << "Testing elementwise_not:" << std::endl;
    std::cout << "Input A: ";
    for (uint32_t i = 0; i < M; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;
    testElementwiseNot(M, c);

    return 0;
}
