#include <iostream>
#include <cuda_runtime.h>
#include <gemm.cuh>

/*
    usage: 
    nvcc gemm_test.cu -o gemm_test
    ./gemm_test
*/

__global__ void testGemmAB(float alpha, float* A, float* B, float beta, float* C, uint32_t m, uint32_t n, uint32_t k) {
    extern __shared__ char shared[];
    float* s_temp = (float*)shared;
    gemm_v2<float, false, false>(m, n, n, k, A, m, B, n, C, s_temp);
    __syncthreads();
}

__global__ void testGemmTransposedA(float alpha, float* A, float* B, float beta, float* C, uint32_t m, uint32_t n, uint32_t k) {
    extern __shared__ char shared[];
    float* s_temp = (float*)shared;
    gemm_v2<float, true, false>(m, n, m, k, A, m, B, m, C, s_temp);
    __syncthreads();
}

__global__ void testGemmTransposedB(float alpha, float* A, float* B, float beta, float* C, uint32_t m, uint32_t n, uint32_t k) {
    extern __shared__ char shared[];
    float* s_temp = (float*)shared;
    gemm_v2<float, false, true>(m, n, k, n, A, m, B, m, C, s_temp);
    __syncthreads();
}

void printMatrix(float* matrix, uint32_t m, uint32_t n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", matrix[j * m + i]);
        }
        printf("\n");
    }
}

void testGemm3x3() {
    /*
        Test the gemm function with 3x3 matrices
    */
    printf("Running testGemm3x3\n");
    const std::uint32_t m = 3;
    const std::uint32_t n = 3;
    const std::uint32_t k = 3;
    float alpha = 1.0f;
    float beta = 1.0f;
    float A[m * n] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    float B[n * k] = {5, 4, 6, 2, 4, 3, 1, 1, 1};
    // print matrices A and B, which are stored in column major order
    // but print them out in row major order for easier reading
    printf("Matrix A:\n");
    printMatrix(A, m, n);
    printf("Matrix B:\n");
    printMatrix(B, n, k);

    // Allocate memory on the GPU and copy the matrices over
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * k * sizeof(float));
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
    
    float C_init[m * k] = {0}; // to initialize result matrix
    float C[m * k] = {0}; // result matrix
    cudaMalloc(&d_C, m * k * sizeof(float));

    // start test 1 - A * B
    float C_expected[m * k] = {16, 61, 106, 10, 37, 64, 3, 12, 21};

    cudaMemcpy(d_C, C_init, m * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16); // Choose suitable block size
    dim3 gridSize(1); // small test, only need 1 block
    size_t sharedMemSize = blockSize.x * sizeof(float); // each thread needs one float of shared mem to accumulate its dot product result
    testGemmAB<<<gridSize, blockSize, sharedMemSize>>>(alpha, d_A, d_B, beta, d_C, m, n, k);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "A * B results in matrix C:" << std::endl;
    printMatrix(C, m, k);

    // Compare the result with the expected result
    bool passed = true;
    for (int i = 0; i < m * k; ++i) {
        if (C[i] != C_expected[i]) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Test 1 passed!" << std::endl;
    } else {
        std::cout << "Test 1 failed!" << std::endl;
    }
    // end test 1

    // start test 2 - A^T * B
    float C_expected_transposedA[n * k] = {48, 63, 78, 30, 39, 48, 9, 12, 15};

    cudaMemcpy(d_C, C_init, m * k * sizeof(float), cudaMemcpyHostToDevice);

    testGemmTransposedA<<<gridSize, blockSize, sharedMemSize>>>(alpha, d_A, d_B, beta, d_C, m, n, k);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "A^T * B results in matrix C:" << std::endl;
    printMatrix(C, n, k);

    // Compare the result with the expected result
    passed = true;
    for (int i = 0; i < m * k; ++i) {
        if (C[i] != C_expected_transposedA[i]) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Test 2 passed!" << std::endl;
    } else {
        std::cout << "Test 2 failed!" << std::endl;
    }
    // end test 2

    // start test 3 - A * B^T
    float C_expected_transposedB[m * n] = {4, 28, 52, 6, 33, 60, 5, 35, 65};

    cudaMemcpy(d_C, C_init, m * k * sizeof(float), cudaMemcpyHostToDevice);

    testGemmTransposedB<<<gridSize, blockSize, sharedMemSize>>>(alpha, d_A, d_B, beta, d_C, m, n, k);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "A * B^T results in matrix C:" << std::endl;
    printMatrix(C, m, n);

    // Compare the result with the expected result
    passed = true;
    for (int i = 0; i < m * k; ++i) {
        if (C[i] != C_expected_transposedB[i]) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Test 3 passed!" << std::endl;
    } else {
        std::cout << "Test 3 failed!" << std::endl;
    }
    // end test 3
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Finished testGemm3x3\n\n");
}

void testGemm3x2_3x3() {
    /*
        Test the gemm function with Matrix A being 3x2 and Matrix B being 3x3,
        so that matrix A must be transposed for a valid matmul
    */
    printf("Running testGemm3x2_3x3\n");
    const std::uint32_t m = 3;
    const std::uint32_t n = 2;
    const std::uint32_t k = 3;
    float alpha = 1.0f;
    float beta = 1.0f;
    float A[m * n] = {0, 2, 4, 1, 3, 5};
    float B[m * k] = {5, 4, 6, 2, 4, 3, 1, 1, 1};
    // print matrices A and B, which are stored in column major order
    // but print them out in row major order for easier reading
    printf("Matrix A:\n");
    printMatrix(A, m, n);
    printf("Matrix B:\n");
    printMatrix(B, m, k);

    // Allocate memory on the GPU and copy the matrices over
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, m * k * sizeof(float));
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);

    float C_init[n * k] = {0}; // to initialize result matrix
    float C[n * k] = {0}; // result matrix
    cudaMalloc(&d_C, n * k * sizeof(float));

    // start test 1 - A^T * B
    float C_expected_transposedA[n * k] = {32, 47, 20, 29, 6, 9};

    cudaMemcpy(d_C, C_init, n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16); // Choose suitable block size
    dim3 gridSize(1); // small test, only need 1 block
    size_t sharedMemSize = blockSize.x * sizeof(float); // each thread needs one float of shared mem to accumulate its dot product result
    testGemmTransposedA<<<gridSize, blockSize, sharedMemSize>>>(alpha, d_A, d_B, beta, d_C, m, n, k);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(C, d_C, n * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "A^T * B results in matrix C:" << std::endl;
    printMatrix(C, n, k);

    // Compare the result with the expected result
    bool passed = true;
    for (int i = 0; i < n * k; ++i) {
        if (C[i] != C_expected_transposedA[i]) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Test 1 passed!" << std::endl;
    } else {
        std::cout << "Test 1 failed!" << std::endl;
    }
    // end test 1

    printf("Finished testGemm3x2_3x3\n\n");
}

void testGemm3x2_3x2() {
    /*
        Test the gemm function with Matrix A being 3x2 and Matrix B being 3x2,
        and we will transpose matrix B
    */
    printf("Running testGemm3x2_3x2\n");
    const std::uint32_t m = 3;
    const std::uint32_t n = 2;
    const std::uint32_t k = 3;
    float alpha = 1.0f;
    float beta = 1.0f;
    float A[m * n] = {0, 2, 4, 1, 3, 5};
    float B[m * k] = {5, 4, 6, 2, 4, 3};
    // print matrices A and B, which are stored in column major order
    // but print them out in row major order for easier reading
    printf("Matrix A:\n");
    printMatrix(A, m, n);
    printf("Matrix B:\n");
    printMatrix(B, k, n);

    // Allocate memory on the GPU and copy the matrices over
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, m * k * sizeof(float));
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, m * k * sizeof(float), cudaMemcpyHostToDevice);

    float C_init[m * k] = {0}; // to initialize result matrix
    float C[m * k] = {0}; // result matrix
    cudaMalloc(&d_C, m * k * sizeof(float));

    // start test 1 - A^T * B
    float C_expected_transposedA[m * k] = {2, 16, 30, 4, 20, 36, 3, 21, 39};

    cudaMemcpy(d_C, C_init, m * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16); // Choose suitable block size
    dim3 gridSize(1); // small test, only need 1 block
    size_t sharedMemSize = blockSize.x * sizeof(float); // each thread needs one float of shared mem to accumulate its dot product result
    testGemmTransposedB<<<gridSize, blockSize, sharedMemSize>>>(alpha, d_A, d_B, beta, d_C, m, n, k);
    cudaDeviceSynchronize();

    // Copy back the result
    cudaMemcpy(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "A * B^T results in matrix C:" << std::endl;
    printMatrix(C, m, k);

    // Compare the result with the expected result
    bool passed = true;
    for (int i = 0; i < m * k; ++i) {
        if (C[i] != C_expected_transposedA[i]) {
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << "Test 1 passed!" << std::endl;
    } else {
        std::cout << "Test 1 failed!" << std::endl;
    }
    // end test 1

    printf("Finished testGemm3x2_3x2\n\n");
}

int main() {
    testGemm3x3();
    testGemm3x2_3x3();
    testGemm3x2_3x2();

    return 0;
}
