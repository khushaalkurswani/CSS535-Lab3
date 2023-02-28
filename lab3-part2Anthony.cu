#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "cublas_v2.h"

using namespace std;


#define N 35
#define BLOCK_SIZE 1024

__global__ void matrixVectorMul(double* A, double* x, double* y) {
    __shared__ double s_x[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // Load vector x into shared memory
    s_x[tx] = x[tx];

    __syncthreads();

    // Compute dot product of row i and vector x
    double sum = 0.0;
    for (int j = 0; j < N; j++) {
        sum += A[i * N + j] * s_x[j % BLOCK_SIZE];
    }

    // Store result in vector y
    y[i] = sum;
}

int main() {
    double* A, * x, * y;
    double* d_A, * d_x, * d_y;

    // Allocate memory on host
    A = (double*)malloc(N * N * sizeof(double));
    x = (double*)malloc(N * sizeof(double));
    y = (double*)malloc(N * sizeof(double));

    // Initialize input data
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() / (double)RAND_MAX;
    }
    for (int i = 0; i < N; i++) {
        x[i] = rand() / (double)RAND_MAX;
        y[i] = 0.0;
    }

    printf("The matrix is:\n");
    printf("***************");
    for (int i = 0;i < N * N;i++) {
        if (i % N == 0) {

            printf("\n");
        }
        cout << A[i] << " ";
    }
    printf("\n\n");

    printf("The vector is:\n");
    printf("***************\n");
    for (int i = 0;i < N;i++) {
        cout << x[i] << " ";
    }
    printf("\n\n");

    // Allocate memory on device
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));

    // Copy input data from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    matrixVectorMul << <dimGrid, dimBlock >> > (d_A, d_x, d_y);

    // Copy output data from device to host
    cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Print result
    printf("The result vector is:\n");
    printf("***************\n");
    for (int i = 0; i < N; ++i) {
        cout<<y[i] << " ";
    }
    printf("\n");


    double  *d_blasResult, *blasResult;
    cudaMalloc(&d_blasResult, N * sizeof(double));
    blasResult = (double*)malloc(N * sizeof(double));
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double scale = 1;

    cublasDgemv(handle, CUBLAS_OP_T, N, N, &scale, d_A, N, d_x,
                1, &scale, d_blasResult, 1);
    cudaMemcpy(blasResult, d_blasResult, N * sizeof(double), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    printf("The  blas result vector is:\n");
    printf("***************\n");
    for (int i = 0; i < N; ++i) {
        cout<<blasResult[i] << " ";
    }
    printf("\n");
    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    // Free memory on host
    free(A);
    free(x);
    free(y);

    return 0;
}