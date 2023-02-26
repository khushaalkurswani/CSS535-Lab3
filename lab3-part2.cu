#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string>
#include "cublas_v2.h"

using namespace std;

__global__ void multiplyMV(double *matrix, double *vector, double *result, int N)
{
    __shared__ double cache[1024];       // Cache the rows of x[] corresponding to this block.
    __shared__ double res[1024]; 
    int offset = blockIdx.x * blockDim.x;
    int row = (offset + threadIdx.x) % N;
 
        if (row < N) 
	    {	
        cache [row] = vector[threadIdx.x];
        __syncthreads();
        res[threadIdx.x] = cache[threadIdx.x] * matrix[blockIdx.x * blockDim.x + threadIdx.x];
        }

        if(threadIdx.x == 0) {
           double temp = 0.0; 
            for(int i = 0 ;  i < N; i++){
                temp += res[i];
            }
            result[blockIdx.x] = temp;
        }

}

// Returns a random double between 0.01 and 10
double generateRandDouble()
{
    return 10 * (double)rand() / (double)RAND_MAX + 0.01;
}

// Populates given array with random doubles
// array's length must match the passed in size parameter
void fillRandom(double *arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        arr[i] = generateRandDouble();
    }
}

// Populates given array with zeros (empty array)
// array's length must match the passed in size parameter
void fillDefault(double *arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        arr[i] = 0;
    }
}

void printConfig(int N, int numBlocks, int numThreads)
{
    cout << "Execution Configuration:" << endl;
    cout << "N = " << N << ", "
         << "Number of Blocks =  " << numBlocks << ", "
         << "Number of Threads Per Block = " << numThreads << endl;
    cout << endl;
}

// Print's matrix array's elements
// If matrix array has too many elements, then truncated matrix is printed
// matrix array is 1D array not a 2D array but logically represents a 2d
//      array such that each row is stored in order
// matrix array's length is N * N where N is the passed in parameter
//		representing number of rows (same as number of columns)
void printMatrix(double *matrix, int N)
{
    bool tooLarge = false;

    // check if matrix is too large
    if (N > 10)
    {
        N = 10;
        tooLarge = true;
    }

    // print matrix name
    cout << "Matrix" << endl;

    // print matrix elements
    for (int j = 0; j < N; j++)
    {
        for (int k = 0; k < N; k++)
        {
            cout << matrix[j * N + k] << " ";
        }

        // add ellipsis to represent truncation if matrix too large
        if (tooLarge)
        {
            cout << " ...";
        }

        cout << endl;
    }

    // add ellipsis to represent truncation if matrix too large
    if (tooLarge)
    {
        cout << " ..." << endl;
    }

    cout << endl;
}

// prints the elements in the given array vec
//		array vec's length much match N where N
//		is the number of elements in array
// Turncates array if too large
void printVec(double *vec, int N, string name)
{
    bool tooLarge = false;
    if (N > 100)
    {
        tooLarge = true;
        N = 100; // truncate array to 100 elements
    }

    cout << name << " :" << endl;
    for (int i = 0; i < N; i++)
    {
        cout << vec[i] << "  ";
    }

    if (tooLarge)
    {
        cout << " ... "; // elipsis represents truncated
    }
    cout << endl
         << endl;
}

int main(int argc, char *argv[])
{
    int N = 1024;
    int numBlocks = 1024;
    int numThreads = 1024;

    if (argc == 4) // argv[0] is the name of program
    {
        N = stoi(argv[1]);
        numBlocks = stoi(argv[2]);
        numThreads = stoi(argv[3]);

        if (N <= 0)
        {
            N = 1;
            numBlocks = 1;
            numThreads = 1;
        }

        if (numBlocks <= 0 && numThreads <= 0)
        {
            numBlocks = numThreads = 1;
        }
        else if (numBlocks == -1)
        {
            numBlocks = (N + numThreads - 1) / numThreads;
        }
        else if (numThreads == -1)
        {
            numThreads = (N + numBlocks - 1) / numBlocks;
        }
    }

    // host copies of matrix, vector, result
    double *matrix, *vector, *result, *blasResult;

    // device copies of matrix, vector, result
    double *d_matrix, *d_vector, *d_result, *d_blasResult;

    // allocate memory on host
    int vectorSize = N * sizeof(double);
    int matrixSize = N * N * sizeof(double);

    matrix = (double *)malloc(matrixSize);
    fillRandom(matrix, N * N);

    vector = (double *)malloc(vectorSize);
    fillRandom(vector, N);

    result = (double *)malloc(vectorSize);
    blasResult = (double *)malloc(vectorSize);

    // allocate memory on device
    cudaMalloc((void **)&d_matrix, matrixSize);
    cudaMalloc((void **)&d_vector, vectorSize);
    cudaMalloc((void **)&d_result, vectorSize);
    cudaMalloc((void **)&d_blasResult, vectorSize);

    // Copy inputs to device
    cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, vectorSize, cudaMemcpyHostToDevice);

    // lauch kernel function
    // int numBlocks = 1;
    // int numThreads = (N + numBlocks - 1) / numBlocks;
    
    multiplyMV<<<numBlocks, numThreads>>>(d_matrix, d_vector, d_result, N);

    // Copy result back to host
    cudaMemcpy(result, d_result, vectorSize, cudaMemcpyDeviceToHost);

    // Calculate using cuBLAS
    // cuBLAS is column-major but matrix and vector are stored in row-major
    //		so need to transpose matrix to ensure correct computation
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double scale = 1;
    cublasDgemv(handle, CUBLAS_OP_T, N, N, &scale, d_matrix, N, d_vector,
                1, &scale, d_blasResult, 1);
    cudaMemcpy(blasResult, d_blasResult, vectorSize, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);

    // print all data
    printConfig(N, numBlocks, numThreads);
    printMatrix(matrix, N);
    printVec(vector, N, "Vector");
    printVec(result, N, "Kernel Result");
    printVec(blasResult, N, "cuBLAS Result");

    // free memory on device
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
    cudaFree(d_blasResult);

    // free memory on host
    free(matrix);
    free(vector);
    free(result);
    free(blasResult);

    return 0;
}
