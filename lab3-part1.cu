/**
 * Anthony Bustamante, Jesse Leu, Khushaal Kurswani
 * CSS 535 High Performance Computing
 * Professor Erika Parsons
 * 1 March 2023
 *
 * Lab 3 - Profiling GPU GEMV
 * Part 1 - Global Memory
 *
 * Calculate matrix-vector product using implemented GPU kernel function
 * Verify kernel function's result with cuBLAS result
 * Profile the kernel function to gain insight related to GPU
 * 5 execution configurations that are optimal for global memory are chosen to 
 *      test the kernel function
 *
 * Requires cuBLAS libraries
 *
 * Compile in CLI using the following command:
 *      nvcc lab3-part1.cu -lcublas
 *
 * It is recommended to transfer the output of the program to a file when 
 *      running the program
 * To do so, use the following command:
 *      ./a.out > lab3Part1.txt
 * 
 * To profile the kernel functions, nsight compute or nvprof can be used
 *      nsight compute CLI command: 
 *          ncu -o <profiler_output_file_name>  --set full <executable_file>
 * 
 *      nvprof CLI command:
 *          TODO: add nvprof CLI command
 */

#include <iostream>
#include <stdlib.h> 
#include <math.h>
#include <string>
#include <vector>
#include "cublas_v2.h"

using namespace std;

// kernel function where each thread performs matrix-vector multiplication 
//		for their corresponding element of the result vector
__global__ void multiplyMV(double *matrix, double *vector, double *result, int N) 
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) 
	{	
		for (int i = 0; i < N; i++) 
		{
			result[row] += matrix[row * N + i] * vector[i];
		}

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

// Subtracts the result vector array from blasResult vector array
// Stores the calculated difference in the residual vector array
// result, blasResult, and residual arrays' lenghths must match the passed in
//      N parameter
void calcResidual(double *result, double *blasResult, double *residual, int N)
{
    for (int i = 0; i < N; i++) 
    {
        residual[i] = blasResult[i] - result[i];
    }
}

// checks if residual vector has very small values
//residual array's length must match the passed in
//      N parameter
bool isResidualSmall(double *residual, int N)
{
    for (int i = 0; i < N; i++) 
    {
        if (abs(residual[i]) > 0.0001)
        {
            return false;
        }
    }
    
    return true;
}

// prints execution configuation
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
void printVec(double* vec, int N, string name) 
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
		cout <<  vec[i] << "  ";
	}
	
	if (tooLarge) 
	{
		cout << " ... "; // elipsis represents truncated
	}
	cout << endl << endl;
}

// sets up the execution configuration in the configs
// each element in configs is a vector where 
//      1st element is number of elements, 
//      2nd element is number of blocks, and 
//      3rd element is number of threads per block
void setUpConfigs(vector<vector<int>> &configs)
{
	// 1024 elements, 4 blocks, and 256 threads per block
    vector<int> config1 = {1024, 4, 256};
    configs.push_back(config1);
    
    // 4096 elements, 16 blocks, and 256 threads per block
	vector<int> config2 = {4096, 16, 256}; 
    configs.push_back(config2);
    
    // 12 elements, 1 block, 12 threads per block
	vector<int> config3 = {12, 1, 12}; 
    configs.push_back(config3);
    
    // 8192 elements, 16 blocks, 512 threads per block
    vector<int> config4 = {8192, 16, 512}; 
    configs.push_back(config4);
    
    // 11585 elements, 182 blocks, 64 threads per block
    vector<int> config5 = {11585, 182, 64}; 
    configs.push_back(config5); 
}

int main(int argc, char *argv[]) 
{
    // Set up execution configurations
    vector<vector<int>> configs;
    setUpConfigs(configs);
    
    // host copies of matrix, vector, result, blasResult, and residual
    double *matrix, *vector, *result, *blasResult, *residual; 

    // device copies of matrix, vector, result
    double *d_matrix, *d_vector, *d_result, *d_blasResult;
  
    for (int i = 0; i < configs.size(); i++)
    {
        // get execution configuration
        int N = configs[i][0];
        int numBlocks = configs[i][1];
        int numThreads = configs[i][2];
        
        // allocate memory on host
        int vectorSize = N * sizeof(double);
        int matrixSize = N * N * sizeof(double);

        matrix = (double *)malloc(matrixSize);
        fillRandom(matrix, N*N);
        
        vector = (double *)malloc(vectorSize);
        fillRandom(vector, N);

        result = (double *)malloc(vectorSize);
        blasResult = (double *)malloc(vectorSize);
        residual = (double *)malloc(vectorSize);

        // allocate memory on device
        cudaMalloc((void **)&d_matrix, matrixSize);
        cudaMalloc((void **)&d_vector, vectorSize);
        cudaMalloc((void **)&d_result, vectorSize);
        cudaMalloc((void **)&d_blasResult, vectorSize);

        // Copy inputs to device
        cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector, vector, vectorSize, cudaMemcpyHostToDevice);

        // lauch kernel function 
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
        
        // calculate residual
        calcResidual(result, blasResult, residual, N);
        bool isSmallResidual = isResidualSmall(residual, N);

        // print all data
        printConfig(N, numBlocks, numThreads);
        printMatrix(matrix, N);
        printVec(vector, N, "Vector");
        printVec(result, N, "Kernel Result");
        printVec(blasResult, N, "cuBLAS Result");
        printVec(residual, N, "Residual");
        cout << "Is residual close to or equal to 0? ";
        if (isSmallResidual) 
        {
            cout << "Yes" << endl;
        }
        else 
        {
            cout << "No" << endl;
        }        
        cout << endl << endl;        

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
        free(residual);
    }

	return 0;
}
