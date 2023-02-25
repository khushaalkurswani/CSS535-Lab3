#include <iostream>
#include <stdlib.h> 
#include <math.h>
#include <string>
#include <vector>
#include "cublas_v2.h"

using namespace std;

#define UNROLL_CONST 4;

// kernel function where each thread performs matrix-vector multiplication 
//		for their corresponding 4 elements of the result vector
__global__ void multiplyMV(double *matrix, double *vector, double *result, int N) 
{
	int row = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (row < N && row + 3 < N) 
	{	
		for (int i = 0; i < N; i++) 
		{
			result[row] += matrix[row * N + i] * vector[i];
            result[row + 1] += matrix[(row + 1) * N + i] * vector[i];
            result[row + 2] += matrix[(row + 2) * N + i] * vector[i];
            result[row + 3] += matrix[(row + 3) * N + i] * vector[i];
            
		}
	}
    else if (row < N)
    {
        int leftOver = N - row;
        for (int i = 0; i < N; i++) 
		{
            for (int j = 0; j < leftOver; j++)
            {
                result[row + j] += matrix[(row + j) * N + i] * vector[i];
            }
        }
    }
}

// kernel function where each thread performs matrix-vector multiplication 
//		for their corresponding element of the result vector after the 
//      offset index
__global__ void multiplyMVLeftOver(double *matrix, double *vector, 
    double *result, int N, int offset)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x + offset;
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

bool isResidualSmall(double *residual, int N)
{
    for (int i = 0; i < N; i++) 
    {
        if (residual[i] > 0.0001)
        {
            return false;
        }
    }
    
    return true;
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
	// 1024 elements, 5 blocks, and 205 threads per block
    vector<int> config1 = {1025, 2, 256};
    configs.push_back(config1);
    
    // 4095 elements, 12 blocks, 342 threads per block
	//vector<int> config2 = {4095, 12, 342}; 
    //configs.push_back(config2);
    
    // 12 elements, 12 blocks, 1 threads per block
	vector<int> config3 = {15, 1, 4}; 
    configs.push_back(config3);
    
    //8190/13 =630
    //vector<int> config4 = {8190, 13, 630}; 
    //configs.push_back(config4);
    
    //11585/200=58
    //vector<int> config5 = {11585, 200, 58}; 
    //configs.push_back(config5);
}

int main(int argc, char *argv[]) 
{
    // Set up execution configurations
    vector<vector<int>> configs;
    setUpConfigs(configs);
    
    // host copies of matrix, vector, result
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
        
        //int completed = (N / 4) * 4;
        //int leftOver = N - completed;
        //multiplyMVLeftOver<<<1, leftOver>>>(d_matrix, d_vector, d_result, N, completed); 
        
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
