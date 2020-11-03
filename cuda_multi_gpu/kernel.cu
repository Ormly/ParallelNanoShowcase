#include "kernel.h"


__global__ void matrixMultiplication(const int *a, const int *b, int *c, int matrixDim) 
{
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * matrixDim + col] = 0;
  for (int k = 0; k < matrixDim; k++) {
    // Accumulate results for a single element
    	c[row * matrixDim + col] += a[row * matrixDim + k] * b[k * matrixDim + col];
  }	
}

__global__ void mysgemm(int m, int n, int k, const int *A, const int *B,
        int* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    int Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < k; ++i){
        Cvalue += (A[row * m + i]) * (B[i * n + col]);
    }
    C[row * n + col] = Cvalue;
}

__global__ void mm_kernel(int m, int n, int colsA, const int *A, const int *B, int *C){

  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int sum = 0;
  if ((row < m) && (col < n))
  {
    for (int i = 0; i < colsA; i++) 
	sum += A[colsA*row + i] * B[i*n+col];
    C[row*n+col] = sum;
  }
}


void launchKernel(const int THREADS, const int BLOCKS, const int *a, const int *b, int *c, int matrixDim)
{
	
	
	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);
	
	//Launch Kernel
	matrixMultiplication<<<blocks, threads>>>(a, b, c, matrixDim);
}

void launchKernelCUDA(int m, int n, int k, const int THREADS, const int BLOCKS, const int *a, const int *b, int *c)
{
	
	
	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);
	
	//Launch Kernel
	mm_kernel<<<blocks, threads>>>(m, n, k, a, b, c);
}


