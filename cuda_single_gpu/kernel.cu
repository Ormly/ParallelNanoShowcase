#include "kernel.h"


__global__ void matrixMultiplication(const int *a, const int *b, int *c, int matrixDim) 
{
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

void launchKernel(const int THREADS, const int BLOCKS, const int *a, const int *b, int *c, int matrixDim)
{
	
	
	// Use dim3 structs for block  and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);
	
	//Launch Kernel
	matrixMultiplication<<<blocks, threads>>>(a, b, c, N);
}
