#include <cuda_runtime.h>

void launchKernel(const int THREADS, const int BLOCKS, const int *a, const int *b, int *c, int matrixDim);

void launchKernelCUDA(int m, int n, int k, const int THREADS, const int BLOCKS, const int *a, const int *b, int *c);
