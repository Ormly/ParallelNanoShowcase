#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <kernel.h>

#define CUDA_THREADS 	32 
#define MATRIX_DIM 		1024


/// Compares matrices a and b and returns true they're equal
/// \param a
/// \param b
/// \param dims
/// \return
bool mat_comp(const vector<int>& a, const vector<int>& b, const size_t dims){
    for(int r = 0; r < dims; ++r) {
        for (int c = 0; c < dims; ++c) {
            if(a[r*dims + c] != b[r*dims + c]){return false;}
        }
    }
    return true;
}


/// Fill mat random integers
/// \param mat
/// \param rows
/// \param cols
void fill_random(int* mat, const size_t rows, const size_t cols){
    srand (static_cast <unsigned> (time(nullptr)));

    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            mat[r*rows + c] =  -100 + static_cast <int> (rand()) /( static_cast <int> (RAND_MAX/(100+100)));
        }
    }
}

void mat_mul(const vector<int>& a, const vector<int>& b, const vector<int>& result, const size_t dims){
    for(int r = 0; r < dims; ++r) {
        for (int c = 0; c < dims; ++c) {
            for (int e = 0; e < dims; e++){
                result[r*dims + c] += a[r*dims + e] * b[e*dims + c];
            }
        }
    }
}

int main()
{
	int cudaBlocks = MATRIX_DIM / CUDA_THREADS;
	
	
	//size of the matrix in bytes (need for mem allocation)
	size_t bytes = MATRIX_DIM * MATRIX_DIM * sizeof(int);
	
	//Host vectors (CPP style)
	std::vector<int> h_a(N * N);
	std::vector<int> h_b(N * N);
	std::vector<int> h_c(N * N);
	std::vector<int> cpu_result(N * N);
	
	//Fill the vectors with random values (CPP style)
	std::generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	std::generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
	
	mat_mul(h_a, h_b, cpu_result, MATRIX_DIM);
	
	//GPU allocation
	int *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);
	
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
	
	launchKernel(CUDA_THREADS, cudaBlocks, d_a, d_b, d_c);
	
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
	
	//Compare the matrices
	if(mat_comp(h_c, cpu_result, MATRIX_DIM))
		cout << "Computation successful" << endl;
	else
		cout << "Computation not successful" << endl;
	
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
	
	
}
