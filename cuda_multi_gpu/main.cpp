// Inspired by https://github.com/tejaswiagarwal/multigpumatmul
#include <iostream>
#include <mpi.h>
#include "kernel.h"
#include <chrono>
#include <iomanip> 

//#define PROBLEM_SIZE    128
//#define CUDA_THREADS 	32

/// Fill mat random integers
/// \param mat
/// \param rows
/// \param cols
void fill_random(int* mat, const size_t rows, const size_t cols){
    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            mat[r*rows + c] =  static_cast <int> (rand()) /( static_cast <int> (RAND_MAX/10));
        }
    }
}

/// Multiplies matrices a and b and writes the results to mat c
/// \param a - left matrix (mxm)
/// \param b - right matrix (mxp)
/// \param c - result matrix (nxp)
/// \param n - rows in left matrix
/// \param m - columns in left matrix, rows in right right matrix
/// \param p - columns in right matrix
void mat_mul_generic(
        const int* a,
        const int* b,
        int* c,
        const size_t n,
        const size_t m,
        const size_t p
    ){
    for(int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            int sum = 0;
            for (int k = 0; k < m; ++k){
                sum += a[i*p + k] * b[k*m+j];
            }
            c[i*p +j] = sum;
        }
    }
}


/// Compares matrices a and b and returns true they're equal
/// \param a
/// \param b
/// \param dims
/// \return
bool mat_comp(const int* a, const int* b, const size_t dims){
    for(int r = 0; r < dims; ++r) {
        for (int c = 0; c < dims; ++c) {
            if(a[r*dims + c] != b[r*dims + c]){return false;}
        }
    }
    return true;
}

/// print mat row by row
/// \param mat
/// \param rows
/// \param cols
void print_mat(int* mat, const size_t rows, const size_t cols, const char* name) {
    for (int r = 0; r < rows; ++r) {
        std::cout << name << " row: " << r << " - ";
        for (int c = 0; c < cols; ++c) {
            std::cout << mat[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) 
{
	 auto start_time_total = std::chrono::high_resolution_clock::now();
	 
	 //Command line arguments
	 char * argument_a = argv[1];
	 char * argument_b = argv[2];
	 int PROBLEM_SIZE = atoi(argument_a);
    int CUDA_THREADS = atoi(argument_b);
    
    // seed RNG for kinda random stuff
    srand (static_cast <unsigned> (time(nullptr)));

    //GPU memory pointers
    int *d_a, *d_b, *d_c;
    // pointers to the matrices
    int *matA, *matB, *matC;
    const unsigned int ROOT_NODE = 0;
	
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
	
	auto start_time_mult = std::chrono::high_resolution_clock::now();
	auto end_time_mult = std::chrono::high_resolution_clock::now();
	 
	 

    // global metrics (each node has its own)
    std::chrono::duration<double, std::milli> kernel_execution_time = end_time - start_time;
   

    //root node metrics
    std::chrono::duration<double, std::milli> verification_time, matB_broadcast, matA_gather, matA_scatter, total_exec_time, mul_time; 

    // init MPI
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // ############ STEP #1 ############
    //  initialize matrices A, B and C and fill A and B with random data
    // only performed by root node
    if(rank == ROOT_NODE)
	{
        std::cout << "problem size: " << PROBLEM_SIZE << "X" << PROBLEM_SIZE << std::endl;
        std::cout << "World size: " << world_size << std::endl;
		 std::cout << "CUDA Threads: " << CUDA_THREADS << std::endl;
        matA = new int[PROBLEM_SIZE*PROBLEM_SIZE];
        matB = new int[PROBLEM_SIZE*PROBLEM_SIZE];
        matC = new int[PROBLEM_SIZE*PROBLEM_SIZE];
        cudaMalloc(&d_b, PROBLEM_SIZE * PROBLEM_SIZE * sizeof(int));

        std::cout << "Filling matrices with random numbers" << std::endl;
        fill_random(matA, PROBLEM_SIZE, PROBLEM_SIZE);
        fill_random(matB, PROBLEM_SIZE, PROBLEM_SIZE);
    }
	else
	{
        // allocate memory for matB on all other nodes before broadcasting the matrix
        matB = new int[PROBLEM_SIZE*PROBLEM_SIZE];
		cudaMalloc(&d_b, PROBLEM_SIZE * PROBLEM_SIZE * sizeof(int));
    }

    // all processes have to wait until matrix initialization is finished
    MPI_Barrier(MPI_COMM_WORLD);

    // ############ STEP #2 ############
    if (rank == ROOT_NODE)
		start_time_mult = std::chrono::high_resolution_clock::now();
    	//std::cout << "Distributing B matrix to all processes" << std::endl;
    // Broadcast matrix B to all processes
    if (rank == ROOT_NODE)
		start_time = std::chrono::high_resolution_clock::now();

    MPI_Bcast(matB, PROBLEM_SIZE*PROBLEM_SIZE, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);

    if (rank == ROOT_NODE)
    {
		end_time = std::chrono::high_resolution_clock::now();
		matB_broadcast = end_time - start_time;
    }
	
    cudaMemcpy(d_b, matB, PROBLEM_SIZE * PROBLEM_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // ############ STEP #3 ############
    //Scatter rows of A to all processes
    int numOfRowsPerProcess = PROBLEM_SIZE / world_size;

    // allocate memory for the partial A matrix received by each process
    int *partialMatA = new int[numOfRowsPerProcess*PROBLEM_SIZE];

    //Allocate memory for partial matrix A on each GPU
    cudaMalloc(&d_a, numOfRowsPerProcess * PROBLEM_SIZE * sizeof(int));

    if(rank == ROOT_NODE)
        //std::cout << "Distributing pieces of matrix A to all processes" << std::endl;
    if(rank == ROOT_NODE)
    	start_time = std::chrono::high_resolution_clock::now();
    MPI_Scatter(
            matA,                                               // the matrix to scatter
            numOfRowsPerProcess*PROBLEM_SIZE,          // how many element sent to each process
            MPI_INT,                                            // datatype sent
            partialMatA,                                        // receive buffer
            numOfRowsPerProcess*PROBLEM_SIZE,          // how many elements are received by each process
            MPI_INT,                                            // datatype received
            ROOT_NODE,                                          // rank of root node
            MPI_COMM_WORLD                                      // communicator
            );

    if(rank == ROOT_NODE)
    {
		end_time = std::chrono::high_resolution_clock::now();
		matA_scatter = end_time - start_time;
    }
    MPI_Barrier(MPI_COMM_WORLD); 	
    cudaMemcpy(d_a, partialMatA, numOfRowsPerProcess * PROBLEM_SIZE * sizeof(int), cudaMemcpyHostToDevice); 	
    // ############ STEP #4 ############
    // Each process calculates respective rows of C with their data
    int * partialMatC = new int[numOfRowsPerProcess*PROBLEM_SIZE];
    int cudaBlocks = PROBLEM_SIZE / CUDA_THREADS; 

    //Allocate memory for partial matrix C on each GPU 
    cudaMalloc(&d_c, numOfRowsPerProcess * PROBLEM_SIZE * sizeof(int));

    cudaMemcpy(d_c, partialMatC, numOfRowsPerProcess * PROBLEM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
 

    std::cout << "rank " << rank << " got " << numOfRowsPerProcess << " rows of matrix A" << std::endl;
    std::cout << "rank " << rank << " got " << numOfRowsPerProcess*PROBLEM_SIZE << " elements of matrix A" << std::endl;

    // TODO: replace with call to CUDA matrix multiplication
    //mat_mul_generic(partialMatA, matB, partialMatC, numOfRowsPerProcess, PROBLEM_SIZE, PROBLEM_SIZE);
    	
	launchKernelCUDA(numOfRowsPerProcess, PROBLEM_SIZE, PROBLEM_SIZE, CUDA_THREADS, cudaBlocks, d_a, d_b, d_c);


	end_time = std::chrono::high_resolution_clock::now();
	kernel_execution_time = end_time - start_time;
    

    cudaMemcpy(partialMatC, d_c, numOfRowsPerProcess * PROBLEM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

		
    std::cout << "rank " << rank << " finished calculating " << numOfRowsPerProcess << " rows of result matrix" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == ROOT_NODE)
      std::cout << "All processes finished gathering results" << std::endl;
    if (rank == ROOT_NODE)
		start_time = std::chrono::high_resolution_clock::now();

    // ############ STEP #5 ############
    // Gather rows of C from all processes and assemble them into the final C
    MPI_Gather(
            partialMatC,                                    // the partial result sent by each processor to the root
            numOfRowsPerProcess*PROBLEM_SIZE,      // number of elements in partial result
            MPI_INT,                                        // sent datatype
            matC,                                           // the receive buffer - final result matrix
            numOfRowsPerProcess*PROBLEM_SIZE,      // number of elements received from each process
            MPI_INT,                                        // received datatype
            ROOT_NODE,                                      // rank of root node
            MPI_COMM_WORLD                                  // communicator
            );

    if (rank == ROOT_NODE)
    {
		end_time = std::chrono::high_resolution_clock::now();
		matA_gather = end_time - start_time;
    }
    MPI_Barrier(MPI_COMM_WORLD);
	

    // ############ STEP #6 ############
    // Verify the multiplication was successful by comparing with a serial implementation
    if(rank == ROOT_NODE)
    {
		end_time_mult = std::chrono::high_resolution_clock::now();
        int *controlMatC = new int[PROBLEM_SIZE*PROBLEM_SIZE];

        std::cout << "Verifying result.." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
		
        mat_mul_generic(matA, matB, controlMatC, PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE);

	/*
	for(int r = 0; r < PROBLEM_SIZE; ++r) 
	{
        	for (int c = 0; c < PROBLEM_SIZE; ++c) 
		{
            		std::cout << matC[r*PROBLEM_SIZE + c] << "    " << controlMatC[r*PROBLEM_SIZE + c] << 	std::endl;
        	}
    	}
*/
	
        if(mat_comp(matC, controlMatC, PROBLEM_SIZE))
		{
	     	end_time = std::chrono::high_resolution_clock::now();
            verification_time = end_time - start_time;
            std::cout << "Algorithm successful!\n" << std::endl;
            /*
            std::cout << "Result metrics:" << std::endl;
	    	std::cout << "Verification duration: " << (verification_time/std::chrono::milliseconds(1)) << " milliseconds" << std::endl;
            std::cout << "Matrix B broadcast duration: " << (matB_broadcast/std::chrono::milliseconds(1)) << " milliseconds" << std::endl;
            std::cout << "Matrix A scatter duration: " << (matA_scatter/std::chrono::milliseconds(1)) << " milliseconds" << std::endl;
            std::cout << "Matrix A gather duration: " << (matA_gather/std::chrono::milliseconds(1)) << " milliseconds" << std::endl;

			std::cerr << (verification_time/std::chrono::milliseconds(1)) << std::endl;
			std::cerr << (matB_broadcast/std::chrono::milliseconds(1)) << std::endl;
			std::cerr << (matA_scatter/std::chrono::milliseconds(1)) << std::endl;
			std::cerr << (matA_gather/std::chrono::milliseconds(1)) << std::endl;
			*/
		}	
		else
		{
			std::cout << "Tough shit :(" << std::endl;
		}
		
    }
      //std::cout << "Node " << processor_name << " with rank: " << rank << " finished the kernel execution in "  
      //<< kernel_execution_time/std::chrono::milliseconds(1) << " milliseconds" <<std::endl; 
	  
	  //std::cerr << (kernel_execution_time/std::chrono::milliseconds(1)) << std::endl;
  
	if(rank == ROOT_NODE)
	{
		auto end_time_total = std::chrono::high_resolution_clock::now();
		total_exec_time = end_time_total - start_time_total;
		std::cout << "Program Execution time: " << total_exec_time/std::chrono::milliseconds(1) << " milliseconds" << std::endl;
		std::cerr << (total_exec_time/std::chrono::milliseconds(1)) << std::endl;
		
		std::cout << "Multiplication execution time Execution time: " << mul_time/std::chrono::milliseconds(1) << milliseconds << std::endl;
		std::cerr << (mul_time/std::chrono::milliseconds(1)) << std::endl;
	}
	
	

    // Finalize the MPI environment.
	MPI_Finalize();
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	std::cerr << " " << std::endl;
	

    return 0;
}
