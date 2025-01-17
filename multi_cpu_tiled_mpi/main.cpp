// Inspired by https://github.com/tejaswiagarwal/multigpumatmul
#include <iostream>
#include <mpi.h>

#define PROBLEM_SIZE 1024

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

int main() {
    // seed RNG for kinda random stuff
    srand (static_cast <unsigned> (time(nullptr)));

    // pointers to the matrices
    int *matA, *matB, *matC;
    const unsigned int ROOT_NODE = 0;

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
    if(rank == ROOT_NODE){
        std::cout << "problem size: " << PROBLEM_SIZE << "X" << PROBLEM_SIZE << std::endl;
        std::cout << "World size: " << world_size << std::endl;
        matA = new int[PROBLEM_SIZE*PROBLEM_SIZE];
        matB = new int[PROBLEM_SIZE*PROBLEM_SIZE];
        matC = new int[PROBLEM_SIZE*PROBLEM_SIZE];

        std::cout << "Filling matrices with random numbers" << std::endl;
        fill_random(matA, PROBLEM_SIZE, PROBLEM_SIZE);
        fill_random(matB, PROBLEM_SIZE, PROBLEM_SIZE);
    }else{
        // allocate memory for matB on all other nodes before broadcasting the matrix
        matB = new int[PROBLEM_SIZE*PROBLEM_SIZE];
    }

    // all processes have to wait until matrix initialization is finished
    MPI_Barrier(MPI_COMM_WORLD);

    // ############ STEP #2 ############
    if (rank == ROOT_NODE)
        std::cout << "Distributing B matrix to all processes" << std::endl;
    // Broadcast matrix B to all processes
    MPI_Bcast(matB, PROBLEM_SIZE*PROBLEM_SIZE, MPI_INT, ROOT_NODE, MPI_COMM_WORLD);

    // ############ STEP #3 ############
    //Scatter rows of A to all processes
    int numOfRowsPerProcess = PROBLEM_SIZE / world_size;

    // allocate memory for the partial A matrix received by each process
    int *partialMatA = new int[numOfRowsPerProcess*PROBLEM_SIZE];

    if(rank == ROOT_NODE)
        std::cout << "Distributing pieces of matrix A to all processes" << std::endl;

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

    // ############ STEP #4 ############
    // Each process calculates respective rows of C with their data
    int *partialMatC = new int[numOfRowsPerProcess*PROBLEM_SIZE];
    std::cout << "rank " << rank << " got " << numOfRowsPerProcess << " rows of matrix A" << std::endl;
    std::cout << "rank " << rank << " got " << numOfRowsPerProcess*PROBLEM_SIZE << " elements of matrix A" << std::endl;

    // TODO: replace with call to CUDA matrix multiplication
    mat_mul_generic(partialMatA, matB, partialMatC, numOfRowsPerProcess, PROBLEM_SIZE, PROBLEM_SIZE);

    std::cout << "rank " << rank << " finished calculating " << numOfRowsPerProcess << " rows of result matrix" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == ROOT_NODE)
        std::cout << "All processes finished gathering results" << std::endl;

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

    // ############ STEP #6 ############
    // Verify the multiplication was successful by comparing with a serial implementation
    if(rank == ROOT_NODE){
        int *controlMatC = new int[PROBLEM_SIZE*PROBLEM_SIZE];

        std::cout << "Verifying result.." << std::endl;
        mat_mul_generic(matA, matB, controlMatC, PROBLEM_SIZE, PROBLEM_SIZE, PROBLEM_SIZE);

        if(mat_comp(matC, controlMatC, PROBLEM_SIZE))
            std::cout << "Algorithm successful!" << std::endl;
        else
            std::cout << "Tough shit :(" << std::endl;
    }

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
