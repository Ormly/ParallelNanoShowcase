#include <iostream>
#include <cstdlib>
#include <ctime>

#define PROBLEM_SIZE 32

int *mat_a, *mat_b, *mat_c;

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

/// Sets mat as an identity matrix
/// \param mat
/// \param rows
/// \param cols
void fill_identity(int* mat, const size_t rows, const size_t cols){
    for(int r = 0; r < rows; ++r){
        for(int c = 0; c < cols; ++c){
            if(c == r)
                mat[r*rows + c] = static_cast<double>(1.0);
            else
                mat[r*rows + c] = 0;
        }
    }
}

/// Multiplies matrices a and b and writes the results to mat c
/// \param a
/// \param b
/// \param result
/// \param dims
void mat_mul(const int* a, const int* b, int* result, const size_t dims){
    for(int r = 0; r < dims; ++r) {
        for (int c = 0; c < dims; ++c) {
            for (int e = 0; e < dims; e++){
                result[r*dims + c] += a[r*dims + e] * b[e*dims + c];
            }
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
void print_mat(int* mat, const size_t rows, const size_t cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << mat[r * rows + c] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    mat_a = new int[PROBLEM_SIZE*PROBLEM_SIZE];
    mat_b = new int[PROBLEM_SIZE*PROBLEM_SIZE];
    mat_c = new int[PROBLEM_SIZE*PROBLEM_SIZE];


//    fill_identity(mat_a, PROBLEM_SIZE, PROBLEM_SIZE);
    fill_random(mat_a, PROBLEM_SIZE, PROBLEM_SIZE);
    fill_identity(mat_b, PROBLEM_SIZE, PROBLEM_SIZE);

    mat_mul(mat_a, mat_b, mat_c, PROBLEM_SIZE);

    if(mat_comp(mat_a, mat_c, PROBLEM_SIZE)){
        std::cout << "Matrices equal :)" << std::endl;
    }else{
        std::cout << "Matrices not equal :(" << std::endl;
    }


    return 0;
}
