#include <mpi.h>
#include <iostream>
#include <vector>
#include <unistd.h>  
#include <cstring>   


const int N = 6;

void print_vector(const std::vector<double>& v) {
    for (auto val : v)
        std::cout << val << " ";
    std::cout << "\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (comm_sz != 3) {
        if (rank == 0)
            std::cerr << "Este programa esta disenado para ejecutarse con 3 procesos.\n";
        MPI_Finalize();
        return 1;
    }


    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::cout << "Proceso " << rank << " ejecutandose en host: " << hostname << std::endl;

    int block_size = N / comm_sz; /

    std::vector<double> A; 
    std::vector<double> x(N);

    if (rank == 0) {
 
        A.resize(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                A[i * N + j] = i + j;


        for (int i = 0; i < N; ++i)
            x[i] = 1.0;

        std::cout << "Matriz A:\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                std::cout << A[i * N + j] << " ";
            std::cout << "\n";
        }

        std::cout << "Vector x:\n";
        print_vector(x);
    }

    std::vector<double> A_block(block_size * N);

    MPI_Scatter(
        A.data(), block_size * N, MPI_DOUBLE,
        A_block.data(), block_size * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);


    MPI_Bcast(x.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> y_block(block_size, 0.0);
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < N; ++j) {
            y_block[i] += A_block[i * N + j] * x[j];
        }
    }

    std::vector<double> y;
    if (rank == 0) {
        y.resize(N);
    }

    MPI_Gather(
        y_block.data(), block_size, MPI_DOUBLE,
        y.data(), block_size, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Resultado y = A * x:\n";
        print_vector(y);
    }

    MPI_Finalize();
    return 0;
}
