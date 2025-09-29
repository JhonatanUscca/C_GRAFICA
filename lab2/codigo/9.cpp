#include <mpi.h>
#include <iostream>
#include <vector>
#include <unistd.h> 
#include <cstring>

const int VECTOR_SIZE = 9;  

void print_hostname(int rank) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    std::cout << "Rank " << rank << " en host: " << hostname << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char hostname[256];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) std::cerr << "Este programa requiere exactamente 3 procesos." << std::endl;
        MPI_Finalize();
        return 1;
    }

    gethostname(hostname, sizeof(hostname));
    std::cout << "Proceso " << rank << " ejecutandose en: " << hostname << std::endl;

    std::vector<int> full_vector;

    if (rank == 0) {

        full_vector.resize(VECTOR_SIZE);
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            full_vector[i] = i;
        }
    }

    int block_size = VECTOR_SIZE / size;
    std::vector<int> local_block(block_size);

    double start_block = MPI_Wtime();
    MPI_Scatter(full_vector.data(), block_size, MPI_INT,
                local_block.data(), block_size, MPI_INT,
                0, MPI_COMM_WORLD);
    double end_block = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Proceso " << rank << " recibio (bloque): ";
    for (int val : local_block) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::vector<int> local_cyclic(block_size);

    double start_cyclic = MPI_Wtime();

    for (int i = 0; i < block_size; ++i) {
        int global_index = rank * block_size + i;
        int dest_rank = global_index % size;
        MPI_Send(&local_block[i], 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD);
    }

    for (int i = 0; i < VECTOR_SIZE / size; ++i) {
        MPI_Recv(&local_cyclic[i], 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double end_cyclic = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Proceso " << rank << " recibio (ciclico): ";
    for (int val : local_cyclic) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::vector<int> back_to_block(block_size);

    double start_reblock = MPI_Wtime();

    for (int i = 0; i < block_size; ++i) {
        int global_index = i * size + rank;
        int dest_rank = global_index / block_size;
        MPI_Send(&local_cyclic[i], 1, MPI_INT, dest_rank, 1, MPI_COMM_WORLD);
    }

    for (int i = 0; i < block_size; ++i) {
        MPI_Recv(&back_to_block[i], 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double end_reblock = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Proceso " << rank << " recibio (bloques de nuevo): ";
    for (int val : back_to_block) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

 
    if (rank == 0) {
        std::cout << "\nTiempo de distribucion por bloques: " << (end_block - start_block) << " segundos" << std::endl;
        std::cout << "Tiempo de redistribucion a ciclica: " << (end_cyclic - start_cyclic) << " segundos" << std::endl;
        std::cout << "Tiempo de redistribucion de regreso a bloques: " << (end_reblock - start_reblock) << " segundos" << std::endl;
    }

    MPI_Finalize();
    return 0;
}


