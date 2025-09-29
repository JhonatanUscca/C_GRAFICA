#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <cmath>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, comm_sz;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ((comm_sz & (comm_sz - 1)) != 0) {
        if (rank == 0) {
            std::cerr << "El numero de procesos debe ser potencia de dos." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int value = rank + 1;  
    int global_sum = value;

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_len;
    MPI_Get_processor_name(hostname, &hostname_len);

    std::cout << "Proceso " << rank << " ejecutandose en " << hostname << " con valor inicial " << value << std::endl;

    int steps = (int)log2(comm_sz);
    for (int step = 0; step < steps; ++step) {
        int partner = rank ^ (1 << step);

        if (rank < partner) {
            std::cout << "Paso " << step << ": Proceso " << rank << " (" << hostname << ") envia/recibe con proceso " << partner << std::endl;
        }

        int recv_value;
        MPI_Sendrecv(&global_sum, 1, MPI_INT, partner, 0,
                     &recv_value, 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        global_sum += recv_value;
    }

    if (rank == 0) {
        std::cout << "Suma global final: " << global_sum << std::endl;
    }

    MPI_Finalize();
    return 0;
}
