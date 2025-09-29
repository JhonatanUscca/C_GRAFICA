#include <mpi.h>
#include <iostream>
#include <cmath>
#include <unistd.h>  
int main(int argc, char* argv[]) {
    int rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    int local_val = rank;
    int temp;
    int partner;

    int next_power = std::pow(2, std::floor(std::log2(comm_sz)));

    for (int stride = 1; stride < next_power; stride *= 2) {
        if (rank % (2 * stride) == 0) {
            partner = rank + stride;
            if (partner < comm_sz) {
                MPI_Recv(&temp, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::cout << "Proceso " << rank << " (host: " << hostname << ") recibio " 
                          << temp << " de proceso " << partner << std::endl;
                local_val += temp;
            }
        } else {
            partner = rank - stride;
            std::cout << "Proceso " << rank << " (host: " << hostname << ") envia " 
                      << local_val << " a proceso " << partner << std::endl;
            MPI_Send(&local_val, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            break;
        }
    }

    if (rank >= next_power) {
        std::cout << "Proceso " << rank << " (host: " << hostname << ") envia " 
                  << local_val << " a proceso 0" << std::endl;
        MPI_Send(&local_val, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else if (rank == 0) {
        for (int i = next_power; i < comm_sz; ++i) {
            MPI_Recv(&temp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::cout << "Proceso 0 (host: " << hostname << ") recibio " << temp 
                      << " de proceso " << i << std::endl;
            local_val += temp;
        }

        std::cout << "Suma global = " << local_val << std::endl;
    }

    MPI_Finalize();
    return 0;
}



