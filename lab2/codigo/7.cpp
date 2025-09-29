#include <mpi.h>
#include <iostream>
#include <unistd.h>    
#include <iomanip>    

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "Este programa requiere al menos 2 procesos." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    const int PING_PONG_COUNT = 1000;
    int ping_pong_count = 0;
    int partner_rank = (rank == 0) ? 1 : 0;

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    MPI_Barrier(MPI_COMM_WORLD); 
    while (ping_pong_count < PING_PONG_COUNT) {
        if (rank == ping_pong_count % 2) {
           
            double start = MPI_Wtime();

   
            ping_pong_count++;
            MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);

          
            double end = MPI_Wtime();
            double duration = end - start;

            std::cout << std::fixed << std::setprecision(9);
            std::cout << "ENVIO-Proceso " << rank << " (" << hostname << ") -> Proceso " << partner_rank
                      << " | Ping-Pong " << ping_pong_count
                      << " | Tiempo: " << duration << " s" << std::endl;

        } else {
            double start = MPI_Wtime();

   
            MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

 
            double end = MPI_Wtime();
            double duration = end - start;

            std::cout << std::fixed << std::setprecision(9);
            std::cout << "RECEPCION-Proceso " << rank << " (" << hostname << ") <- Proceso " << partner_rank
                      << " | Ping-Pong " << ping_pong_count
                      << " | Tiempo: " << duration << " s" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
