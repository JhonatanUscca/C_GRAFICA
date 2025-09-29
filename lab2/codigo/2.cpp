#include <mpi.h>
#include <iostream>
#include <cstdlib>  
#include <ctime>    
#include <cmath>    
#include <unistd.h> 

double random_double() {
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    long long int number_of_tosses = 0;

    if (world_rank == 0) {
      
        std::cout << "Ingrese el numero total de lanzamientos: ";
        std::cin >> number_of_tosses;
    }

    MPI_Bcast(&number_of_tosses, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    long long int tosses_per_proc = number_of_tosses / world_size;
    if (world_rank == world_size - 1) {
    
        tosses_per_proc += number_of_tosses % world_size;
    }


    char hostname[256];
    gethostname(hostname, 256);

 
    std::cout << "Proceso " << world_rank << " en host " << hostname
              << " procesara " << tosses_per_proc << " lanzamientos." << std::endl;

 
    srand(time(NULL) + world_rank);

    long long int local_number_in_circle = 0;
    for (long long int toss = 0; toss < tosses_per_proc; toss++) {
        double x = random_double();
        double y = random_double();
        if (x*x + y*y <= 1.0) {
            local_number_in_circle++;
        }
    }

    long long int global_number_in_circle = 0;


    MPI_Reduce(&local_number_in_circle, &global_number_in_circle, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double pi_estimate = 4.0 * (double)global_number_in_circle / (double)number_of_tosses;
        std::cout << "Estimacion de pi = " << pi_estimate << std::endl;
    }

    MPI_Finalize();
    return 0;
}
