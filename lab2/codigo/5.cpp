#include <mpi.h>
#include <iostream>
#include <vector>
#include <unistd.h> 
#include <cstring>  

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int comm_sz, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   
    char hostname[256];
    gethostname(hostname, 256);

    // Parámetros
    const int n = 6; 
    int block_size = n / comm_sz;

    std::vector<double> A_local(n * block_size); 
    std::vector<double> x(n);                     
    std::vector<double> y_local(n, 0.0);          

    if (rank == 0) {
     
        std::vector<double> A(n * n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                A[i * n + j] = i + j + 1;  

        for (int i = 0; i < n; i++)
            x[i] = 1.0; 

        
        for (int p = 0; p < comm_sz; p++) {
            if (p == 0) {
           
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < block_size; j++)
                        A_local[i * block_size + j] = A[i * n + p * block_size + j];
            } else {
                
                MPI_Send(&A[p * block_size], n * block_size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
            }
        }
    } else {
       
        MPI_Recv(A_local.data(), n * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Bcast(x.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::cout << "Proceso " << rank << " en host: " << hostname << std::endl;


    for (int i = 0; i < n; i++) {
        for (int j = 0; j < block_size; j++) {
            int col_index = rank * block_size + j; 
            y_local[i] += A_local[i * block_size + j] * x[col_index];
        }
    }


    std::vector<double> y_final(n, 0.0);
    MPI_Reduce(y_local.data(), y_final.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Resultado final y = A * x:\n";
        for (int i = 0; i < n; i++)
            std::cout << y_final[i] << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
