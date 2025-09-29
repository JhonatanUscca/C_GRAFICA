#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <unistd.h>  

int find_bin(float value, const std::vector<float>& bin_maxes, float min_meas) {
    if (value < bin_maxes[0]) return 0;
    for (size_t b = 1; b < bin_maxes.size(); b++) {
        if (value < bin_maxes[b]) return b;
  
    return bin_maxes.size() - 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    std::cout << "Proceso " << rank << " ejecutandose en nodo: " << hostname << std::endl;

 
    std::vector<float> data;
    int data_count = 20;
    float min_meas = 0.0f;
    float max_meas = 5.0f;
    int bin_count = 5;

    if (rank == 0) {

        data = {1.3, 2.9, 0.4, 0.3, 1.3, 4.4, 1.7, 0.4, 3.2, 0.3, 
                4.9, 2.4, 3.1, 4.4, 3.9, 0.4, 4.2, 4.5, 4.9, 0.9};
        std::cout << "Proceso 0 leyendo datos y distribuyendo\n";
    }

    int chunk_size = data_count / size;
    std::vector<float> local_data(chunk_size);

    MPI_Scatter(data.data(), chunk_size, MPI_FLOAT,
                local_data.data(), chunk_size, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    float bin_width = (max_meas - min_meas) / bin_count;
    std::vector<float> bin_maxes(bin_count);
    for (int b = 0; b < bin_count; b++) {
        bin_maxes[b] = min_meas + bin_width * (b + 1);
    }


    std::vector<int> local_bin_counts(bin_count, 0);
    for (float val : local_data) {
        int bin = find_bin(val, bin_maxes, min_meas);
        local_bin_counts[bin]++;
    }


    std::vector<int> global_bin_counts(bin_count, 0);
    MPI_Reduce(local_bin_counts.data(), global_bin_counts.data(),
               bin_count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    
    if (rank == 0) {
        std::cout << "Histograma final:\n";
        for (int b = 0; b < bin_count; b++) {
            float bin_min = (b == 0) ? min_meas : bin_maxes[b - 1];
            float bin_max = bin_maxes[b];
            std::cout << "Bin [" << bin_min << ", " << bin_max << "): "
                      << global_bin_counts[b] << " elementos\n";
        }
    }

    MPI_Finalize();
    return 0;
}
