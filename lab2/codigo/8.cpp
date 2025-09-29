#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <unistd.h>

using namespace std;


vector<int> merge(const vector<int>& left, const vector<int>& right) {
    vector<int> result;
    auto it_left = left.begin();
    auto it_right = right.begin();

    while (it_left != left.end() && it_right != right.end()) {
        if (*it_left < *it_right) {
            result.push_back(*it_left++);
        } else {
            result.push_back(*it_right++);
        }
    }

    result.insert(result.end(), it_left, left.end());
    result.insert(result.end(), it_right, right.end());
    return result;
}


vector<int> generate_random_list(int n) {
    vector<int> v(n);
    for (int &x : v) {
        x = rand() % 100; 
    }
    return v;
}

void print_list(const vector<int>& v, const string& label) {
    cout << label << ": ";
    for (int x : v) {
        cout << x << " ";
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Get_processor_name(hostname, &name_len);

    if (argc != 2) {
        if (world_rank == 0)
            cerr << "Uso: mpirun -np <num_proc> ./programa <n_total>" << endl;
        MPI_Finalize();
        return 1;
    }

    int n_total = atoi(argv[1]); 
    int n_local = n_total / world_size;


    srand(time(nullptr) + world_rank);

    vector<int> local_list = generate_random_list(n_local);
    sort(local_list.begin(), local_list.end());


    if (world_rank == 0) {
        cout << "Listas locales ordenadas:\n";
    }

    for (int i = 0; i < world_size; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == world_rank) {
            cout << "Proceso " << world_rank << " en " << hostname << ": ";
            print_list(local_list, "Local");
        }
    }


    int step = 1;
    vector<int> merged = local_list;

    while (step < world_size) {
        if (world_rank % (2 * step) == 0) {
            if (world_rank + step < world_size) {

                MPI_Status status;
                int recv_size;



                MPI_Recv(&recv_size, 1, MPI_INT, world_rank + step, 0, MPI_COMM_WORLD, &status);
                vector<int> recv_buffer(recv_size);

                MPI_Recv(recv_buffer.data(), recv_size, MPI_INT, world_rank + step, 0, MPI_COMM_WORLD, &status);

                char sender_hostname[MPI_MAX_PROCESSOR_NAME];
                MPI_Recv(sender_hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, world_rank + step, 0, MPI_COMM_WORLD, &status);

                cout << "Proceso " << world_rank << " en " << hostname
                     << " recibio " << recv_size << " elementos de proceso "
                     << world_rank + step << " en " << sender_hostname << endl;

                merged = merge(merged, recv_buffer);
            }
        } else {
            int dest = world_rank - step;
            int send_size = merged.size();

            MPI_Send(&send_size, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(merged.data(), send_size, MPI_INT, dest, 0, MPI_COMM_WORLD);
            MPI_Send(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    if (world_rank == 0) {
        cout << "\nLista global ordenada en " << hostname << ": ";
        for (int x : merged) cout << x << " ";
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
