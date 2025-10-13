#include <iostream>
#include <vector>
#include <pthread.h>
#include <chrono>
#include <cstdlib>

struct Work {
    int from_row, to_row;  
    int ncols;
    const double* matrix;  
    const double* vec;
    double* out;
};

void* worker(void* arg) {
    Work* w = (Work*)arg;
    for (int i = w->from_row; i < w->to_row; ++i) {
        double sum = 0.0;
        const double* row = w->matrix + (size_t)i * w->ncols;
        for (int j = 0; j < w->ncols; ++j) sum += row[j] * w->vec[j];
        w->out[i] = sum;
    }
    return nullptr;
}

void evaluate(int N, int M, int nthreads) {
    std::vector<double> matrix((size_t)N * M);
    std::vector<double> vec(M);
    std::vector<double> out(N);

   
    for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = (double)(rand() % 100) / 10.0;
    for (int j = 0; j < M; ++j) vec[j] = (double)(rand() % 100) / 10.0;

    pthread_t threads[nthreads];
    std::vector<Work> works(nthreads);

    int base = N / nthreads;
    int rem = N % nthreads;
    int start = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < nthreads; ++t) {
        int rows = base + (t < rem ? 1 : 0);
        works[t].from_row = start;
        works[t].to_row = start + rows;
        works[t].ncols = M;
        works[t].matrix = matrix.data();
        works[t].vec = vec.data();
        works[t].out = out.data();
        pthread_create(&threads[t], nullptr, worker, &works[t]);
        start += rows;
    }

    for (int t = 0; t < nthreads; ++t) pthread_join(threads[t], nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = t1 - t0;

    std::cout << "N=" << N << " M=" << M << " threads=" << nthreads << " time=" << dur.count() << " s\n";

    
    for (int i = 0; i < std::min(5, N); ++i) std::cout << out[i] << " ";
    std::cout << "\n";
}

int main(int argc, char** argv) {
   
    std::vector<int> thread_counts = {1, 2, 4};

  
    std::vector<std::tuple<int, int>> matrix_sizes = {
        {4000, 4000},  
        {400, 400},   
        {4, 4000}      
    };

    
    for (auto& size : matrix_sizes) {
        int N = std::get<0>(size);
        int M = std::get<1>(size);

        for (int nthreads : thread_counts) {
            std::cout << "Evaluando matriz " << N << "x" << M << " con " << nthreads << " hilos\n";
            evaluate(N, M, nthreads);
        }
    }

    return 0;
}
