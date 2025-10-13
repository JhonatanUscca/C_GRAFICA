#include <iostream>
#include <vector>
#include <string>
#include <pthread.h>
#include <cstring>
#include <pthread.h> // Para barreras
#include <chrono>  // Para medir el tiempo
#include <unistd.h>

const char* lines[] = {

    "El cielo está azul",
    "Las estrellas brillan fuerte",
    "El viento sopla suavemente",
    "El sol se pone en el horizonte",
    "Una nueva línea con más palabras aquí",
    nullptr
};


struct Arg {
    int id;
    int start_idx;
    int stride;
    pthread_barrier_t* barrier; 
};

void* unsafe_worker(void* a) {
    Arg* arg = (Arg*)a;
    for (int i = arg->start_idx; lines[i]; i += arg->stride) {
        char buf[256];
        strncpy(buf, lines[i], sizeof(buf)); buf[255] = 0;
        char* tok = strtok(buf, " ");
        std::cout << "[unsafe t" << arg->id << "] tokens:";
        while (tok) {
            std::cout << "{" << tok << "}";
            tok = strtok(nullptr, " ");
        }
        std::cout << "\n";
    }


    pthread_barrier_wait(arg->barrier);
    return nullptr;
}

void* safe_worker(void* a) {
    Arg* arg = (Arg*)a;
    for (int i = arg->start_idx; lines[i]; i += arg->stride) {
        char buf[256];
        strncpy(buf, lines[i], sizeof(buf)); buf[255] = 0;
        char* saveptr = nullptr;
        char* tok = strtok_r(buf, " ", &saveptr);
        std::cout << "[safe t" << arg->id << "] tokens:";
        while (tok) {
            std::cout << "{" << tok << "}";
            tok = strtok_r(nullptr, " ", &saveptr);
        }
        std::cout << "\n";
    }

  
    pthread_barrier_wait(arg->barrier);
    return nullptr;
}

void run_with_barrier(int nthreads) {
    std::cout << "Demostracion con barrera: sincronizacion entre hilos (hilos: " << nthreads << ")\n";

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, nthreads);

    pthread_t t1[nthreads];
    Arg args[nthreads];

    
    auto start = std::chrono::high_resolution_clock::now();

    
    for (int i = 0; i < nthreads; i++) {
        args[i].id = i;
        args[i].start_idx = i;
        args[i].stride = nthreads;
        args[i].barrier = &barrier;
        pthread_create(&t1[i], nullptr, unsafe_worker, &args[i]);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(t1[i], nullptr);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Tiempo con barrera: " << duration.count() << " segundos.\n";

    pthread_barrier_destroy(&barrier);
}

int main() {
    
    for (int nthreads : {2, 4}) {
        run_with_barrier(nthreads);
    }

    return 0;
}
