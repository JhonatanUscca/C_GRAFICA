#include <iostream>
#include <vector>
#include <string>
#include <pthread.h>
#include <cstring>
#include <semaphore.h>
#include <chrono>  
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
};

sem_t sem; 

void* unsafe_worker(void* a) {
    Arg* arg = (Arg*)a;
    for (int i = arg->start_idx; lines[i]; i += arg->stride) {
        char buf[256];
        strncpy(buf, lines[i], sizeof(buf)); buf[255] = 0;
        char* tok = strtok(buf, " ");
        sem_wait(&sem); 
        std::cout << "[unsafe t" << arg->id << "] tokens:";
        while (tok) {
            std::cout << "{" << tok << "}";
            tok = strtok(nullptr, " ");
        }
        std::cout << "\n";
        sem_post(&sem); 
    }
    return nullptr;
}

void* safe_worker(void* a) {
    Arg* arg = (Arg*)a;
    for (int i = arg->start_idx; lines[i]; i += arg->stride) {
        char buf[256];
        strncpy(buf, lines[i], sizeof(buf)); buf[255] = 0;
        char* saveptr = nullptr;
        char* tok = strtok_r(buf, " ", &saveptr);
        sem_wait(&sem); 
        std::cout << "[safe t" << arg->id << "] tokens:";
        while (tok) {
            std::cout << "{" << tok << "}";
            tok = strtok_r(nullptr, " ", &saveptr);
        }
        std::cout << "\n";
        sem_post(&sem); 
    }
    return nullptr;
}

void run_with_semaphore(int nthreads) {
    std::cout << "Demostracion con semáforo: controlando acceso a std::cout (hilos: " << nthreads << ")\n";

    
    sem_init(&sem, 0, 1);

    pthread_t t1[nthreads];
    Arg args[nthreads];

   
    auto start = std::chrono::high_resolution_clock::now();

   
    for (int i = 0; i < nthreads; i++) {
        args[i].id = i;
        args[i].start_idx = i;
        args[i].stride = nthreads;
        pthread_create(&t1[i], nullptr, unsafe_worker, &args[i]);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(t1[i], nullptr);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Tiempo con semaforo: " << duration.count() << " segundos.\n";

    
    sem_destroy(&sem);
}

int main() {
   
    for (int nthreads : {2, 4}) {
        run_with_semaphore(nthreads);
    }

    return 0;
}
