#include <iostream>
#include <pthread.h>
#include <chrono>   

const long long n = 1000000;   
double sum = 0.0;              
int flag = 0;                  
int thread_count;              


void* Thread_sum(void* rank) {
    long my_rank = (long) rank;  
    double factor, my_sum = 0.0;
    long long i;

    long long my_n = n / thread_count;         
    long long my_first_i = my_n * my_rank;     
    long long my_last_i = my_first_i + my_n;   
  
    if (my_first_i % 2 == 0)
        factor = 1.0;
    else
        factor = -1.0;

 
    for (i = my_first_i; i < my_last_i; i++, factor = -factor)
        my_sum += factor / (2 * i + 1);


    while (flag != my_rank);

    sum += my_sum;


    flag = (flag + 1) % thread_count;

    return NULL;
}

int main() {
    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64}; 
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int t = 0; t < num_tests; t++) {
        thread_count = thread_counts[t];
        pthread_t threads[thread_count];
        sum = 0.0;
        flag = 0;

        std::cout << "\nEjecutando con " << thread_count << " hilos " << std::endl;

      
        auto start = std::chrono::high_resolution_clock::now();

      
        for (long thread = 0; thread < thread_count; thread++) {
            pthread_create(&threads[thread], NULL, Thread_sum, (void*)thread);
        }

        
        for (int thread = 0; thread < thread_count; thread++) {
            pthread_join(threads[thread], NULL);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;


        double pi_approx = 4.0 * sum;

        std::cout << "Aproximacion de PI = " << pi_approx << std::endl;
        std::cout << "Tiempo de ejecucion: " << elapsed.count() << " segundos" << std::endl;
    }

    return 0;
}
