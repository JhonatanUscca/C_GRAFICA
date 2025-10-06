#include <iostream>
#include <chrono>  

int main() {
    long long n = 1000000;  
    double factor = 1.0;    
    double sum = 0.0;       
    double pi;             

    auto inicio = std::chrono::high_resolution_clock::now();

    for (long long i = 0; i < n; i++, factor = -factor) {
        sum += factor / (2 * i + 1);    }

   
    pi = 4.0 * sum;

   
    auto fin = std::chrono::high_resolution_clock::now();

  
    std::chrono::duration<double> duracion = fin - inicio;

   
    std::cout << "Aproximacion de PI = " << pi << std::endl;
    std::cout << "Tiempo de ejecucion: " << duracion.count() << " segundos" << std::endl;


    return 0;
}
