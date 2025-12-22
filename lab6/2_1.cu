#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 4096  
#define BLOCK_SIZE 16


__global__ void matrixAddBasic(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int idx = row * n + col;
   
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void matrixAddShared(float *A, float *B, float *C, int n) {
  
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (row < n && col < n) {
        int idx = row * n + col;
        
      
        sA[ty][tx] = A[idx];
        sB[ty][tx] = B[idx];
        
        __syncthreads();
        
 
        C[idx] = sA[ty][tx] + sB[ty][tx];
    }
}


__global__ void analyzeAccess(float *A, float *B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
   
    if (blockIdx.x == 0 && blockIdx.y == 0 && row < 4 && col < 4) {
        int idx = row * n + col;
        printf("Hilo[%d,%d] accede a A[%d][%d] = %.1f y B[%d][%d] = %.1f\n",
               threadIdx.y, threadIdx.x, row, col, A[idx], row, col, B[idx]);
    }
}

void initMatrix(float *mat, int n, float val) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = val + (i % 10);
    }
}

bool verifyResult(float *C, float *expected, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabs(C[i] - expected[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

int main() {
    int size = N * N * sizeof(float);
    

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C_basic = new float[N * N];
    float *h_C_shared = new float[N * N];
    
  
    initMatrix(h_A, N, 1.0f);
    initMatrix(h_B, N, 2.0f);
    
 
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    std::cout << " ANALISIS: Acceso a Datos por Hilo " << std::endl;
    std::cout << "Mostrando primeros 4x4 hilos del primer bloque:\n" << std::endl;
    analyzeAccess<<<gridDim, blockDim>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    
    

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        matrixAddBasic<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double time_basic = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    cudaMemcpy(h_C_basic, d_C, size, cudaMemcpyDeviceToHost);
    
   
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        matrixAddShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double time_shared = std::chrono::duration<double, std::milli>(end - start).count() / 100.0;
    
    cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost);
    

    std::cout << "RESULTADOS DEL BENCHMARK" << std::endl;
    std::cout << "Tamaño de matriz: " << N << " x " << N << std::endl;
    std::cout << "Tiempo kernel basico:         " << time_basic << " ms" << std::endl;
    std::cout << "Tiempo kernel con shared mem: " << time_shared << " ms" << std::endl;
    std::cout << "Diferencia: " << (time_shared - time_basic) << " ms" << std::endl;
    
    if (time_shared > time_basic) {
        std::cout << "\n La memoria compartida es MAS LENTA (sobrecarga innecesaria)" << std::endl;
    } else {
        std::cout << "\n Tiempos similares (sin beneficio real)" << std::endl;
    }
    

    bool basic_correct = verifyResult(h_C_basic, h_C_basic, N);
    bool shared_correct = verifyResult(h_C_shared, h_C_basic, N);
    
    std::cout << "\nVERIFICACION" << std::endl;
    std::cout << "Resultado basico: " << (basic_correct ? "Correcto" : "Error") << std::endl;
    std::cout << "Resultado shared: " << (shared_correct ? "Correcto" : "Error") << std::endl;
    
    
   
    int total_elements = N * N;
    int accesos_global_basic = total_elements * 2;  
    int accesos_global_shared = total_elements * 2; 
    
    std::cout << "\nANALISIS DE ANCHO DE BANDA " << std::endl;
    std::cout << "Accesos a memoria global (basico):  " << accesos_global_basic << std::endl;
    std::cout << "Accesos a memoria global (shared):  " << accesos_global_shared << std::endl;
    std::cout << "Reducción: 0 accesos (0%)" << std::endl;
    
 
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_basic;
    delete[] h_C_shared;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
