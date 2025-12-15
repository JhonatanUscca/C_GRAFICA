#include <cuda_runtime.h>
#include <iostream>


__global__ void matVecMultKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        float sum = 0.0f;
        
        

        for (int j = 0; j < n; j++) {
            sum += B[i * n + j] * C[j];
        }
        A[i] = sum;
    }
}


void matVecMult(float* A, float* B, float* C, int n) {

    int matrixSize = n * n * sizeof(float);
    int vectorSize = n * sizeof(float);
    

    float *d_A, *d_B, *d_C;
    
   
    cudaMalloc(&d_A, vectorSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, vectorSize);
    

    cudaMemcpy(d_B, B, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, vectorSize, cudaMemcpyHostToDevice);
    
   
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    

    matVecMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
  
    cudaMemcpy(A, d_A, vectorSize, cudaMemcpyDeviceToHost);
   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    int n = 4; 
    
    float* A = new float[n];
    float* B = new float[n * n];
    float* C = new float[n];
    
    std::cout << "Matriz B:" << std::endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] = i + j + 1;
            std::cout << B[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nVector C:" << std::endl;
    for (int i = 0; i < n; i++) {
        C[i] = i + 1;
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;
    

    matVecMult(A, B, C, n);
    
   
    std::cout << "\nVector A (resultado):" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
    
    
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}
