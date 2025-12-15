#include <stdio.h>
#include <cuda_runtime.h>


//PARTE B:UN elemento

__global__ void matrixAddKernel_OneElementPerThread(float* C, float* A, float* B, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}



// PARTE C:UNA FILA
__global__ void matrixAddKernel_OneRowPerThread(float* C, float* A, float* B, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n) {
        for (int col = 0; col < n; col++) {
            int idx = row * n + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// PARTE D: UNA COLUMNA
__global__ void matrixAddKernel_OneColumnPerThread(float* C, float* A, float* B, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (col < n) {
        for (int row = 0; row < n; row++) {
            int idx = row * n + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}


// PARTE A: Host stub function
void matrixAddition(float* h_C, float* h_A, float* h_B, int n, int kernelType = 1) {
    size_t size = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);



    if (kernelType == 1) {
        dim3 blockDim(16, 16);
        dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                     (n + blockDim.y - 1) / blockDim.y);

        matrixAddKernel_OneElementPerThread<<<gridDim, blockDim>>>(d_C, d_A, d_B, n);

    } else if (kernelType == 2) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        matrixAddKernel_OneRowPerThread<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, n);

    } else if (kernelType == 3) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        matrixAddKernel_OneColumnPerThread<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, n);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Tiempo de ejecución del kernel %d: %.4f ms\n", kernelType, elapsedTime);





    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



// Función para inicializar matrices
void initializeMatrix(float* matrix, int n, float value) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = value;
    }
}


// Función para imprimir matrices
void printMatrix(float* matrix, int n, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}




int main() {
    int n = 4;  
    size_t size = n * n * sizeof(float);
    
    
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    
    
    initializeMatrix(h_A, n, 1.0f);
    initializeMatrix(h_B, n, 2.0f);
    
    printf("Prueba con matriz %dx%d \n\n", n, n);
    printMatrix(h_A, n, "Matriz A");
    printMatrix(h_B, n, "Matriz B");
    
   
    printf("---- Kernel B: Un elemento por thread\n");
    matrixAddition(h_C, h_A, h_B, n, 1);
    printMatrix(h_C, n, "Resultado C");
    
    printf("---- Kernel C: Una fila por thread\n");
    matrixAddition(h_C, h_A, h_B, n, 2);
    printMatrix(h_C, n, "Resultado C");
    
    printf("---- Kernel D: Una columna por thread\n");
    matrixAddition(h_C, h_A, h_B, n, 3);
    printMatrix(h_C, n, "Resultado C");
    
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
