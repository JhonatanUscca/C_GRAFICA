#include <iostream>
#include <cuda_runtime.h>

#define N 200  

__global__ void matrixMulTiled(int *A, int *B, int *C, int tileSize) {
    __shared__ int tileA[4][4];
    __shared__ int tileB[4][4];

    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;

    int temp = 0;

    for (int t = 0; t < N; t += tileSize) {
        if(row < N && t + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + (t + threadIdx.x)];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if(t + threadIdx.y < N && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < tileSize; ++k)
            temp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = temp;
}

int main() {
    int size = N * N * sizeof(int);
    int h_A[N*N], h_B[N*N], h_C[N*N];

    for(int i=0; i<N*N; i++){ h_A[i]=1; h_B[i]=2; h_C[i]=0; }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,size); cudaMalloc(&d_B,size); cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int tileSize;
    std::cout << "Ingrese el tamaño del tile (2 o 4): ";
    std::cin >> tileSize;

    dim3 dimBlock(tileSize, tileSize);
    dim3 dimGrid((N+tileSize-1)/tileSize, (N+tileSize-1)/tileSize);

 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, tileSize);
    cudaEventRecord(stop);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Tiempo de ejecucion (ms): " << milliseconds << " ms\n";

    std::cout << "Resultado:\n";
    /*
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++)
            std::cout << h_C[i*N+j] << " ";
        std::cout << std::endl;
    }*/

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
