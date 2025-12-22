#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Kernel CORRECTO

__global__ void matrixMulCorrect(float* d_M, float* d_N, float* d_P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
 

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    
  
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
      
        Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        __syncthreads(); 
        
    }
    
    d_P[Row*Width + Col] = Pvalue;
}

// Kernel SIN primera 

__global__ void matrixMulNoSync1(float* d_M, float* d_N, float* d_P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
    
        Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
        
        // __syncthreads(); // OMITIDO
    
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        __syncthreads(); 
    }
    
    d_P[Row*Width + Col] = Pvalue;
}

// Kernel SIN segunda 
__global__ void matrixMulNoSync2(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        // __syncthreads(); // OMITIDO - PROBLEMA 2
      
    }
    
    P[Row * Width + Col] = Pvalue;
}

// Kernel SIN ninguna 
__global__ void matrixMulNoSyncBoth(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        
        // __syncthreads(); // OMITIDO
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        // __syncthreads(); // OMITIDO
    }
    
    P[Row * Width + Col] = Pvalue;
}



// errores CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}



void initMatrix(float* mat, int size, float value) {
    for (int i = 0; i < size * size; i++) {
        mat[i] = value;
    }
}




float compareMatrices(float* A, float* B, int size) {
    float maxDiff = 0.0f;
    for (int i = 0; i < size * size; i++) {
        float diff = fabs(A[i] - B[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    return maxDiff;
}

int main() {
    const int Width = 64; 
    const int size = Width * Width;
    const int bytes = size * sizeof(float);
    

    
    float *h_M, *h_N, *h_P_correct, *h_P_nosync1, *h_P_nosync2, *h_P_nosync_both;
    h_M = new float[size];
    h_N = new float[size];
    h_P_correct = new float[size];
    h_P_nosync1 = new float[size];
    h_P_nosync2 = new float[size];
    h_P_nosync_both = new float[size];
    
    

    initMatrix(h_M, Width, 1.0f);
    initMatrix(h_N, Width, 2.0f);
    
   
    
    float *d_M, *d_N, *d_P;
    CHECK_CUDA(cudaMalloc(&d_M, bytes));
    CHECK_CUDA(cudaMalloc(&d_N, bytes));
    CHECK_CUDA(cudaMalloc(&d_P, bytes));
    
    
    CHECK_CUDA(cudaMemcpy(d_M, h_M, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_N, h_N, bytes, cudaMemcpyHostToDevice));
    
   
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);
    
    std::cout << "Demostracion de __syncthreads() en CUDA " << std::endl;
    std::cout << "Tamano de matriz: " << Width << "x" << Width << std::endl;
    std::cout << "TILE_WIDTH: " << TILE_WIDTH << std::endl << std::endl;
    
    // 1. Kernel CORRECTO
    matrixMulCorrect<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_P_correct, d_P, bytes, cudaMemcpyDeviceToHost));
    std::cout << "1. Kernel CORRECTO (con ambos __syncthreads):" << std::endl;
    std::cout << "   Resultado esperado en cada elemento: " << Width * 2.0f << std::endl;
    std::cout << "   P[0][0] = " << h_P_correct[0] << std::endl << std::endl;
    
    // 2. Sin primera sincronización
    CHECK_CUDA(cudaMemset(d_P, 0, bytes));
    matrixMulNoSync1<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_P_nosync1, d_P, bytes, cudaMemcpyDeviceToHost));
    float diff1 = compareMatrices(h_P_correct, h_P_nosync1, Width);
    std::cout << "2. Sin PRIMER __syncthreads (lectura no sincronizada):" << std::endl;
    std::cout << "   P[0][0] = " << h_P_nosync1[0] << std::endl;
    std::cout << "   Diferencia maxima: " << diff1 << std::endl;
    std::cout << "   Estado: " << (diff1 > 0.01f ? "INCORRECTO " : "Correcto (por suerte)") << std::endl << std::endl;
    
    // 3. Sin segunda sincronizacion
    CHECK_CUDA(cudaMemset(d_P, 0, bytes));
    matrixMulNoSync2<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_P_nosync2, d_P, bytes, cudaMemcpyDeviceToHost));
    float diff2 = compareMatrices(h_P_correct, h_P_nosync2, Width);
    std::cout << "3. Sin SEGUNDO __syncthreads (escritura no sincronizada):" << std::endl;
    std::cout << "   P[0][0] = " << h_P_nosync2[0] << std::endl;
    std::cout << "   Diferencia maxima: " << diff2 << std::endl;
    std::cout << "   Estado: " << (diff2 > 0.01f ? "INCORRECTO " : "Correcto (por suerte)") << std::endl << std::endl;
    
    // 4. Sin ninguna sincronizacion
    CHECK_CUDA(cudaMemset(d_P, 0, bytes));
    matrixMulNoSyncBoth<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_P_nosync_both, d_P, bytes, cudaMemcpyDeviceToHost));
    float diff_both = compareMatrices(h_P_correct, h_P_nosync_both, Width);
    std::cout << "4. Sin NINGUN __syncthreads (caos total):" << std::endl;
    std::cout << "   P[0][0] = " << h_P_nosync_both[0] << std::endl;
    std::cout << "   Diferencia maxima: " << diff_both << std::endl;
    std::cout << "   Estado: " << (diff_both > 0.01f ? "INCORRECTO " : "Correcto (muy improbable)") << std::endl << std::endl;
    

    
   
    delete[] h_M;
    delete[] h_N;
    delete[] h_P_correct;
    delete[] h_P_nosync1;
    delete[] h_P_nosync2;
    delete[] h_P_nosync_both;
    
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_N));
    CHECK_CUDA(cudaFree(d_P));
    
    return 0;
}
