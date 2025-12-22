#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 8  

__global__ void BlockTranspose(float* A_elements, float* B_elements, int width, int height) {
    __shared__ float blockA[BLOCK_SIZE][BLOCK_SIZE];

    int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (xIndex < width && yIndex < height) {
        int index_in = yIndex * width + xIndex;
        blockA[threadIdx.y][threadIdx.x] = A_elements[index_in];
    }

    __syncthreads(); 

    xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x; 
    yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (xIndex < height && yIndex < width) {
        int index_out = yIndex * height + xIndex;
        B_elements[index_out] = blockA[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int width = 32;
    const int height = 32;

    size_t size = width * height * sizeof(float);

    float* h_A = new float[width * height];
    float* h_B = new float[width * height];

 
    for (int i = 0; i < width * height; ++i) {
        h_A[i] = static_cast<float>(i);
    }

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    
    BlockTranspose<<<grid, block>>>(d_A, d_B, width, height);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);


    std::cout << "Matriz transpuesta:" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << h_B[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    delete[] h_A;
    delete[] h_B;

    return 0;
}
