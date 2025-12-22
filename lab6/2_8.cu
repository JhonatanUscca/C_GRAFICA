#include <stdio.h>
#include <cuda_runtime.h>

#define N 8
#define T 4  


__global__ void matMulNoTiling(float *M, float *N_mat, float *P, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
       
        for (int k = 0; k < width; k++) {
            sum += M[row * width + k] * N_mat[k * width + col];
        }
        P[row * width + col] = sum;
    }
}


__global__ void matMulTiling(float *M, float *N_mat, float *P, int width) {
    __shared__ float Mds[T][T];
    __shared__ float Nds[T][T];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * T + ty;
    int col = bx * T + tx;
    
    float sum = 0.0f;
    

    for (int ph = 0; ph < width/T; ph++) {
       
        if (row < width && (ph * T + tx) < width)
            Mds[ty][tx] = M[row * width + ph * T + tx];
        else
            Mds[ty][tx] = 0.0f;
            
        if (col < width && (ph * T + ty) < width)
            Nds[ty][tx] = N_mat[(ph * T + ty) * width + col];
        else
            Nds[ty][tx] = 0.0f;
        
        __syncthreads();
        
    
        for (int k = 0; k < T; k++) {
            sum += Mds[ty][k] * Nds[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < width && col < width)
        P[row * width + col] = sum;
}


void initMatrix(float *mat, int size) {
    for (int i = 0; i < size * size; i++) {
        mat[i] = (float)(rand() % 10);
    }
}


void printMatrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%.1f ", mat[i * size + j]);
        }
        printf("\n");
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_M, *h_N, *h_P_noTiling, *h_P_tiling;
    float *d_M, *d_N, *d_P;
    

    h_M = (float*)malloc(size);
    h_N = (float*)malloc(size);
    h_P_noTiling = (float*)malloc(size);
    h_P_tiling = (float*)malloc(size);
    
   
    initMatrix(h_M, N);
    initMatrix(h_N, N);
    
    printf("ANALISIS DE ACCESOS A MEMORIA GLOBAL\n\n");
    printf("Dimensiones de matriz: %d x %d\n", N, N);
    printf("Tamaño de tile: %d x %d\n\n", T, T);
    
   
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);
    
    
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    
 
    printf("A) SIN TILING:\n");
    printf("   - Cada elemento se accede desde memoria global: N = %d veces\n", N);
    printf("   - Total de accesos para un elemento: %d\n", N);
    printf("   - Total de accesos para toda la matriz: %d x %d = %d\n\n", N*N, N, N*N*N);
    
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 
                 (N + dimBlock.y - 1) / dimBlock.y);
    
    matMulNoTiling<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, N);
    cudaMemcpy(h_P_noTiling, d_P, size, cudaMemcpyDeviceToHost);
  
    printf("B) CON TILING (%dx%d):\n", T, T);
    printf("   - Cada elemento se accede desde memoria global: N/T = %d/%d = %d veces\n", 
           N, T, N/T);
    printf("   - Luego se reutiliza T = %d veces desde shared memory\n", T);
    printf("   - Total de accesos a memoria global por elemento: %d\n", N/T);
    printf("   - Total de accesos para toda la matriz: %d x %d = %d\n\n", N*N, N/T, N*N*N/T);
    
    dim3 dimBlockTile(T, T);
    dim3 dimGridTile((N + T - 1) / T, (N + T - 1) / T);
    
    matMulTiling<<<dimGridTile, dimBlockTile>>>(d_M, d_N, d_P, N);
    cudaMemcpy(h_P_tiling, d_P, size, cudaMemcpyDeviceToHost);
    
  
    printf("REDUCCIÓN DE ACCESOS A MEMORIA GLOBAL:\n");
    printf("   Factor de reduccion: %dx\n", T);
    printf("   Accesos sin tiling: %d\n", N*N*N);
    printf("   Accesos con tiling: %d\n", N*N*N/T);
    printf("   Accesos evitados: %d (%.1f%%)\n\n", 
           N*N*N - N*N*N/T, 
           100.0 * (N*N*N - N*N*N/T) / (float)(N*N*N));
    

    bool equal = true;
    for (int i = 0; i < N*N; i++) {
        if (fabs(h_P_noTiling[i] - h_P_tiling[i]) > 0.01f) {
            equal = false;
            break;
        }
    }
    
    printf("Verificacion: %s\n", equal ? " Ambos metodos producen el mismo resultado" : " Error en calculo");
 
    if (N <= 8) {
        printf("\nMatriz M:\n");
        printMatrix(h_M, N);
        printf("\nMatriz N:\n");
        printMatrix(h_N, N);
        printf("\nResultado P:\n");
        printMatrix(h_P_tiling, N);
    }
    

    free(h_M); free(h_N); free(h_P_noTiling); free(h_P_tiling);
    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    
    return 0;
}
