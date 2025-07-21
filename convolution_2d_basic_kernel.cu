#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// 卷积核
__global__ void convolution_2D_basic_kernel(float *N, float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outRow >= height || outCol >= width) return;
    
    float Pvalue = 0.0f;
    int filter_size = 2 * r + 1;
    
    for (int fRow = 0; fRow < filter_size; fRow++) {
        for (int fCol = 0; fCol < filter_size; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * filter_size + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

// 初始化矩阵
void initMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            matrix[i * width + j] = rand() % 10; // 随机0-9的值
        }
    }
}

// 打印矩阵
void printMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    // 设置矩阵和滤波器尺寸
    const int width = 5;
    const int height = 5;
    const int r = 1; // 滤波器半径
    const int filter_size = 2 * r + 1; // 3x3滤波器

    // 分配主机内存
    float *h_N = (float *)malloc(width * height * sizeof(float));
    float *h_F = (float *)malloc(filter_size * filter_size * sizeof(float));
    float *h_P = (float *)malloc(width * height * sizeof(float));

    // 初始化输入矩阵和滤波器
    initMatrix(h_N, width, height);
    initMatrix(h_F, filter_size, filter_size);

    // 打印输入
    printf("Input Matrix N:\n");
    printMatrix(h_N, width, height);
    printf("\nFilter F:\n");
    printMatrix(h_F, filter_size, filter_size);

    // 分配设备内存
    float *d_N, *d_F, *d_P;
    cudaMalloc(&d_N, width * height * sizeof(float));
    cudaMalloc(&d_F, filter_size * filter_size * sizeof(float));
    cudaMalloc(&d_P, width * height * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_N, h_N, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);

    // 调用内核
    convolution_2D_basic_kernel<<<gridSize, blockSize>>>(d_N, d_F, d_P, r, width, height);

    // 拷贝结果回主机
    cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("\nOutput Matrix P:\n");
    printMatrix(h_P, width, height);

    // 释放内存
    free(h_N);
    free(h_F);
    free(h_P);
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);

    return 0;
}