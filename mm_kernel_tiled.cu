#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>

#define N 1024
#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* A, float* B, float* C, unsigned int n) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for(unsigned int tile=0; tile < n/TILE_DIM; ++tile){
        A_s[threadIdx.y][threadIdx.x] = A[row*n + tile*TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM + threadIdx.y) *n + col];
        __syncthreads();

        for(unsigned int i = 0; i < TILE_DIM; ++i){
            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();

    }
    C[row*n + col] = sum;
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // 分配主机内存
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // 初始化矩阵数据
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 分配设备内存
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置 block 和 grid 尺寸
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // 创建 CUDA event 用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 启动 kernel 并计时
    cudaEventRecord(start);

    // use std::chrono 测量cpu时间
    auto cpu_start = std::chrono::high_resolution_clock::now();

    mm_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Kernel time: " << ms << " ms" << std::endl;

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 简单验证（可选）
    /*
    float sum = 0;
    for (int i = 0; i < N; ++i)
        sum += h_C[i * N + i];
    std::cout << "Sum of diagonal: " << sum << std::endl;
    */

    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
