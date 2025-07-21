#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include "timer.h"

#define BLOCK_DIM 8

__global__ void stencil_kernel(float *in, float* out, unsigned int N) {
    // 3D thread and block index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = z * N * N + y * N + x;

    if (x > 0 && x < N-1 && y > 0 && y < N-1 && z > 0 && z < N-1) {
        float result = 0.0f;
        result += in[idx]; // center
        result += in[idx + 1];     // right
        result += in[idx - 1];     // left
        result += in[idx + N];     // down
        result += in[idx - N];     // up
        result += in[idx + N*N];   // front
        result += in[idx - N*N];   // back
        out[idx] = result / 7.0f;
    } else {
        out[idx] = in[idx]; // for simplicity, copy border as-is
    }
}

void stencil_gpu(float *in, float *out, unsigned int N) {
    Timer timer;

    // Allocate GPU memory
    float *in_d, *out_d;
    startTime(&timer);
    cudaMalloc((void**)&in_d, N * N * N * sizeof(float));
    cudaMalloc((void**)&out_d, N * N * N * sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(in_d, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Launch kernel
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / BLOCK_DIM);

    startTime(&timer);
    stencil_kernel<<<gridDim, blockDim>>>(in_d, out_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel execution time");

    // Copy result back to host
    startTime(&timer);
    cudaMemcpy(out, out_d, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy back to CPU time");

    // Cleanup
    cudaFree(in_d);
    cudaFree(out_d);
}

// 初始化输入数据
void init_data(float* data, unsigned int N) {
    for (unsigned int z = 0; z < N; ++z) {
        for (unsigned int y = 0; y < N; ++y) {
            for (unsigned int x = 0; x < N; ++x) {
                unsigned int idx = z * N * N + y * N + x;
                data[idx] = static_cast<float>((x + y + z) % 10);
            }
        }
    }
}

int main() {
    const unsigned int N = 32;  // 可以改为 64 或更大测试
    size_t num_bytes = N * N * N * sizeof(float);

    // 分配主机内存
    float* input = (float*)malloc(num_bytes);
    float* output = (float*)malloc(num_bytes);

    // 初始化输入
    init_data(input, N);

    // 调用 GPU stencil
    stencil_gpu(input, output, N);

    // 输出部分结果
    std::cout << "Sample output (z = N/2):" << std::endl;
    unsigned int z = N / 2;
    for (unsigned int y = 0; y < N; y += 4) {
        for (unsigned int x = 0; x < N; x += 4) {
            unsigned int idx = z * N * N + y * N + x;
            std::cout << output[idx] << "\t";
        }
        std::cout << std::endl;
    }

    // 清理
    free(input);
    free(output);

    return 0;
}