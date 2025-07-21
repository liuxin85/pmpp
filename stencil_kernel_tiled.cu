#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include "timer.h"

#define BLOCK_DIM 8
#define IN_BLOCK_DIM BLOCK_DIM
#define OUT_BLOCK_DIM (IN_BLOCK_DIM - 2)

__global__ void stencil_kernel_tiled(float* in, float* out, unsigned int N) {
    // Global coordinates
    int gx = blockIdx.x * OUT_BLOCK_DIM + threadIdx.x - 1;
    int gy = blockIdx.y * OUT_BLOCK_DIM + threadIdx.y - 1;
    int gz = blockIdx.z * OUT_BLOCK_DIM + threadIdx.z - 1;

    // Shared memory tile (BLOCK_DIM = 8, 所以 10x10x10 tile 以容纳halo)
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM][BLOCK_DIM];

    // Clamp to boundary (handle global memory access safely)
    int x = max(0, min((int)(N - 1), gx));
    int y = max(0, min((int)(N - 1), gy));
    int z = max(0, min((int)(N - 1), gz));

    // Load data into shared memory
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = in[z * N * N + y * N + x];

    __syncthreads();

    // Only compute on valid output threads (avoid halo)
    if (threadIdx.x >= 1 && threadIdx.x < BLOCK_DIM - 1 &&
        threadIdx.y >= 1 && threadIdx.y < BLOCK_DIM - 1 &&
        threadIdx.z >= 1 && threadIdx.z < BLOCK_DIM - 1 &&
        gx > 0 && gx < N-1 && gy > 0 && gy < N-1 && gz > 0 && gz < N-1) {

        float result = 0.0f;
        result += tile[threadIdx.z][threadIdx.y][threadIdx.x];
        result += tile[threadIdx.z][threadIdx.y][threadIdx.x + 1]; // right
        result += tile[threadIdx.z][threadIdx.y][threadIdx.x - 1]; // left
        result += tile[threadIdx.z][threadIdx.y + 1][threadIdx.x]; // down
        result += tile[threadIdx.z][threadIdx.y - 1][threadIdx.x]; // up
        result += tile[threadIdx.z + 1][threadIdx.y][threadIdx.x]; // front
        result += tile[threadIdx.z - 1][threadIdx.y][threadIdx.x]; // back

        out[gz * N * N + gy * N + gx] = result / 7.0f;
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
    dim3 gridDim((N + BLOCK_DIM - 1) / OUT_BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / OUT_BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / OUT_BLOCK_DIM);

    startTime(&timer);
    stencil_kernel_tiled<<<gridDim, blockDim>>>(in_d, out_d, N);
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