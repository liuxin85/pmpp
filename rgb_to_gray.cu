#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void rgb2gray_kernel(unsigned char* red, unsigned char* green, unsigned char* blue, 
                               unsigned char* gray, unsigned int width, unsigned int height) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < height && col < width) {
        unsigned int i = row * width + col;
        gray[i] = red[i] * 0.3f + green[i] * 0.6f + blue[i] * 0.1f;
    }
}

void rgb2gray_gpu(unsigned char* red, unsigned char* green, unsigned char* blue, 
                 unsigned char* gray, unsigned int width, unsigned int height) {
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaEvent_t start, stop;
    float gpu_time = 0.0f;
    
    // 创建CUDA事件
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start, 0);
    
    // 分配设备内存
    cudaError_t err;
    err = cudaMalloc((void**)&red_d, width * height * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate red_d (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaMalloc((void**)&green_d, width * height * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate green_d (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(red_d);
        return;
    }
    
    err = cudaMalloc((void**)&blue_d, width * height * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate blue_d (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(red_d);
        cudaFree(green_d);
        return;
    }
    
    err = cudaMalloc((void**)&gray_d, width * height * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate gray_d (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(red_d);
        cudaFree(green_d);
        cudaFree(blue_d);
        return;
    }
    
    // 拷贝数据到设备
    err = cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy red to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(red_d);
        cudaFree(green_d);
        cudaFree(blue_d);
        cudaFree(gray_d);
        return;
    }
    
    err = cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy green to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(red_d);
        cudaFree(green_d);
        cudaFree(blue_d);
        cudaFree(gray_d);
        return;
    }
    
    err = cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy blue to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(red_d);
        cudaFree(green_d);
        cudaFree(blue_d);
        cudaFree(gray_d);
        return;
    }
    
    // 设置网格和块尺寸
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 
                   (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    
    // 记录内核启动前时间
    cudaEventRecord(start, 0);
    
    // 启动内核
    rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);
    
    // 记录内核完成时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 计算内核执行时间
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Kernel execution time: %.3f ms\n", gpu_time);
    
    // 检查内核执行是否有错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch rgb2gray_kernel (error code %s)!\n", cudaGetErrorString(err));
    }
    
    // 将结果拷贝回主机
    err = cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy gray from device (error code %s)!\n", cudaGetErrorString(err));
    }
    
    // 记录总处理时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Total GPU processing time: %.3f ms\n", gpu_time);
    
    // 释放设备内存
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
    
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 示例测试函数
int main() {
    const unsigned int width = 1024;
    const unsigned int height = 768;
    
    // 分配主机内存
    unsigned char *red = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *green = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *blue = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned char *gray = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    
    // 初始化测试数据
    for (unsigned int i = 0; i < width * height; i++) {
        red[i] = i % 256;
        green[i] = (i + 85) % 256;
        blue[i] = (i + 170) % 256;
    }
    
    // 调用GPU函数
    printf("Starting GPU processing...\n");
    rgb2gray_gpu(red, green, blue, gray, width, height);
    printf("GPU processing completed.\n");
    
    // 验证前10个像素的结果
    printf("First 10 pixels results:\n");
    for (int i = 0; i < 10; i++) {
        printf("Pixel %d: R=%d, G=%d, B=%d -> Gray=%d\n", 
               i, red[i], green[i], blue[i], gray[i]);
    }
    
    // 释放主机内存
    free(red);
    free(green);
    free(blue);
    free(gray);
    
    return 0;
}