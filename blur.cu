#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define BLUR_SIZE 1   // 宏末尾不要分号！

__global__ void blur_kernel(const unsigned char* __restrict__ image,
                            unsigned char*        blurred,
                            unsigned int          width,
                            unsigned int          height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outRow < height && outCol < width) {
        unsigned int avg = 0;
     

        for (int r = outRow - BLUR_SIZE; r < outRow + BLUR_SIZE +1; ++r)
            for (int c = outCol - BLUR_SIZE; c < outCol + BLUR_SIZE + 1; ++c)
                if (r >= 0 && r < height && c >= 0 && c < width) {
                    avg += image[r * width + c];
                
                }

        blurred[outRow * width + outCol] = (unsigned char)(avg / ((2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)));
    }
}

int main()
{
    const unsigned int width  = 1080;
    const unsigned int height = 1080;
    const size_t       bytes  = width * height * sizeof(unsigned char);

    /* 1. 主机端分配并初始化图像 */
    unsigned char *h_img   = (unsigned char*)malloc(bytes);
    unsigned char *h_blur  = (unsigned char*)malloc(bytes);
    for (size_t i = 0; i < width * height; ++i)
        h_img[i] = rand() & 0xFF;

    /* 2. 设备端分配内存 */
    unsigned char *d_img, *d_blur;
    cudaMalloc(&d_img,  bytes);
    cudaMalloc(&d_blur, bytes);

    /* 3. 拷贝输入数据到 GPU */
    cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice);

    /* 4. 网格/线程块大小 */
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);

    /* 5. 计时并启动 kernel */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    blur_kernel<<<grid, block>>>(d_img, d_blur, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU blur took %.3f ms\n", ms);

    /* 6. 传回结果并简单校验 */
    cudaMemcpy(h_blur, d_blur, bytes, cudaMemcpyDeviceToHost);
    printf("First 10 output pixels: ");
    for (int i = 0; i < 10; ++i) printf("%u ", h_blur[i]);
    printf("\n");

    /* 7. 清理 */
    free(h_img);
    free(h_blur);
    cudaFree(d_img);
    cudaFree(d_blur);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}