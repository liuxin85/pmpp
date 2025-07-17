#include <stdio.h>
#define BLUR_SIZE 1

__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){
    unsigned int outRow = blockIdx.y *blockDim.y + threadIdx.y;
    unsigned int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if(outRow < height && outCol < width){
        unsigned int average = 0;
        for(int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow){
            for(int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1; ++inCol){
                // 检查边界
             if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width){
                 average += image[inRow*width + inCol];
            }
            }

        }
        blurred[outRow*width + outCol] =(unsigned char) (average/((2*BLUR_SIZE + 1)*(2*BLUR_SIZE + 1)));

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