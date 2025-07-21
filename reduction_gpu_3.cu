#include "timer.h"

#define BLOCK_DIM 1024

__global__ void reduce_kernel(float* input, float* partialSums, unsigned int N){
    unsigned int segment = blockIdx.x* blockDim.x *2;
    unsigned int i = segment + threadIdx.x;

    // shared memory
    __shared__ float input_s[BLOCK_DIM];
    input_s[threadIdx.x] = input[i] + input[i + BLOCK_DIM];
    __syncthreads();

    for(unsigned int stride=BLOCK_DIM/2; stride > 0; stride /=2){
        if(threadIdx.x < stride){
          input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        partialSums[blockIdx.x] = input_s[threadIdx.x];
    }
}

float reduce_gpu(float* input, unsigned int N){
    Timer timer;

    // allocate memory
    startTime(&timer);
    float *input_d;
    cudaMalloc((void**)&input_d, N*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // copy data to gpu
    startTime(&timer);
    cudaMemcpy(input_d, input, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");

    // Allocate partial sums
    startTime(&timer);
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    float* partialSums = (float*)malloc(numBlocks*sizeof(float));
    float* partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Partial sum allocation time");

    // call kernel;
    startTime(&timer);
    reduce_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "kernel time");

    // copy data from GPU
    startTime(&timer);
    cudaMemcpy(partialSums, partialSums_d, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    // Reduce partial sums on CPU
    startTime(&timer);
    float sum = 0.0f;
    for(unsigned int i = 0; i < numBlocks; ++i){
        sum += partialSums[i];
    }
    stopTime(&timer);
    printElapsedTime(timer, "Reduce partial sums on host time");

    // Free memeory
    startTime(&timer);
    cudaFree(input_d);
    free(partialSums);
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
    stopTime(&timer);

    return sum;
}

int main() {
    const unsigned int N = 1 << 20;  // 1048576 个元素
    float* input = new float[N];

    // 初始化数据
    srand(static_cast<unsigned int>(time(0)));
    for (unsigned int i = 0; i < N; ++i) {
        input[i] = 1.0f;  // 也可以用 rand() / (float)RAND_MAX
    }

    // 调用 GPU reduce 函数
    float sum = reduce_gpu(input, N);

    std::cout << "Final reduced sum: " << sum << std::endl;

    delete[] input;
    return 0;
}