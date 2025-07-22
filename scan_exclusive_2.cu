#include "timer.h"

#define BLOCK_DIM 1024
#define M (1 << 20)  // 1M elements

__global__ void scan_kernel(float *input, float* output, float* partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // double buffer method
    __shared__ float buffer1_s[BLOCK_DIM];
    __shared__ float buffer2_s[BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;
    if(threadIdx.x == 0){
        inBuffer_s[threadIdx.x] = 0.0f;
    }else{
        inBuffer_s[threadIdx.x] = input[i - 1];
    }
    __syncthreads();

    for(unsigned int stride=1; stride <= BLOCK_DIM/2; stride*=2){
        if(threadIdx.x >=stride){
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        }else{
             outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        float* tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }
    if(threadIdx.x == BLOCK_DIM - 1){
        partialSums[blockIdx.x] = inBuffer_s[threadIdx.x] + input[i];
    }
    output[i] = inBuffer_s[threadIdx.x];

}
__global__ void add_kernel(float* output, float* partialSums, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] += partialSums[blockIdx.x];
    }
}

void scan_gpu_d(float* input_d, float* output_d, unsigned int N){
    Timer timer;

    // configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock -1 )/numElementsPerBlock;

    // allocate partial sums
    startTime(&timer);
    float *partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Partial sums allocation time");

    // call kernel
    startTime(&timer);
    scan_kernel<<<numBlocks, numThreadsPerBlock,numThreadsPerBlock * sizeof(float)>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time");

    // scan partial sums then add
    if(numBlocks > 1){
        // scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

        // Add scanned sums
        add_kernel<<<numBlocks, numThreadsPerBlock>>>(output_d, partialSums_d, N);

    }
    // Free memory
    startTime(&timer);
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}



void exclusive_scan_cpu(const float* input, float* output, unsigned int N) {
    output[0] = 0.0f;
    for (unsigned int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

int main() {
    Timer timer;

    // 分配主机内存
    float* h_input = new float[M];
    float* h_output = new float[M];
    float* h_reference = new float[M];

    // 初始化输入数据
    std::srand(static_cast<unsigned int>(std::time(0)));
    for (unsigned int i = 0; i < M; ++i) {
        h_input[i] = static_cast<float>(std::rand() % 10);
    }

    // 分配 GPU 内存
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, M * sizeof(float));
    cudaMalloc((void**)&d_output, M * sizeof(float));

    // 拷贝输入数据到 GPU
    startTime(&timer);
    cudaMemcpy(d_input, h_input, M * sizeof(float), cudaMemcpyHostToDevice);
    stopTime(&timer);
    printElapsedTime(timer, "Host to Device Memcpy");

    // 调用 GPU scan
    scan_gpu_d(d_input, d_output, M);

    // 拷贝结果回主机
    startTime(&timer);
    cudaMemcpy(h_output, d_output, M * sizeof(float), cudaMemcpyDeviceToHost);
    stopTime(&timer);
    printElapsedTime(timer, "Device to Host Memcpy");

    // 验证结果
    exclusive_scan_cpu(h_input, h_reference, M);

    bool correct = true;
    for (unsigned int i = 0; i < M; ++i) {
        if (fabs(h_output[i] - h_reference[i]) > 1e-3f) {
            std::cerr << "Mismatch at index " << i << ": GPU = "
                      << h_output[i] << ", CPU = " << h_reference[i] << std::endl;
            correct = false;
            break;
        }
    }

    std::cout << (correct ? "Scan result correct ✅" : "Scan result incorrect ❌") << std::endl;

    // 清理资源
    delete[] h_input;
    delete[] h_output;
    delete[] h_reference;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}