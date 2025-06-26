#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "设备 " << i << ": " << prop.name << std::endl;
        std::cout << "  计算能力 (Compute Capability): " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  多处理器数量(SM): " << prop.multiProcessorCount << std::endl;
        std::cout << "  每个线程块最大线程数: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  全局内存大小: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  共享内存每块: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  每个线程块最大维度: (" 
                  << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " 
                  << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  网格最大尺寸: (" 
                  << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " 
                  << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << std::endl;

        std::cout << "Warp size: "<< prop.warpSize << std::endl;
    }

    return 0;
}
