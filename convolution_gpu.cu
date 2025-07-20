#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define the kernel size (3x3 in this example)
#define KERNEL_SIZE 3
#define TILE_SIZE 16

// CUDA kernel for 2D convolution
__global__ void convolution2DKernel(float* input, float* output, float* kernel, 
                                   int width, int height, int channels) {
    // Calculate output pixel coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only proceed if within image bounds
    if (col < width && row < height) {
        for (int ch = 0; ch < channels; ch++) {
            float sum = 0.0f;
            
            // Iterate over kernel elements
            for (int ky = -KERNEL_SIZE/2; ky <= KERNEL_SIZE/2; ky++) {
                for (int kx = -KERNEL_SIZE/2; kx <= KERNEL_SIZE/2; kx++) {
                    // Calculate input pixel coordinates
                    int ix = col + kx;
                    int iy = row + ky;
                    
                    // Handle boundary conditions (clamp to edge)
                    ix = max(0, min(ix, width - 1));
                    iy = max(0, min(iy, height - 1));
                    
                    // Get kernel value
                    int kernelIdx = (ky + KERNEL_SIZE/2) * KERNEL_SIZE + (kx + KERNEL_SIZE/2);
                    float kernelValue = kernel[kernelIdx];
                    
                    // Get input value and multiply by kernel value
                    int inputIdx = (iy * width + ix) * channels + ch;
                    float inputValue = input[inputIdx];
                    
                    sum += inputValue * kernelValue;
                }
            }
            
            // Store the result
            int outputIdx = (row * width + col) * channels + ch;
            output[outputIdx] = sum;
        }
    }
}

// Host function to perform convolution with timing
void convolution2D(float* h_input, float* h_output, float* h_kernel, 
                  int width, int height, int channels) {
    // Allocate device memory
    size_t imageSize = width * height * channels * sizeof(float);
    size_t kernelSize = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    
    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSize);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    convolution2DKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, 
                                             width, height, channels);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // Record stop event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate and print elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Execution Time: %.3f ms\n", milliseconds);
    
    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// CPU version for comparison
void convolution2D_cpu(float* input, float* output, float* kernel,
                      int width, int height, int channels) {
    clock_t start = clock();
    
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int ch = 0; ch < channels; ch++) {
                float sum = 0.0f;
                
                for (int ky = -KERNEL_SIZE/2; ky <= KERNEL_SIZE/2; ky++) {
                    for (int kx = -KERNEL_SIZE/2; kx <= KERNEL_SIZE/2; kx++) {
                        int ix = col + kx;
                        int iy = row + ky;
                        
                        ix = max(0, min(ix, width - 1));
                        iy = max(0, min(iy, height - 1));
                        
                        int kernelIdx = (ky + KERNEL_SIZE/2) * KERNEL_SIZE + (kx + KERNEL_SIZE/2);
                        int inputIdx = (iy * width + ix) * channels + ch;
                        
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
                
                int outputIdx = (row * width + col) * channels + ch;
                output[outputIdx] = sum;
            }
        }
    }
    
    clock_t end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Execution Time: %.3f ms\n", cpu_time);
}

int main() {
    // Example image dimensions
    int width = 1024;  // Larger image for meaningful timing
    int height = 1024;
    int channels = 1; // Grayscale image
    
    // Allocate host memory
    size_t imageSize = width * height * channels * sizeof(float);
    size_t kernelSize = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);
    
    float* h_input = (float*)malloc(imageSize);
    float* h_output_gpu = (float*)malloc(imageSize);
    float* h_output_cpu = (float*)malloc(imageSize);
    float* h_kernel = (float*)malloc(kernelSize);
    
    // Initialize input image with random values
    for (int i = 0; i < width * height * channels; i++) {
        h_input[i] = (float)rand() / RAND_MAX; // Random values between 0 and 1
    }
    
    // Initialize kernel (edge detection)
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        h_kernel[i] = -1.0f;
    }
    h_kernel[KERNEL_SIZE * KERNEL_SIZE / 2] = KERNEL_SIZE * KERNEL_SIZE - 1.0f;
    
    printf("Image size: %dx%d, Kernel size: %dx%d\n", width, height, KERNEL_SIZE, KERNEL_SIZE);
    
    // Perform convolution on GPU
    printf("\nGPU Convolution:\n");
    convolution2D(h_input, h_output_gpu, h_kernel, width, height, channels);
    
    // Perform convolution on CPU for comparison
    printf("\nCPU Convolution:\n");
    convolution2D_cpu(h_input, h_output_cpu, h_kernel, width, height, channels);
    
    // Verify results (check first few pixels)
    int errors = 0;
    for (int i = 0; i < 10; i++) {
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5) {
            errors++;
            if (errors < 5) {
                printf("Mismatch at %d: GPU %.6f vs CPU %.6f\n", 
                      i, h_output_gpu[i], h_output_cpu[i]);
            }
        }
    }
    if (errors > 0) {
        printf("Total mismatches in first 10 pixels: %d\n", errors);
    } else {
        printf("GPU and CPU results match for first 10 pixels\n");
    }
    
    // Free host memory
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    free(h_kernel);
    
    return 0;
}