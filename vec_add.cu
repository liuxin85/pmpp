// compute vector sum h_c = h_a + h_b
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#define N 10

// Compute vector sum C=A + B
// Each thread performs on pair-wise addtion
__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vecAdd(float *A, float *B, float *C, int n)
{

    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Run ceil(n/256) block of 256 threads each
    vecAddKernel<<<ceil(n / 256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize h_A and h_B with some vluaes
    for(int i=0; i<N; i++){
        h_A[i] = i;
        h_B[i] = 2 * i;
    }
    vecAdd(h_A, h_B, h_C, N);
    
    // print the results
    for(int i = 0; i<N; i++){
        printf("%f + %f = %f \n", h_A[i], h_B[i], h_C[i]);
    }

    // free host memeory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
