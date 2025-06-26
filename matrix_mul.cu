#define TILE_WIDTH 16

__global__ void MatrixMulKernel(float* d_M, float* d_N, float* d_P, int Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    

}

int main(){
    return 0;
}