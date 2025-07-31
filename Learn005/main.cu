#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "stdio.h"
/**
* Reduce 算法
*/

#define mBlockNum 256
__global__ void Reduce(float* A)
{
    int tid = threadIdx.x + blockIdx.x* blockDim.x;
    printf("I'm run %d\n",tid);
}

int main(int argc , char** argv)
{
    int num= 10000;
    int size = num* sizeof(float);
    float* h_A = (float*)malloc(size);

    float* d_A;
    cudaMalloc(&d_A, size);

    int blockNum = mBlockNum;
    int gridNum = ( num + blockNum - 1) / mBlockNum;

    Reduce<<<gridNum,blockNum>>>(d_A);
    cudaDeviceSynchronize();
    return 0;
}