#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "stdio.h"
/**
* Reduce 算法
*/

#define mBlockNum 256
__global__ void Reduce(unsigned int* A,unsigned int* Out){
    __shared__ int sdata[mBlockNum];
    unsigned int i = threadIdx.x + blockIdx.x* blockDim.x;
    unsigned int tid = threadIdx.x;
    sdata[tid] = A[i];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1 ; s<blockDim.x ; s *= 2) {
        if (tid%(2*s) == 0) {
            sdata[tid] +=sdata[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) {
        Out[blockIdx.x] = sdata[tid];
        __syncthreads();
        printf("Out Value [%d] = %d \n", blockIdx.x,sdata[tid]);
    }
}

int main(int argc , char** argv)
{
    unsigned int num= 100000;
    int size = num* sizeof(unsigned int);
    unsigned int* h_A = (unsigned int*)malloc(size);
    unsigned int* h_Out = (unsigned int*)malloc(size);

    for (int i; i<num;i++) {
        h_A[i] = i+1;
    }

    unsigned int* d_A;
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyKind::cudaMemcpyHostToDevice);


    int blockNum = mBlockNum;
    int gridNum = ( num + blockNum - 1) / mBlockNum;

    
    unsigned int* d_Out;
    int d_num = gridNum* sizeof(unsigned int);
    cudaMalloc(&d_Out, d_num);

    printf("gridnum is =%d \n",gridNum);

    Reduce<<<gridNum,blockNum>>>(d_A,d_Out);

    cudaMemcpy(h_Out, d_Out, d_num, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    unsigned int value = 0;
    for (int i=0;i< gridNum;i++) {
        value += h_Out[i];
    }

    printf("Num is %d\n",value);
    return 0;
}