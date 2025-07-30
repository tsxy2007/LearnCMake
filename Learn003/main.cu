
#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>

const static int blockSize = 256;

__global__ void AddVec(const float* A,const float* B, float* C ,int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<size)
    {
        C[i] = A[i] - B[i];
    }
    
}

int main(int argc,char** argv)
{
    cudaError_t err = cudaSuccess;
    int num  = 3000000;
    int size = num* sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    for (int i = 0; i < num; i++)
    {
        h_A[i] = (double)rand() / (RAND_MAX +1.0);
        h_B[i] = (double)rand() / (RAND_MAX +1.0);
    }
    float* h_C = (float*)malloc(size);
    for (int i = 0; i < num; i++)
    {
        h_C[i] = h_A[i] - h_B[i];
        // printf("the h_A[%d]+h_B[%d] = h_C[%d] { %f+%f=%f} \n",i,i,i,h_A[i],h_B[i],h_C[i]);
    }
    
    
    float* d_A;
    err = cudaMalloc((void **)&d_A, size);
    float* d_B;
    err = cudaMalloc((void **)&d_B,size);
    float* d_C;
    err = cudaMalloc((void **)&d_C,size);

    err = cudaMemcpy(d_A,h_A, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B,h_B,size,cudaMemcpyKind::cudaMemcpyHostToDevice);

    int blocknum = blockSize;
    int gridnum = (num + blockSize - 1) / (blockSize);
    AddVec<<<gridnum, blocknum>>>(d_A, d_B, d_C, num);

    float* h_C1 = (float*)malloc(size);
    err = cudaMemcpy(h_C1,d_C,size,cudaMemcpyKind::cudaMemcpyDeviceToHost);
    for (int i = 0; i < num; i++)
    {
        if (fabs(h_A[i] - h_B[i] - h_C1[i]) > 1e-5)
        {
            printf("error!!!\n");
            break;
        }
    }
    
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
    printf("第一个程序!!!");
}