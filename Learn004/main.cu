#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
/**
* 求矩阵的乘积
*/
struct Matrix
{
    Matrix() {}
    float* Data;
    int mWidth;
    int mHeight;
};

#define BLOCK_SIZE 16

__global__ void MatrixMulti(float* A, int A_W, int A_H,float* B, int B_W, int B_H, int C_W, float* C )
{
    int row = blockDim.x * blockIdx.x  + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < A_W && col < B_H)
    {    
        float Cvalue = 0;
        for (int i = 0; i < A_W; i++)
        {
            Cvalue += A[row * A_W + i] * B[i * B_W + col];
        }
        C[row * C_W + col] = Cvalue;
    }
    
}

int main(int argc,char** argv)
{
    int w = 30 , h = 30;
    int size = w * h * sizeof(float);
    Matrix h_A;
    Matrix h_B;
    h_A.mWidth = w;
    h_A.mHeight = h;
    size = h_A.mWidth * h_A.mHeight * sizeof(float);
    h_A.Data = (float*)malloc(size);

    h_B.mHeight = h;
    h_B.mWidth = w;    
    size = h_B.mWidth * h_B.mHeight * sizeof(float);
    h_B.Data = (float*)malloc(size);

    Matrix h_C;
    h_C.mWidth = h_A.mWidth;
    h_C.mHeight = h_B.mHeight;
    size = h_C.mWidth * h_C.mHeight * sizeof(float);
    h_C.Data = (float*)malloc(size);

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            int Index = i * w  + j;
            if (i == j)
            {
                h_A.Data[Index] = 1.f;
            }
            else
            {
                h_A.Data[Index] = 0.f;
            }
            h_B.Data[Index] = (double)rand() / (RAND_MAX +1.0);
            printf("h_B[%d][%d] = %f; ",i,j,h_B.Data[Index]);
        }
        printf("\n");
    }
    Matrix d_A;
    d_A.mWidth = h_A.mWidth;
    d_A.mHeight = h_A.mHeight;
    size = h_A.mWidth * h_A.mHeight * sizeof(float);
    cudaMalloc(&d_A.Data, size);
    cudaMemcpy(d_A.Data, h_A.Data, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.mWidth = h_B.mWidth;
    d_B.mHeight = h_B.mHeight;
    size = h_B.mWidth * h_B.mHeight * sizeof(float);
    cudaMalloc(&d_B.Data, size);
    cudaMemcpy(d_B.Data, h_B.Data, size, cudaMemcpyHostToDevice);

    
    // Allocate C in device memory
    Matrix d_C;
    d_C.mWidth = h_C.mWidth; d_C.mHeight = h_C.mHeight;
    size = h_C.mWidth * h_C.mHeight * sizeof(float);
    cudaMalloc(&d_C.Data, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h_A.mWidth + dimBlock.x -1 )/ dimBlock.x, (h_B.mHeight + dimBlock.y - 1)/ dimBlock.y);
    MatrixMulti <<<dimGrid, dimBlock >>> (d_A.Data,d_A.mWidth,d_A.mHeight, d_B.Data,d_B.mWidth,d_B.mHeight,d_C.mWidth, d_C.Data);

    cudaMemcpy(h_C.Data, d_C.Data, size,
        cudaMemcpyDeviceToHost);

    printf("-----------------------------------------------------------------------------\n");
    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        { 
            int Index = i * w  + j;
            printf("h_C[%d][%d] = %f; ",i,j,h_C.Data[Index]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A.Data);
    cudaFree(d_B.Data);
    cudaFree(d_C.Data);
    
    return 0;
}