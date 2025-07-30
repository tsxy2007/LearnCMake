#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>

struct Matrix
{
    Matrix() {}
    float* Data;
    int mWidth;
    int mHeight;
};

#define BLOCK_SIZE 16

__global__ void MatrixMulti(Matrix A,Matrix B,Matrix C)
{
    int row = blockDim.x * blockIdx.x  + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < A.mWidth && col < B.mHeight)
    {    
        float Cvalue = 0;
        for (int i = 0; i < A.mWidth; i++)
        {
            Cvalue += A.Data[row * A.mWidth + i] * B.Data[i * B.mWidth + col];
        }
        C.Data[row * C.mWidth + col] = Cvalue;
    }
    
}

int main(int argc,char** argv)
{
    int w = 3 , h = 3;
    int size = w * h * sizeof(float);
    Matrix h_A;
    Matrix h_B;
    h_A.mWidth = w;
    h_A.mWidth = h;
    h_A.Data = (float*)malloc(size);
    h_B.mHeight = h;
    h_B.mWidth = w;
    h_B.Data = (float*)malloc(size);

    Matrix h_C;
    h_C.mWidth = h_A.mWidth;
    h_C.mHeight = h_B.mHeight;

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
            printf("h_A[%d][%d] = %f",i,j,h_A.Data[Index]);
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
    dim3 dimGrid(h_A.mWidth / dimBlock.x, h_B.mHeight / dimBlock.y);
    MatrixMulti <<<dimGrid, dimBlock >>> (d_A, d_B, d_C);

    cudaMemcpy(h_C.Data, d_C.Data, size,
        cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.Data);
    cudaFree(d_B.Data);
    cudaFree(d_C.Data);
    return 0;
}