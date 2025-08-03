#include <cuda_runtime.h>
#include <cstdio>

#define BLOCKNUM 32

__global__ void reduce(float* Input, float* Out)
{
	__shared__ float sdata[BLOCKNUM];
	int Index = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	sdata[tid] = Input[Index];
	//printf("input value = %f \n", Input[Index]);
	__syncthreads();

	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if (tid % (2 * i) == 0) {
			sdata[tid] += sdata[tid + i];
			//printf("add value = %f\n", sdata[tid]);
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		Out[blockIdx.x] = sdata[tid];
		//printf("Out Value [%d] = %f \n", blockIdx.x, sdata[tid]);
	}
}

int main(int argc, char* argv)
{
	printf("ÎÒÊÇLearn006!\n");

	int Num = 100;
	int size = Num * sizeof(float);
	float* h_Input = (float*)malloc(size);
	for (int i = 0; i < Num; i++)
	{
		h_Input[i] = i + 1;
	}

	float* d_Input;
	auto err = cudaMalloc(&d_Input, size);
	err = cudaMemcpy(d_Input, h_Input, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	int blockNum = BLOCKNUM;
	int GridNum = (Num + blockNum - 1) / blockNum;

	float* d_Out;
	int Size_Out = GridNum * sizeof(float);
	cudaMalloc(&d_Out, Size_Out);

	reduce << <GridNum, blockNum >> > (d_Input, d_Out);

	float* h_Out = (float*)malloc(Size_Out);
	cudaMemcpy(h_Out, d_Out, Size_Out, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	float Value = 0.f;
	for (int i = 0; i < GridNum; i++)
	{
		Value += h_Out[i];
	}


	printf("The Result is = %f", Value);
	return 0;
}