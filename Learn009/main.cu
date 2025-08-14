#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ostream>
#include "vec3.h"
#include "ray.h"

#define BLOCKNUM 128

// 检查CUDA错误的宏
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__host__ __device__ vec3 color(const ray& r){
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return  vec3(1.0f,1.0f,1.0f) * (1.0-t) +  vec3(0.5,0.7,1.0) * t;
}

__global__ void MakeColor(vec3* OutColor,int width,int height,vec3 lower_left_corner,vec3 horizontal,vec3 vertical,vec3 origin)
{
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;
    if (u < width && v < height) {
        int Index = v * width  +  u;
        ray *r = new ray(origin,lower_left_corner+horizontal * u + vertical * v);
        OutColor[Index].e[0] = color(*r).e[0];
        OutColor[Index].e[1] = color(*r).e[1];
        OutColor[Index].e[2] = color(*r).e[2];
    }
}

auto main() -> int
{
    int nx = 4080;
    int ny = 2160;

    std::ofstream outFile("output.ppm");

    if (!outFile.is_open()) {
        std::cerr<<"can't wirte to file!"<<std::endl;
        return 0;
    }

    int size = nx * ny * sizeof(vec3);
    std::cout<< "size = " << size << "Vec3 size = " << sizeof(vec3) <<std::endl;

    vec3* d_Output;
    cudaMalloc(&d_Output, size);
    outFile<<"P3\n" << nx <<" "<< ny<<"\n255\n";

    vec3 lower_left_corner(-2.f,-1.f,-1.f);
    vec3 horizontal(4.0f,0.f,0.f);
    vec3 vertical(0.f,2.f,0.f);
    vec3 origin(0.f,0.f,0.f);

    dim3 blockDim(BLOCKNUM, 1024 / BLOCKNUM); // 如果是2维 一个block 最大为1024个线程；
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y);
    MakeColor<<<gridDim,blockDim>>>(d_Output,nx,ny,lower_left_corner,horizontal,vertical,origin);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();  // 此时可捕获执行阶段的错误
    printf("Error is %d \n",err);
    vec3* h_Output = (vec3*)malloc(size);
    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);



    for (int j = 0; j<ny; j++) {
        for (int i = 0;i<nx;i++) {
            int Index = j * nx + i;
            vec3 v = h_Output[Index];

            int ir = int(255.99 * v.r());
            int ig = int(255.99 * v.g());
            int ib = int(255.99 * v.b());
            outFile << ir << " " << ig << " " << ib << "\n";
        }
    }
    
    return 0;
}