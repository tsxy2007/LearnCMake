#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ostream>
#include "vec3.h"
#include "ray.h"

#define BLOCKNUM 32

// 检查CUDA错误的宏
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__host__ __device__ bool hit_sphere(const vec3& center, float radius, const ray& r)
 {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc,r.direction());
    float c = dot(oc,oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;  
    return (discriminant > 0);
}

__host__ __device__ bool equals(float a, float b)
{
    return fabs(a - b) < 0.000001;
}

__host__ __device__ vec3 color(const ray& r)
{
    if (hit_sphere(vec3(0,0,-1),0.5,r)){
        return vec3(1,0,0);
    }
    else {  
        vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5 * (unit_direction.y + 1.0);
        return  vec3(1.0f,1.0f,1.0f) * (1.0-t) +  vec3(0.5,0.7,1.0) * t;
    }
}

__global__ void MakeColor(vec3* OutColor,int width,int height)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= width || j >= height) {
        return;
    }

    vec3 lower_left_corner(-2.f,-1.f,-1.f);
    vec3 horizontal(4.0f,0.f,0.f);
    vec3 vertical(0.f,2.f,0.f);
    vec3 origin(0.f,0.f,0.f);

    int Index = j * width  +  i;

    float u = float(i) / float(width);
    float v = float(j) / float(height);
    vec3 direct = (lower_left_corner + horizontal * u + vertical * v - origin);
    ray r(origin,direct);
    auto tColor = color(r);
    OutColor[Index].x = tColor.x;
    OutColor[Index].y = tColor.y;
    OutColor[Index].z = tColor.z;
}

// 将帧缓冲区数据写入PPM文件
void write_ppm(const std::string& filename, const vec3* framebuffer, int width, int height) {
    std::ofstream out(filename);
    out << "P3\n" << width << " " << height << "\n255\n";
    
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            const vec3& c = framebuffer[idx];
            
            int r = static_cast<int>(255.99f * c.x);
            int g = static_cast<int>(255.99f * c.y);
            int b = static_cast<int>(255.99f * c.z);
            
            out << r << " " << g << " " << b << "\n";
        }
    }
    
    out.close();
}

/**
 * @brief 主函数，执行CUDA程序的主要逻辑
 * @return int 程序退出状态码
 */
auto main() -> int
{

    // 定义图像的宽度和高度
    int nx = 1920;
    int ny = 1080;  // 图像宽度（像素）

     // 1. 设置设备（若多GPU，需指定目标设备）
    CHECK(cudaSetDevice(0));  // 使用第0块GPU

    // 2. 查询当前栈大小限制
    size_t currentStackSize;
    CHECK(cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize));
    std::cout << "默认栈大小: " << currentStackSize << " 字节" << std::endl;

    // 3. 设置新的栈大小（例如 64KB）
    size_t newStackSize = 64 * 1024;  // 64KB
    CHECK(cudaDeviceSetLimit(cudaLimitStackSize, newStackSize));
    std::cout << "已设置栈大小: " << newStackSize << " 字节" << std::endl;

    int size = nx * ny * sizeof(vec3);
    std::cout<< "size = " << size << " Vec3 size = " << sizeof(vec3) <<std::endl;

    vec3* d_Output;
    cudaMalloc(&d_Output, size);

   

    dim3 blockDim(BLOCKNUM, 1024 / BLOCKNUM); // 如果是2维 一个block 最大为1024个线程；
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y);
    MakeColor<<<gridDim,blockDim>>>(d_Output,nx,ny);

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();  // 此时可捕获执行阶段的错误
    printf("Error is %d \n",err);
    vec3* h_Output = new vec3[nx * ny];
    cudaMemcpy(h_Output, d_Output, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    write_ppm("output.ppm", h_Output, nx, ny);
    
    return 0;
}