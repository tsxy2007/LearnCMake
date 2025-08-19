#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ostream>
#include <sys/select.h>
#include "hitable.h"
#include "vec3.h"
#include "ray.h"
#include "hitable_list.h"
#include "sphere.h"
#include "camera.h"
#include <curand_kernel.h>  
#include "material.h"

#define BLOCKNUM 16

// 检查CUDA错误的宏
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__host__ __device__ bool equals(float a, float b)
{
    return fabs(a - b) < 0.000001;
}

__host__ __device__ float hit_sphere(const vec3& center, float radius, const ray& r)
 {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc,r.direction());
    float c = dot(oc,oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;  
    if (discriminant < 0) 
    {
        return -1.f;
    }
    else 
    {
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}

__host__ __device__ vec3 color(const ray& r)
{
    vec3 center = vec3(0,0,-1);
    float t = hit_sphere(center,0.5,r);
    if ( t > 0.f )
    {
        vec3 N = unit_vector(r.point_at_parameter(t) - center);
        return vec3(N.x + 1 , N.y + 1 , N.z + 1) * 0.5;
    }
    else 
    {  
        vec3 unit_direction = unit_vector(r.direction());
        t = 0.5 * (unit_direction.y + 1.0);
        return  vec3(1.0f,1.0f,1.0f) * (1.0-t) +  vec3(0.5,0.7,1.0) * t;
    }
}

// __device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
//     vec3 p;
//     do {
//         p = RANDVEC3 * 2.0f - vec3(1,1,1);
//     } while (p.squared_length() >= 1.0f);
//     return p;
// }

__device__ vec3 color_hit(curandState* LocalRandState,const ray& r, const hitable_list& world)
{

    ray cur_ray = r;
    float cur_attenuation = 1.0f;
    for(int i = 0; i < 4; i++) 
    {
        hit_record rec;
        if (world.hit(cur_ray, 0.001f, MAXFLOAT, rec)) 
        {
            vec3 target = rec.p + rec.normal + random_in_unit_sphere(LocalRandState);
            cur_attenuation *= 0.5f;
            cur_ray = ray(rec.p, target-rec.p);
        }
        else 
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = (unit_direction.y + 1.0f) * 0.5f;
            vec3 c = vec3(1.0, 1.0, 1.0) * (1.0f-t) + vec3(0.5, 0.7, 1.0) * t;
            return  c * cur_attenuation;
        }
    }
    return vec3(0.0,0.0,0.0);
}

// 随机数生成器初始化
__global__ void init_rand(curandState* rand_states, int width, int height,int sample_pix) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    
    if (i >= width || j >= height || z >= sample_pix) return;
    
    int idx = z * width * height + j * width + i;
    curand_init(1984 + idx, 0, 0, &rand_states[idx]);

    // printf("init_rand = rand_states[%d]",idx);
}

__global__ void MakeColor(sphere** input_list,int size ,curandState* input_rand_states, vec3* OutColor,int width,int height,int samples_per_pixel)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= width || j >= height || z >= samples_per_pixel) 
    {
        return;
    }
    int Index = z * width * height + j * width  +  i;
    curandState LocalRandState = input_rand_states[Index];
    camera cam;
    hitable_list hlist (input_list,size);
    OutColor[Index] = color_hit(&LocalRandState,cam.get_ray(float(i + curand_uniform(&LocalRandState)) / float(width - 1),float(j + curand_uniform(&LocalRandState)) / float(height - 1)),hlist);
}

// 抗锯齿
__global__ void AntiAliasing(vec3* Input_Color,int Sample_pix,int width,int height,vec3* Out_Color)
{
    int u = blockDim.x * blockIdx.x + threadIdx.x;
    int v = blockDim.y * blockIdx.y + threadIdx.y;

    if ( u >= width || v >= height) 
    {
        return;
    }
    int ColorIndex = u + v * width;

    // printf("AntiAliasing = u = [%d] v = [%d] \n",u,v);

    for (int i = 0; i < Sample_pix ; i++) 
    {
        int Index = i * width * height + v * width + u;
        Out_Color[ColorIndex] += Input_Color[Index];
    }

    Out_Color[ColorIndex] /= (float)Sample_pix;
    Out_Color[ColorIndex] = vec3(sqrt(Out_Color[ColorIndex][0]),sqrt(Out_Color[ColorIndex][1]),sqrt(Out_Color[ColorIndex][2]));
}

// 创建元素
__global__ void create_world(sphere **d_list, hitable **d_world, camera **d_camera) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0,0,-1), 0.5,
                               new lambertian(vec3(0.8, 0.3, 0.3)));
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
                               new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1,0,-1), 0.5,
                               new metal(vec3(0.8, 0.6, 0.2), 1.0));
        d_list[3] = new sphere(vec3(-1,0,-1), 0.5,
                               new metal(vec3(0.8, 0.8, 0.8), 0.3));
        *d_world  = new hitable_list(d_list,4);
        *d_camera = new camera();
    }
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
    const int nx = 2000;
    const int ny = 1000;  // 图像宽度（像素）
    const int samples_per_pixel = 10;

     // 1. 设置设备（若多GPU，需指定目标设备）
    CHECK(cudaSetDevice(0));  // 使用第0块GPU

    // 2. 查询当前栈大小限制
    size_t currentStackSize;
    CHECK(cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize));
    std::cout << "默认栈大小: " << currentStackSize << " 字节" << std::endl;

    // 3. 设置新的栈大小（例如 64KB）
    size_t newStackSize = 256 * 1024;  // 64KB
    CHECK(cudaDeviceSetLimit(cudaLimitStackSize, newStackSize));
    std::cout << "已设置栈大小: " << newStackSize << " 字节" << std::endl;


    dim3 blockDim(BLOCKNUM, 16, 4); // 如果是2维 一个block 最大为1024个线程；
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                 (ny + blockDim.y - 1) / blockDim.y,
                (samples_per_pixel + blockDim.z -1) / blockDim.z);

    // 初始化随机数生成器
    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, nx * ny * samples_per_pixel * sizeof(curandState));

    init_rand<<<gridDim, blockDim>>>(d_rand_states, nx, ny,samples_per_pixel);
    cudaDeviceSynchronize();


    int size = nx * ny * samples_per_pixel * sizeof(vec3);
    std::cout<< "size = " << size << " Vec3 size = " << sizeof(vec3) <<std::endl;

    vec3* d_Output;
    cudaMalloc(&d_Output, size);

    // 4. 创建sphere
    const int num = 4;
    sphere **d_list;
    cudaMalloc((void **)&d_list, num*sizeof(sphere *));
    hitable **d_world;
    cudaMalloc((void **)&d_world, sizeof(hitable *));
    camera **d_camera;
    cudaMalloc((void **)&d_camera, sizeof(camera *));
    create_world<<<1,1>>>(d_list,d_world,d_camera);
    

    // 5. 渲染主程序
    MakeColor<<<gridDim,blockDim>>>(d_list, num,d_rand_states, d_Output,nx,ny,samples_per_pixel);
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();  // 此时可捕获执行阶段的错误
    printf("Error is %d \n",err);


    // 6. 抗锯齿
    vec3* d_Output_Color;
    int rsize = nx * ny * sizeof(vec3);
    cudaMalloc(&d_Output_Color, rsize);
	dim3 bNum(32,32);
	dim3 gNum((nx + bNum.x -1) / bNum.x ,(ny + bNum.y - 1) / bNum.y);
    AntiAliasing<<<gNum,bNum>>>(d_Output, samples_per_pixel, nx, ny,d_Output_Color);

    // 7. 输出到图片
    vec3* h_Output = new vec3[nx * ny];
    cudaMemcpy(h_Output, d_Output_Color, rsize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    write_ppm("output.ppm", h_Output, nx, ny);
    
    return 0;
}