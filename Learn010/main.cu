#include <cuda_runtime.h>  
#include <curand_kernel.h>  
#include <cmath>
#include <iostream>
#include <fstream>

// 检查 CUDA 操作是否成功
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 向量类，用于表示3D空间中的点、方向等
struct vec3 {
    float x, y, z;
    
    __host__ __device__  vec3() : x(0), y(0), z(0) {}
    __host__ __device__  vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__  ~vec3() {}
    
    __host__ __device__  vec3 operator-() const { return vec3(-x, -y, -z); }
    __host__ __device__  vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__  vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__  vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__  vec3 operator*(float s) const { return vec3(x * s, y * s, z * s); }
    __host__ __device__  vec3 operator/(float s) const { return vec3(x / s, y / s, z / s); }
    
    __host__ __device__  vec3& operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__  vec3& operator*=(const vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    __host__ __device__  vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    __host__ __device__  vec3& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }
    
    __host__ __device__  float dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__  vec3 cross(const vec3& v) const { 
        return vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    
    __host__ __device__  vec3 normalize() const {
        float len = sqrt(x*x + y*y + z*z);
        return vec3(x/len, y/len, z/len);
    }
};

// 光线类
struct ray {
    vec3 origin;
    vec3 direction;
    
    __host__ __device__  ray() {}
    __host__ __device__  ray(const vec3& o, const vec3& d) : origin(o), direction(d) {}
    
    __host__ __device__  vec3 at(float t) const {
        return origin + direction * t;
    }
};

// 球体类
struct sphere {
    vec3 center;
    float radius;
    vec3 color;  // 球体颜色
    float albedo; // 反射率
    
    __host__ __device__  sphere() {}
    __host__ __device__  sphere(const vec3& c, float r, const vec3& col, float a) 
        : center(c), radius(r), color(col), albedo(a) {}
    
    // 计算光线与球体的相交
    __host__ __device__  bool hit(const ray& r, float t_min, float t_max, float& t) const {
        vec3 oc = r.origin - center;
        float a = r.direction.dot(r.direction);
        float b = 2.0f * oc.dot(r.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = b * b - 4 * a * c;
        
        if (discriminant < 0) {
            return false; // 无交点
        }
        
        // 计算最近的有效交点
        float sqrt_d = sqrt(discriminant);
        float t1 = (-b - sqrt_d) / (2.0f * a);
        if (t1 < t_max && t1 > t_min) {
            t = t1;
            return true;
        }
        
        float t2 = (-b + sqrt_d) / (2.0f * a);
        if (t2 < t_max && t2 > t_min) {
            t = t2;
            return true;
        }
        
        return false;
    }
};

// 随机数生成器初始化
__global__ void init_rand(curandState* rand_states, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (i >= width || j >= height) return;
    
    int idx = j * width + i;
    curand_init(1984 + idx, 0, 0, &rand_states[idx]);

    // printf("init_rand = rand_states[%d]",idx);
}

// 生成随机向量
__device__  vec3 random_vec(curandState* rand_state) {
    return vec3(
        curand_uniform(rand_state),
        curand_uniform(rand_state),
        curand_uniform(rand_state)
    );
}

// 生成单位球内的随机向量
__device__  vec3 random_in_unit_sphere(curandState* rand_state) {
    while (true) {
        vec3 p = random_vec(rand_state) * 2.0f - vec3(1, 1, 1);
        if (p.dot(p) < 1.0f) {
            return p;
        }
    }
}

// 计算光线颜色
__device__  vec3 ray_color(const ray& r, const sphere* world, int world_size, curandState* rand_state, int depth) {
    // 递归深度限制，防止无限递归
    if (depth <= 0) {
        return vec3(0, 0, 0);
    }
    
    float t_min = 0.001f; // 避免自相交
    float t_max = 1e8f;
    float t_hit = t_max;
    int hit_index = -1;
    
    // 检查与所有球体的相交
    for (int i = 0; i < world_size; i++) {
        float t;
        if (world[i].hit(r, t_min, t_hit, t)) {
            t_hit = t;
            hit_index = i;
        }
    }
    
    // 如果有交点，计算反射光线
    if (hit_index != -1) {
        vec3 p = r.at(t_hit);
        vec3 normal = (p - world[hit_index].center).normalize();
        // 确保法向量方向与入射光线方向一致
        vec3 outward_normal = normal;
        if (r.direction.dot(normal) > 0) {
            outward_normal = -normal;
        }
        
        // 生成漫反射方向
        vec3 scatter_dir = outward_normal + random_in_unit_sphere(rand_state).normalize();
        ray scattered = ray(p, scatter_dir);
        
        // 递归计算反射光线颜色
        return  world[hit_index].color * world[hit_index].albedo * 
           ray_color(scattered, world, world_size, rand_state, depth - 1);
    }
    
    // 背景色 - 渐变天空
    vec3 unit_dir = r.direction.normalize();
    float t = 0.5f * (unit_dir.y + 1.0f);
    return vec3(1.0f, 1.0f, 1.0f) * (1.0f - t) + vec3(0.5f, 0.7f, 1.0f) * t;
}

// 渲染内核函数
__global__ void render(vec3* framebuffer, int width, int height, 
                      const sphere* world, int world_size, 
                      curandState* rand_states, int samples_per_pixel, int max_depth) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // 列
    int j = threadIdx.y + blockIdx.y * blockDim.y; // 行
    
    // printf("render i = [%d];j=[%d];width=[%d],height=[%d]\n",i,j,width,height);
    if (i >= width || j >= height) return;
    // printf("render -----------1\n");
    int idx = j * width + i;
    curandState local_rand_state = rand_states[idx];
    // printf("render -----------2\n");
    vec3 color(0, 0, 0);
    
    // 相机参数
    vec3 lookfrom(0, 0, 0);
    vec3 lookat(0, 0, -1);
    vec3 vup(0, 1, 0);
    // float dist_to_focus = 10.0f;
    // float aperture = 0.1f;
    // printf("render -----------3\n");
    // 相机坐标系
    vec3 w = (lookfrom - lookat).normalize();
    vec3 u = vup.cross(w).normalize();
    vec3 v = w.cross(u);
    
    float aspect_ratio = (float)width / height;
    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    
    vec3 horizontal = u * viewport_width;
    vec3 vertical = v * viewport_height;
    vec3 lower_left_corner = lookfrom - horizontal / 2.0f - vertical / 2.0f - w;
    // printf("render -----------4 samples_per_pixel = %d\n",samples_per_pixel);
    // 抗锯齿：每个像素采样多个光线
    for (int s = 0; s < samples_per_pixel; s++) {
        float u_coord = (i + curand_uniform(&local_rand_state)) / (width - 1.0f);
        float v_coord = (j + curand_uniform(&local_rand_state)) / (height - 1.0f);
        
        ray r(lookfrom, lower_left_corner + horizontal * u_coord + vertical * v_coord - lookfrom);
        color += ray_color(r, world, world_size, &local_rand_state, max_depth);
    }
    // printf("render -----------5\n");
    // 平均采样结果
    color /= samples_per_pixel;
    
    // 伽马校正
    color.x = sqrt(color.x);
    color.y = sqrt(color.y);
    color.z = sqrt(color.z);
    // printf(" i = [%d] , j = [%d] ; Color = [%f][%f][%f]\n",i,j,color.x,color.y,color.z);
    
    framebuffer[idx] = color;
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

// 核函数，接收一个整数参数
__global__ void kernel(int value) {
    // 在设备端打印接收的参数值
    printf("Device received: %d\n", value);
}


int main() {

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

    // 图像参数
    const int width = 800;
    const int height = 600;
    const int samples_per_pixel = 10; // 每个像素的采样数，影响抗锯齿效果
    const int max_depth = 10; // 光线最大递归深度
    
    // 分配帧缓冲区
    vec3* framebuffer = new vec3[width * height];
    vec3* d_framebuffer;
    cudaMalloc(&d_framebuffer, width * height * sizeof(vec3));
    
    // 创建场景 - 多个球体
    const int world_size = 4;
    sphere* world = new sphere[world_size];
    sphere* d_world;
    
    // 场景内容：几个不同颜色和位置的球体
    world[0] = sphere(vec3(0, 0, -5), 1.0f, vec3(0.8f, 0.3f, 0.3f), 0.8f); // 红色球
    world[1] = sphere(vec3(2, 0, -5), 1.0f, vec3(0.3f, 0.8f, 0.3f), 0.8f); // 绿色球
    world[2] = sphere(vec3(-2, 0, -5), 1.0f, vec3(0.3f, 0.3f, 0.8f), 0.8f); // 蓝色球
    world[3] = sphere(vec3(0, -1001, -5), 1000.0f, vec3(0.8f, 0.8f, 0.0f), 0.8f); // 黄色地面
    
    cudaMalloc(&d_world, world_size * sizeof(sphere));
    cudaMemcpy(d_world, world, world_size * sizeof(sphere), cudaMemcpyHostToDevice);
    
    // 初始化随机数生成器
    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, width * height * sizeof(curandState));
    
    dim3 block_dim(32, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                 (height + block_dim.y - 1) / block_dim.y);
    
    kernel<<<1,1>>>(100);
    cudaDeviceSynchronize();

    init_rand<<<grid_dim, block_dim>>>(d_rand_states, width, height);
    cudaDeviceSynchronize();
    
    // 执行渲染
    std::cout << "开始渲染..." << std::endl;
    render<<<grid_dim, block_dim>>>(d_framebuffer, width, height, d_world, world_size, 
                                   d_rand_states, samples_per_pixel, max_depth);
    cudaDeviceSynchronize();
    
    auto err = cudaGetLastError();  // 此时可捕获执行阶段的错误
    printf("Error is %d \n",err); // 打印错误码

    // 将结果从设备内存拷贝到主机内存
    cudaMemcpy(framebuffer, d_framebuffer, width * height * sizeof(vec3), cudaMemcpyDeviceToHost);
    
    // 写入文件
    write_ppm("cuda_raytrace.ppm", framebuffer, width, height);
    std::cout << "渲染完成，图像已保存为 cuda_raytrace.ppm" << std::endl;
    
    // 释放内存
    delete[] framebuffer;
    framebuffer = nullptr;
    delete[] framebuffer;
    delete[] world;
    cudaFree(d_framebuffer);
    cudaFree(d_world);
    cudaFree(d_rand_states);
    
    return 0;
}
