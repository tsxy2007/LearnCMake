#include <cstdio>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <cuda_runtime.h>

// 检查CUDA错误的宏
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int getCudaCores(int major, int minor, int mpCount);

// 核函数：使用纹理采样图像并输出到结果
__global__ void processTextureKernel(uchar4* output, cudaTextureObject_t texObj, int width, int height) {
    // 计算线程坐标
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // 纹理坐标是浮点数，范围通常是[0, width)和[0, height)
        float u = static_cast<float>(x);
        float v = static_cast<float>(y);
        
        // 从纹理采样（注意：stb_image加载的图像是上下翻转的，这里v坐标做了翻转）
        uchar4 texel = tex2D<uchar4>(texObj, u + 0.5f,   v + 0.5f);
         unsigned char red = texel.x;
        unsigned char green = texel.y;
        unsigned char blue = texel.z;
        unsigned char alpha = texel.w;

        uchar4 newColor = make_uchar4(1-red, 1-green, 1-blue, alpha);
        // 简单处理：复制到输出（可以在这里添加图像处理逻辑）
        output[y * width + x] = newColor;
    }
}

// 打印设备信息（兼容新版本CUDA）
void printDeviceInfo() {
    int deviceCount;
    CHECK(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "CUDA设备数量: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "\n设备 " << i << ": " << prop.name << std::endl;
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  全局内存: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        
        // 替代totalTextureMem的信息 - 纹理相关属性
        std::cout << "  纹理相关属性:" << std::endl;
        std::cout << "    1D纹理最大尺寸: " << prop.maxTexture1D << std::endl;
        std::cout << "    2D纹理最大尺寸: " << prop.maxTexture2D[0] << "x" << prop.maxTexture2D[1] << std::endl;
        std::cout << "    3D纹理最大尺寸: " << prop.maxTexture3D[0] << "x" << prop.maxTexture3D[1] << "x" << prop.maxTexture3D[2] << std::endl;
        std::cout << "    每层纹理最大尺寸: " << prop.maxTexture2DLayered[0] << "x" << prop.maxTexture2DLayered[1] << "x" << prop.maxTexture2DLayered[2] << std::endl;
        // std::cout << "    纹理参考数限制: " << prop.maxTextureReferences << std::endl;
        
        // 其他有用的设备属性
        std::cout << "  多处理器数量: " << prop.multiProcessorCount << std::endl;
        std::cout << "  每个多处理器的CUDA核心数: " << getCudaCores(prop.major, prop.minor, prop.multiProcessorCount) << std::endl;
        std::cout << "  最大线程块尺寸: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  最大网格尺寸: " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << "x" << prop.maxGridSize[2] << std::endl;

        std::cout << "每个 block 最大线程数: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "grid 最大维度 (x, y, z): " 
              << prop.maxGridSize[0] << ", " 
              << prop.maxGridSize[1] << ", " 
              << prop.maxGridSize[2] << std::endl;
        std::cout << "block 最大维度 (x, y, z): " 
              << prop.maxThreadsDim[0] << ", " 
              << prop.maxThreadsDim[1] << ", " 
              << prop.maxThreadsDim[2] << std::endl;
    }
    std::cout << std::endl;
}

// 计算CUDA核心数量（不同架构核心数不同）
int getCudaCores(int major, int minor, int mpCount) {
    // 不同计算能力对应的每个多处理器的核心数
    switch (major) {
        case 1: return mpCount * 8;    // SM 1.x
        case 2: return mpCount * 32;   // SM 2.x
        case 3: return mpCount * 192;  // SM 3.x
        case 5: return mpCount * 128;  // SM 5.x
        case 6: 
            if (minor == 0) return mpCount * 64;   // SM 6.0
            else return mpCount * 128;             // SM 6.1-6.2
        case 7: 
            if (minor == 0) return mpCount * 64;   // SM 7.0
            else return mpCount * 128;             // SM 7.2+
        case 8: return mpCount * 128;  // SM 8.x
        case 9: return mpCount * 128;  // SM 9.x
        default: return mpCount * 128; // 默认值，适用于新架构
    }
}

int main() 
{
    // 打印机器信息
    printDeviceInfo();
    // 1. 使用stb_image加载图像
    int width, height, channels;
    unsigned char* h_image_data = stbi_load("/mnt/e/Projectes/LearnCMake/Learn008/texture_image.png", &width, &height, &channels, 4);
    if (!h_image_data) {
        std::cerr << "Failed to load image file!" << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << width << "x" << height << ", channels: " << channels << std::endl;

    // 2. 分配设备输出内存
    uchar4* d_output;
    CHECK(cudaMalloc(&d_output, width * height * sizeof(uchar4)));

    // 3. 创建CUDA数组并复制图像数据
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* d_array;
    CHECK(cudaMallocArray(&d_array, &channelDesc, width, height));
    
    // 复制主机图像数据到CUDA数组
    CHECK(cudaMemcpyToArray(d_array, 0, 0, h_image_data, 
                          width * height * sizeof(uchar4), 
                          cudaMemcpyHostToDevice));

    // 4. 配置纹理资源和参数
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;    // x方向寻址模式
    texDesc.addressMode[1] = cudaAddressModeClamp;    // y方向寻址模式
    texDesc.filterMode = cudaFilterModePoint;         // 点过滤（不插值）
    texDesc.readMode = cudaReadModeElementType;       // 按原始类型读取
    texDesc.normalizedCoords = false;                 // 不使用归一化坐标

    // 5. 创建纹理对象
    cudaTextureObject_t texObj;
    CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // 6. 启动核函数处理纹理
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    processTextureKernel<<<gridDim, blockDim>>>(d_output, texObj, width, height);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 7. （可选）将结果复制回主机并保存（需要stb_image_write）
    unsigned char* h_output = (unsigned char*)malloc(width * height * 4);
    CHECK(cudaMemcpy(h_output, d_output, width * height * 4, cudaMemcpyDeviceToHost));

    int success = stbi_write_png("/mnt/e/Projectes/LearnCMake/Learn008/output.png", width, height, 4, h_output, width * 4);
    if (!success) {
        std::cerr << "保存图像失败!" << std::endl;
        return 1;
    }

    // 8. 清理资源
    stbi_image_free(h_image_data);          // 释放stb_image加载的图像数据
    // free(h_output);                       // 释放主机输出内存（如果使用）
    CHECK(cudaDestroyTextureObject(texObj));// 销毁纹理对象
    CHECK(cudaFreeArray(d_array));          // 释放CUDA数组
    CHECK(cudaFree(d_output));              // 释放设备输出内存

    std::cout << "Texture processing completed successfully" << std::endl;
    return 0;
}
