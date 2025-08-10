#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
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
        uchar4 texel = tex2D<uchar4>(texObj, u + 0.5f, (height - 1 - v) + 0.5f);
        
        // 简单处理：复制到输出（可以在这里添加图像处理逻辑）
        output[y * width + x] = texel;
    }
}

int main() {
    // 1. 使用stb_image加载图像
    int width, height, channels;
    unsigned char* h_image_data = stbi_load("./texture_image.png", &width, &height, &channels, 4);
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
    // unsigned char* h_output = (unsigned char*)malloc(width * height * 4);
    // CHECK(cudaMemcpy(h_output, d_output, width * height * 4, cudaMemcpyDeviceToHost));
    // stbi_write_png("output_image.png", width, height, 4, h_output, width * 4);

    // 8. 清理资源
    stbi_image_free(h_image_data);          // 释放stb_image加载的图像数据
    // free(h_output);                       // 释放主机输出内存（如果使用）
    CHECK(cudaDestroyTextureObject(texObj));// 销毁纹理对象
    CHECK(cudaFreeArray(d_array));          // 释放CUDA数组
    CHECK(cudaFree(d_output));              // 释放设备输出内存

    std::cout << "Texture processing completed successfully" << std::endl;
    return 0;
}
