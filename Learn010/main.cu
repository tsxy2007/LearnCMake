/**
 * @file    main.cu
 * @brief   CUDA 光线追踪器 — 基于 GPU 并行的漫反射路径追踪
 *
 * 本程序使用 CUDA 在 GPU 上并行渲染一个包含多个球体的 3D 场景。
 * 每个像素由一个 CUDA 线程独立计算，通过随机采样实现：
 *   - 抗锯齿（每像素多条光线，jittered sampling）
 *   - 软阴影 / 环境光遮蔽（Lambertian 漫反射）
 *   - 间接光照（光线多次弹射，迭代式路径追踪）
 *
 * 关键技术点：
 *   - 迭代式路径追踪（避免 GPU 递归的栈溢出 & Windows TDR 超时）
 *   - 球坐标法随机采样（无分支，避免 warp divergence）
 *   - 所有 CUDA API 调用均通过 CHECK 宏做错误检查
 *
 * 输出格式：PPM（P3 文本格式），可用 Photoshop/GIMP/IrfanView 打开
 */

#include <cuda_runtime.h>    // CUDA Runtime API（cudaMalloc, cudaMemcpy, kernel 启动等）
#include <curand_kernel.h>   // CUDA 设备端随机数生成（curandState, curand_init, curand_uniform）
#include <cmath>             // sqrt, sin, cos, acos, cbrt
#include <cstdio>            // printf
#include <iostream>          // std::cout, std::cerr
#include <fstream>           // std::ofstream（PPM 文件写入）

#ifdef _WIN32
#include <windows.h>         // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

// ============================================================================
// CUDA 错误检查宏
// ============================================================================

/**
 * @brief 对所有 CUDA API 调用做同步错误检查，失败时打印错误信息并退出
 *
 * 用法：CHECK(cudaMalloc(&ptr, size));
 *        CHECK(cudaDeviceSynchronize());
 *
 * 注意：kernel 启动（<<<>>>）不返回 cudaError_t，需在同步后单独调用
 *       CHECK(cudaGetLastError()) 来捕获异步错误。
 */
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// 数学常量
// ============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846  ///< 圆周率 π（CUDA 标准头文件中未定义 M_PI）
#endif

// ============================================================================
// 数据结构定义（同时可用于 host 和 device）
// ============================================================================

/**
 * @brief 3D 向量结构体 — 用于表示坐标、方向、颜色
 *
 * 所有成员函数均标记 __host__ __device__，表示可在 CPU 和 GPU 端调用。
 * 在 GPU 端这很重要：kernel 中调用的任何函数都必须是 __device__。
 */
struct vec3 {
    float x, y, z;

    /// 默认构造函数 — 零向量
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    /// 带参构造函数
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ ~vec3() {}

    // ---- 算术运算符 ----

    __host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }
    __host__ __device__ vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ vec3 operator*(float s) const { return vec3(x * s, y * s, z * s); }
    __host__ __device__ vec3 operator/(float s) const { return vec3(x / s, y / s, z / s); }

    // ---- 复合赋值运算符 ----

    __host__ __device__ vec3& operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ vec3& operator*=(const vec3& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
    __host__ __device__ vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    __host__ __device__ vec3& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

    // ---- 向量运算 ----

    /// 点积（内积）— dot(a,b) = |a||b|cosθ，用于计算投影、夹角余弦
    __host__ __device__ float dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    /// 叉积（外积）— cross(a,b) 得到垂直于 a 和 b 的向量，用于构建正交坐标系
    __host__ __device__ vec3 cross(const vec3& v) const {
        return vec3(
            y * v.z - z * v.y,   // 左手定则：x = ay*bz - az*by
            z * v.x - x * v.z,   //           y = az*bx - ax*bz
            x * v.y - y * v.x    //           z = ax*by - ay*bx
        );
    }

    /// 归一化 — 返回同方向的单位向量（长度 = 1）
    /// @warning 零向量调用会导致除零 → NaN
    __host__ __device__ vec3 normalize() const {
        float len = sqrt(x*x + y*y + z*z);
        return vec3(x/len, y/len, z/len);
    }
};

/**
 * @brief 光线结构体 — 由原点 + 方向定义
 *
 * r(t) = origin + direction * t
 * 当 t > 0 时，光线沿 direction 正向传播
 */
struct ray {
    vec3 origin;     ///< 光线起点
    vec3 direction;  ///< 光线方向（不要求单位长度，但归一化可简化部分计算）

    __host__ __device__ ray() {}
    __host__ __device__ ray(const vec3& o, const vec3& d) : origin(o), direction(d) {}

    /// 计算光线上参数 t 处的点坐标：P(t) = origin + direction * t
    __host__ __device__ vec3 at(float t) const {
        return origin + direction * t;
    }
};

/**
 * @brief 球体结构体 — 场景中的基本几何体
 *
 * 采用最简单的漫反射材质模型：
 *   - color:  基础色（RGB）
 *   - albedo: 反射率（0=全吸收, 1=全反射），控制每次弹射的能量衰减
 */
struct sphere {
    vec3 center;    ///< 球心位置
    float radius;   ///< 球体半径
    vec3 color;     ///< 表面颜色（RGB，分量范围 [0,1]）
    float albedo;   ///< 反射率 [0,1]，控制光线弹射时的能量保留比例

    __host__ __device__ sphere() {}
    __host__ __device__ sphere(const vec3& c, float r, const vec3& col, float a)
        : center(c), radius(r), color(col), albedo(a) {}

    /**
     * @brief 光线-球体相交测试（解析法求解一元二次方程）
     *
     * 推导：
     *   光线方程：P(t) = O + D*t
     *   球体方程：|P - C|² = R²
     *   联立：|O + D*t - C|² = R²
     *   令 OC = O - C：
     *     |OC + D*t|² = R²
     *     |D|²*t² + 2(OC·D)*t + |OC|² - R² = 0    ← 标准一元二次方程 at² + bt + c = 0
     *   其中 a = D·D, b = 2*OC·D, c = OC·OC - R²
     *   判别式 Δ = b² - 4ac
     *     Δ < 0  → 无交点
     *     Δ ≥ 0  → t = (-b ± √Δ) / (2a)，取 t_min < t < t_max 的较小解
     *
     * @param r      入射光线
     * @param t_min  交点距离下限（避免自相交，设为 0.001）
     * @param t_max  交点距离上限（初始为无穷大 / FLT_MAX）
     * @param t      [out] 命中距离
     * @return true  找到有效交点
     * @return false 无交点或交点不在 [t_min, t_max] 范围内
     */
    __host__ __device__ bool hit(const ray& r, float t_min, float t_max, float& t) const {
        vec3 oc = r.origin - center;                // OC 向量：光线起点 → 球心
        float a = r.direction.dot(r.direction);     // a = |D|²
        float b = 2.0f * oc.dot(r.direction);      // b = 2*(OC·D)
        float c = oc.dot(oc) - radius * radius;     // c = |OC|² - R²
        float discriminant = b * b - 4 * a * c;     // Δ = b² - 4ac

        if (discriminant < 0) {
            return false;                           // 判别式 < 0：光线与球体不相交
        }

        // 先测试较小的根（光线最先打到的点），不行再测试较大的根
        float sqrt_d = sqrt(discriminant);

        float t1 = (-b - sqrt_d) / (2.0f * a);      // 较小的根
        if (t1 < t_max && t1 > t_min) {
            t = t1;
            return true;
        }

        float t2 = (-b + sqrt_d) / (2.0f * a);      // 较大的根
        if (t2 < t_max && t2 > t_min) {
            t = t2;
            return true;
        }

        return false;                               // 两个交点都不在有效范围内
    }
};

// ============================================================================
// 设备端（GPU）函数
// ============================================================================

/**
 * @brief 初始化每个像素的随机数生成器状态（CUDA kernel）
 *
 * 每个像素对应一个独立的 curandState，使用像素索引作为种子。
 * 这样每个像素的随机序列相互独立，保证并行渲染的可重复性。
 *
 * 线程组织：2D grid × 2D block，每个线程负责一个像素。
 *
 * @param rand_states 设备端 curandState 数组指针
 * @param width       图像宽度
 * @param height      图像高度
 */
__global__ void init_rand(curandState* rand_states, int width, int height) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;   // 像素列坐标
    int j = threadIdx.y + blockIdx.y * blockDim.y;   // 像素行坐标

    if (i >= width || j >= height) return;           // 边界外线程直接返回

    int idx = j * width + i;                         // 像素一维索引
    curand_init(1984 + idx,   // 种子（不同像素不同种子 → 独立序列）
                0,             // subsequence（高级用法，置 0）
                0,             // offset（高级用法，置 0）
                &rand_states[idx]);
}

/**
 * @brief 生成单位球体内的均匀随机点（球坐标法）
 *
 * ⚡ GPU 优化要点：
 *   - 传统方法（拒绝采样）：在 [-1,1]³ 立方体内随机采样，丢弃球外的点
 *     概率约 52%，每个线程循环次数不同 → warp divergence → 性能下降
 *   - 本方法（球坐标 + 立方根半径）：固定 3 次随机数 + 若干数学运算
 *     所有线程执行路径完全相同 → 零 warp divergence → 最优 GPU 性能
 *
 * 数学推导：
 *   均匀体积分布要求半径 r 的概率密度 ∝ r²
 *   → 累积分布函数 CDF(r) = r³
 *   → 逆变换采样：r = cbrt(uniform_random)
 *
 *   方位角 θ ∈ [0, 2π)：θ = 2π * uniform
 *   极角   φ ∈ [0, π)：φ = acos(2*uniform - 1)   (均匀球面分布)
 *
 * @param rand_state 线程的随机数状态
 * @return 单位球体内均匀分布的随机 3D 点
 */
__device__ vec3 random_in_unit_sphere(curandState* rand_state) {
    // 方位角：绕 Y 轴旋转角度，[0, 2π) 均匀分布
    float theta = 2.0f * M_PI * curand_uniform(rand_state);
    // 极角：与 Y 轴的夹角，cosφ 在 [-1, 1] 均匀分布 → 球面均匀
    float phi = acosf(2.0f * curand_uniform(rand_state) - 1.0f);
    // 半径：立方根变换保证体积内均匀分布（r³ 分布 → CDF 逆变换）
    float r = cbrtf(curand_uniform(rand_state));

    float sin_phi = sinf(phi);
    return vec3(
        r * sin_phi * cosf(theta),   // x = r * sinφ * cosθ
        r * sin_phi * sinf(theta),   // y = r * sinφ * sinθ
        r * cosf(phi)                // z = r * cosφ
    );
}

/**
 * @brief 计算光线的颜色（迭代式路径追踪，GPU 设备函数）
 *
 * 算法概述 — 路径追踪（Path Tracing）：
 *   从相机出发的光线在场景中弹射，每次命中物体后：
 *     1. 计算命中点的表面法向量
 *     2. 在法向量方向的半球内随机采样新的散射方向（Lambertian 漫反射）
 *     3. 用物体的颜色 × 反射率 衰减累积的光能
 *     4. 继续追踪散射光线
 *   如果光线最终未命中任何物体，则采样背景渐变（模拟天空）。
 *
 * 设计决策 — 为什么用迭代而非递归：
 *   - GPU 递归会消耗大量栈空间（CUDA 每线程默认栈仅 1KB）
 *   - 递归函数调用在 GPU 上性能较差（寄存器溢出）
 *   - Windows WDDM 有 TDR（~2 秒 GPU 超时），递归增加执行时间
 *   - 迭代版本将所有中间状态显式管理，编译器可更好地优化寄存器分配
 *
 * @param initial_r   从当前命中点出发的入射光线
 * @param world       场景中的球体数组（设备端指针）
 * @param world_size  球体数量
 * @param rand_state  线程的随机数状态指针
 * @param depth       最大弹射深度（路径追踪的迭代上限）
 * @return 该方向到达相机的颜色（RGB，分量范围 [0,1]）
 */
__device__ vec3 ray_color(const ray& initial_r, const sphere* world, int world_size,
                          curandState* rand_state, int depth) {
    ray cur_ray = initial_r;                            // 当前追踪的光线
    vec3 attenuation(1.0f, 1.0f, 1.0f);                // 累积衰减系数（"颜色滤光片"）

    for (int d = 0; d < depth; d++) {
        float t_min = 0.001f;                           // 最小距离：防止自相交（浮点精度问题）
        float t_max = 1e8f;                             // 最大距离：相当于"无穷远"
        float t_hit = t_max;
        int hit_index = -1;                             // -1 表示未命中任何物体

        // ---- 步骤 1：寻找与所有球体的最近交点 ----
        for (int i = 0; i < world_size; i++) {
            float t;
            if (world[i].hit(cur_ray, t_min, t_hit, t)) {
                t_hit = t;                              // 缩小搜索范围（只保留更近的交点）
                hit_index = i;
            }
        }

        // ---- 步骤 2：命中物体 → 计算散射 ----
        if (hit_index != -1) {
            vec3 p = cur_ray.at(t_hit);                 // 命中点的世界坐标
            // 命中点的表面法向量（从球心指向表面）
            vec3 normal = (p - world[hit_index].center).normalize();

            // 确保法向量指向光线来的方向（即与光线方向相反）
            // 当光线从球体内部穿出时，法向量需翻转
            if (cur_ray.direction.dot(normal) > 0.0f) {
                normal = -normal;
            }

            // Lambertian 漫反射散射：
            //   scatter_dir = normal + random_point_on_unit_sphere
            // 这等价于在法向量半球内做 cosine-weighted 采样
            // 注意：random_in_unit_sphere() 返回球内的点，不需 normalize
            vec3 scatter_dir = normal + random_in_unit_sphere(rand_state);

            // 安全保护：极端情况下 random_point ≈ -normal 导致零向量
            // （概率极低但理论上可能 → 退化到沿法向量出射）
            if (scatter_dir.dot(scatter_dir) < 0.0001f) {
                scatter_dir = normal;
            }

            // 更新光线为散射光线（从命中点出发，沿散射方向）
            cur_ray = ray(p, scatter_dir);
            // Beer-Lambert 衰减：累积颜色 = 之前衰减 × 当前物体颜色 × 反射率
            attenuation = attenuation * world[hit_index].color * world[hit_index].albedo;
        }
        // ---- 步骤 3：未命中物体 → 采样背景天空色 ----
        else {
            vec3 unit_dir = cur_ray.direction.normalize();
            // 使用 Y 分量做线性插值：下方偏白（地平线），上方偏蓝（天空）
            // t ∈ [0, 1]，低 Y（地面方向）→ t≈0 → 白色；高 Y（天空方向）→ t≈1 → 蓝色
            float t = 0.5f * (unit_dir.y + 1.0f);
            vec3 background = vec3(1.0f, 1.0f, 1.0f) * (1.0f - t)   // 白色（地面/地平线）
                            + vec3(0.5f, 0.7f, 1.0f) * t;            // 浅蓝色（天空）
            attenuation = attenuation * background;
            break;  // 光线逃逸到天空，结束迭代
        }
    }

    return attenuation;  // 返回累积颜色（未命中物体时包含背景色）
}

/**
 * @brief 主渲染 kernel — GPU 并行光栅化
 *
 * 线程组织：
 *   2D grid × 2D block → 每个线程处理屏幕上一个像素
 *   通过随机抖动实现多重采样抗锯齿（jittered supersampling）
 *
 * 相机模型：
 *   针孔相机，位于原点，朝向 -Z 方向，Y 轴朝上
 *
 * @param framebuffer      输出帧缓冲（设备端 vec3 数组）
 * @param width            图像宽度（像素）
 * @param height           图像高度（像素）
 * @param world            场景球体数组（设备端）
 * @param world_size       球体数量
 * @param rand_states      每个像素的随机数状态（设备端）
 * @param samples_per_pixel  每像素采样数（用于抗锯齿）
 * @param max_depth        光线最大弹射次数
 */
__global__ void render(vec3* framebuffer, int width, int height,
                      const sphere* world, int world_size,
                      curandState* rand_states, int samples_per_pixel, int max_depth) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;     // 像素列坐标（全局）
    int j = threadIdx.y + blockIdx.y * blockDim.y;     // 像素行坐标（全局）

    if (i >= width || j >= height) return;             // 边界线程直接退出

    int idx = j * width + i;                           // 像素一维索引
    curandState local_rand_state = rand_states[idx];   // 拷贝随机状态到局部变量（后续修改不影响全局）
    vec3 color(0, 0, 0);                               // 累积颜色，初始为黑色

    // ================================================
    // 相机参数 & 视口设置
    // ================================================
    vec3 lookfrom(0, 0, 0);      // 相机位置（世界原点）
    vec3 lookat(0, 0, -1);       // 相机朝向目标点
    vec3 vup(0, 1, 0);           // 世界空间的上方向

    // 构建相机的正交基（u, v, w）— 类似 LookAt 矩阵的逆
    vec3 w = (lookfrom - lookat).normalize();          // w：相机前方（从目标指向相机，指向 -Z）
    vec3 u = vup.cross(w).normalize();                 // u：相机右方（水平）
    vec3 v = w.cross(u);                               // v：相机上方（垂直）

    // 视口：位于 z=-1 平面的矩形，高度为 2 个世界单位
    float aspect_ratio = (float)width / height;
    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;

    vec3 horizontal = u * viewport_width;              // 视口水平跨度向量
    vec3 vertical = v * viewport_height;               // 视口垂直跨度向量
    // 视口左下角坐标（从相机出发，向前 -w，再偏移半宽半高）
    vec3 lower_left_corner = lookfrom - horizontal / 2.0f - vertical / 2.0f - w;

    // ================================================
    // 多重采样抗锯齿（每像素 samples_per_pixel 条光线）
    // ================================================
    for (int s = 0; s < samples_per_pixel; s++) {
        // 在像素范围内做随机抖动（jittered sampling）
        // curand_uniform 返回 [0, 1) 的均匀随机数
        float u_coord = (i + curand_uniform(&local_rand_state)) / (width - 1.0f);    // u ∈ [0, 1]
        float v_coord = (j + curand_uniform(&local_rand_state)) / (height - 1.0f);   // v ∈ [0, 1]

        // 从相机发出光线，穿过视口上的采样点
        ray r(lookfrom, lower_left_corner + horizontal * u_coord + vertical * v_coord - lookfrom);
        color += ray_color(r, world, world_size, &local_rand_state, max_depth);
    }

    // ================================================
    // 后处理
    // ================================================
    color /= samples_per_pixel;   // 平均各采样结果

    // Gamma 校正（gamma = 2.0）
    // 显示器对线性颜色值的响应是非线性的，需要做 gamma 编码
    // 公式：output = input^(1/gamma)，此处 gamma=2 → output = sqrt(input)
    color.x = sqrt(color.x);
    color.y = sqrt(color.y);
    color.z = sqrt(color.z);

    framebuffer[idx] = color;
}

// ============================================================================
// 主机端（CPU）函数
// ============================================================================

/**
 * @brief 将帧缓冲区数据写入 PPM（Portable Pixmap）图像文件
 *
 * PPM P3 格式：
 *   第 1 行: P3（"portable pixmap, ASCII"）
 *   第 2 行: width height（图像尺寸）
 *   第 3 行: 255（颜色最大值）
 *   之后:   每行一个像素的 RGB 值（0-255 整数）
 *
 * 注意：PPM 标准要求从图像顶部开始写入，而帧缓冲的第 0 行是屏幕底部，
 *       因此遍历时 j 从 height-1 递减到 0。
 *
 * @param filename    输出文件名
 * @param framebuffer 帧缓冲数据（host 端）
 * @param width       图像宽度
 * @param height      图像高度
 */
void write_ppm(const std::string& filename, const vec3* framebuffer, int width, int height) {
    std::ofstream out(filename);
    out << "P3\n" << width << " " << height << "\n255\n";   // PPM 头部

    // PPM 格式要求从上到下写，而 framebuffer 第 0 行是底部 → 倒序遍历
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            const vec3& c = framebuffer[idx];

            // 将 [0,1] 浮点颜色映射到 [0,255] 整数
            // 乘以 255.99 而非 256：防止浮点舍入导致 256 → 生成非法 PPM
            int r = static_cast<int>(255.99f * c.x);
            int g = static_cast<int>(255.99f * c.y);
            int b = static_cast<int>(255.99f * c.z);

            out << r << " " << g << " " << b << "\n";
        }
    }

    out.close();
}

// ============================================================================
// 主函数
// ============================================================================

int main() 
{
    // ==== 步骤 0：控制台 & GPU 初始化 ====

    // Windows 控制台默认使用 GBK 编码，设置为 UTF-8 以正确显示中文输出
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // 选择第 0 块 GPU（多 GPU 系统需指定）
    CHECK(cudaSetDevice(0));

    // 查询默认的每线程栈大小（通常为 1024 字节）
    size_t currentStackSize;
    CHECK(cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize));
    std::cout << "默认栈大小: " << currentStackSize << " 字节" << std::endl;

    // 增大 GPU 栈大小至 64KB
    // 原因：尽管 ray_color 已改为迭代，但 kernel 内的局部变量、函数调用链
    //       仍然消耗栈空间。64KB 是经验安全值，实际使用远小于此值
    size_t newStackSize = 64 * 1024;  // 64KB
    CHECK(cudaDeviceSetLimit(cudaLimitStackSize, newStackSize));
    std::cout << "已设置栈大小: " << newStackSize << " 字节" << std::endl;

    // ==== 步骤 1：图像参数 ====

    const int width = 4096;                // 图像宽度（像素）
    const int height = 2560;               // 图像高度（像素）
    const int samples_per_pixel = 100;  // 每像素采样数 → 抗锯齿质量
    const int max_depth = 50;         // 光线最大弹射次数 → 间接光照深度

    // ==== 步骤 2：分配设备内存 ====

    // 帧缓冲（host 端，用于 PPM 写入前暂存）
    vec3* framebuffer = new vec3[width * height];
    // 帧缓冲（device 端，kernel 直接写入）
    vec3* d_framebuffer;
    CHECK(cudaMalloc(&d_framebuffer, width * height * sizeof(vec3)));

    // ==== 步骤 3：构建场景 ====

    const int world_size = 4;
    sphere* world = new sphere[world_size];
    sphere* d_world;

    // 场景布局（俯视图）：
    //
    //          Z = -5 平面
    //     ┌─────────────────┐
    //     │   🔵(-2,0,-5)   │   相机在原点(0,0,0)，看向 -Z
    //     │   🔴( 0,0,-5)   │
    //     │   🟢( 2,0,-5)   │
    //     │                 │
    //     │ 🟡 黄色大球地面  │   地面球心(0, -1001, -5), 半径 1000
    //     └─────────────────┘

    world[0] = sphere(vec3(0, 0, -5), 1.0f, vec3(0.8f, 0.3f, 0.3f), 0.8f);       // 中央红色球
    world[1] = sphere(vec3(2, 0, -5), 1.0f, vec3(0.3f, 0.8f, 0.3f), 0.8f);       // 右侧绿色球
    world[2] = sphere(vec3(-2, 0, -5), 1.0f, vec3(0.3f, 0.3f, 0.8f), 0.8f);     // 左侧蓝色球
    world[3] = sphere(vec3(0, -1001, -5), 1000.0f, vec3(0.8f, 0.8f, 0.0f), 0.8f); // 黄色大地
    // 大地球的技巧：半径 1000，球心在 (0, -1001, -5)
    //   球体顶部在 y = -1001 + 1000 = -1，因此从相机角度看是一个在 y≈-1 处的地面

    // 将场景数据复制到设备端
    CHECK(cudaMalloc(&d_world, world_size * sizeof(sphere)));
    CHECK(cudaMemcpy(d_world, world, world_size * sizeof(sphere), cudaMemcpyHostToDevice));

    // ==== 步骤 4：初始化随机数生成器（设备端） ====

    curandState* d_rand_states;
    CHECK(cudaMalloc(&d_rand_states, width * height * sizeof(curandState)));

    // 线程块 & 网格配置
    // 每个 block 32×16 = 512 个线程（CUDA 上限 1024 线程/block，512 是安全值）
    dim3 block_dim(32, 16);
    // grid 覆盖所有像素（向上取整：多余的边界外线程会提前退出）
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                 (height + block_dim.y - 1) / block_dim.y);
    // 实际：grid = (25, 38)，block = (32, 16)
    //   总线程 = 25×38×512 = 486,400，覆盖 800×600 = 480,000 像素 ✓

    init_rand<<<grid_dim, block_dim>>>(d_rand_states, width, height);
    CHECK(cudaDeviceSynchronize());   // 等待 kernel 完成
    CHECK(cudaGetLastError());        // 检查 kernel 是否有运行时错误

    // ==== 步骤 5：执行渲染 ====

    std::cout << "开始渲染..." << std::endl;
    render<<<grid_dim, block_dim>>>(d_framebuffer, width, height, d_world, world_size,
                                   d_rand_states, samples_per_pixel, max_depth);
    // cudaDeviceSynchronize() 阻塞 CPU 直到 GPU 完成所有 kernel 任务
    // cudaGetLastError() 捕获 kernel 执行期间的异步错误（如 TDR 超时、非法内存访问）
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // ==== 步骤 6：结果回传 & 输出 ====

    // 将渲染结果从设备内存复制回主机内存
    CHECK(cudaMemcpy(framebuffer, d_framebuffer, width * height * sizeof(vec3), cudaMemcpyDeviceToHost));

    // 写入 PPM 文件（可在 Photoshop、GIMP、IrfanView 等软件中打开）
    write_ppm("cuda_raytrace.ppm", framebuffer, width, height);
    std::cout << "渲染完成，图像已保存为 cuda_raytrace.ppm" << std::endl;

    // ==== 步骤 7：释放资源 ====

    delete[] framebuffer;     // host 端帧缓冲
    delete[] world;           // host 端场景数据
    cudaFree(d_framebuffer);  // device 端帧缓冲
    cudaFree(d_world);        // device 端场景数据
    cudaFree(d_rand_states);  // device 端随机数状态

    return 0;
}
