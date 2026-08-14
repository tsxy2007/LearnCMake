/**
 * @file    main.cu
 * @brief   CUDA 雅可比（Jacobi）迭代法 — 并行求解线性方程组 Ax = b
 *
 * 本程序使用 CUDA 在 GPU 上并行求解 N 元线性方程组 Ax = b。
 * 雅可比方法是一种经典的迭代法，其天然的数据独立性使其非常适合 GPU 并行化。
 *
 * 数学原理：
 *   对于方程组 Ax = b，将其第 i 个方程展开：
 *     a_i1*x1 + a_i2*x2 + ... + a_ii*xi + ... + a_in*xn = b_i
 *   从中解出 xi（要求 a_ii != 0）：
 *     xi = (b_i - sum_{j!=i} a_ij * xj) / a_ii
 *   雅可比迭代：用第 k 步的所有旧值计算第 k+1 步的新值
 *     xi^(k+1) = (b_i - sum_{j!=i} a_ij * xj^(k)) / a_ii
 *
 *   关键特征：每个 xi 的更新只依赖于上一步的全体 x，各 xi 之间互不依赖
 *            -> 天然并行（embarrassingly parallel），非常适合 GPU
 *
 *   与高斯消去法的对比：
 *     - 直接法（高斯消去）：精确解，但 O(N^3) 且存在数据依赖，难以并行
 *     - 迭代法（Jacobi）：  近似解，每步 O(N^2)，但各方程独立，可大规模并行
 *     - 当 N 很大时，迭代法的并行优势远超直接法
 *
 * CUDA 并行策略：
 *   - 每个线程负责一个方程（矩阵的一行），独立计算一个 xi 的更新
 *   - 双缓冲（x_old, x_new）：Jacobi 要求"用旧值算新值"，不能就地更新，
 *     因此维护两个缓冲区，每步交替使用（类似双缓冲渲染）
 *   - 共享内存归约求最大差（无穷范数）用于收敛判定
 *   - 使用 CUDA Event 精确测量 GPU 计算时间
 *
 * 收敛条件：
 *   当矩阵 A 严格对角占优（|a_ii| > sum_{j!=i} |a_ij|）时，Jacobi 必收敛。
 *   本程序构造严格对角占优矩阵以保证收敛。
 *
 * 编译运行后输出：迭代次数、误差、GPU 计算时间、吞吐量
 *
 * 注意：字符串字面量使用英文，中文仅用于注释。
 *       原因：NVCC 预处理器在中文 Windows（GBK 代码页）上可能将 UTF-8
 *       多字节序列误配对，导致字符串的闭合引号被吞掉（missing closing quote）。
 *       注释不受影响，因为注释在词法分析早期即被移除。
 */

#include <cuda_runtime.h>   // CUDA Runtime API
#include <cmath>            // fabsf, sqrtf, fmaxf
#include <cstdio>           // printf
#include <cstdlib>          // rand, srand, EXIT_FAILURE
#include <iostream>         // std::cout, std::cerr
#include <vector>           // std::vector

#ifdef _WIN32
#include <windows.h>        // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

// ============================================================================
// CUDA 错误检查宏
// ============================================================================

/**
 * @brief 对所有 CUDA API 调用做同步错误检查，失败时打印错误信息并退出
 *
 * 用法：CHECK(cudaMalloc(&ptr, size));
 *        CHECK(cudaDeviceSynchronize());
 */
#define CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "Error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// 设备端（GPU）kernel
// ============================================================================

/**
 * @brief 雅可比迭代 kernel — 核心计算
 *
 * 每个线程负责矩阵的一行（一个方程），独立计算一个未知数的更新：
 *   x_new[i] = (b[i] - sum_{j!=i} A[i][j] * x_old[j]) / A[i][i]
 *
 * 线程组织：一维 grid x 一维 block，每个线程处理一行。
 *
 * 数据布局：矩阵 A 采用行主序（row-major）存储
 *   A[i][j] 在一维数组中的索引为 i * N + j
 *
 * 内存访问说明：
 *   线程 i 顺序读取 A 的第 i 行（地址连续，对缓存友好）。
 *   warp 内相邻线程读取相邻行（地址相距 N*sizeof(float)），
 *   这意味着跨 warp 的矩阵访问不完全合并（uncoalesced）。
 *   生产级实现可使用共享内存分块（tiling）优化，但本例侧重算法清晰性。
 *
 * @param A      系数矩阵（设备端，行主序，N*N）
 * @param b      右端项向量（设备端，N）
 * @param x_old  上一步的解向量（设备端，N）
 * @param x_new  当前步的解向量（设备端，N，输出）
 * @param N      矩阵维度
 */
__global__ void jacobi_iteration(const float* A, const float* b,
                                 const float* x_old, float* x_new, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // 当前线程负责的行号
    if (row >= N) return;                              // 边界外线程退出

    float dot_product = 0.0f;  // 非对角元素的加权和 sum_{j!=i} a_ij * x_old[j]
    float diag = 0.0f;         // 对角元素 a_ii

    // 遍历该行的所有列，累加非对角项
    for (int j = 0; j < N; j++) {
        float a_ij = A[row * N + j];
        if (j == row) {
            diag = a_ij;        // 记录对角元素
        } else {
            dot_product += a_ij * x_old[j];
        }
    }

    // 雅可比更新公式：x_new = (b - 非对角加权和) / 对角元素
    x_new[row] = (b[row] - dot_product) / diag;
}

/**
 * @brief 计算两个向量的最大绝对差（无穷范数），用于收敛判定
 *
 * 使用共享内存树形归约（reduction）求 max|x_new[i] - x_old[i]|。
 *
 * 归约算法：
 *   1. 每个线程加载一个元素到共享内存
 *   2. 树形归约：每轮线程数减半，相邻两个取 max
 *   3. 最终每个 block 的线程 0 得到该 block 的最大值
 *   4. 使用 atomicMax 将各 block 的最大值合并到全局结果
 *
 * 关于 atomicMax 处理 float 的技巧：
 *   CUDA 的 atomicMax 仅支持 int/unsigned int，不支持 float。
 *   但对于非负浮点数，IEEE 754 的位表示满足"值越大，整数表示越大"，
 *   因此可以用 __float_as_int 转为 int 后做 atomicMax，
 *   结果再 __int_as_float 转回 float，等价于 atomicMax for float。
 *   前提：所有值 >= 0（绝对值满足此条件）。
 *
 *   使用前需将 *result 初始化为 0.0f（即 int 0）。
 *
 * @param x_new  新解向量（设备端，N）
 * @param x_old  旧解向量（设备端，N）
 * @param result 全局最大差（设备端，单个 float，调用前需清零）
 * @param N      向量长度
 */
__global__ void compute_max_diff(const float* x_new, const float* x_old,
                                 float* result, int N) {
    extern __shared__ float sdata[];   // 动态共享内存，大小由启动参数指定

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // 加载元素：边界外线程贡献 0（不影响 max）
    sdata[tid] = (i < N) ? fabsf(x_new[i] - x_old[i]) : 0.0f;
    __syncthreads();

    // 树形归约：步长从 blockDim.x/2 逐轮减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();   // 确保该轮归约完成后再进入下一轮
    }

    // 每个 block 的线程 0 将局部最大值原子合并到全局结果
    if (tid == 0) {
        atomicMax((int*)result, __float_as_int(sdata[0]));
    }
}

// ============================================================================
// 主机端（CPU）函数
// ============================================================================

/**
 * @brief 构造严格对角占优的线性方程组 Ax = b（含已知精确解）
 *
 * 严格对角占优：|a_ii| > sum_{j!=i} |a_ij|
 *   -> 保证 Jacobi 迭代收敛（迭代矩阵的谱半径 < 1）
 *
 * 构造方法：
 *   1. 随机填充非对角元素 a_ij（取值范围 [0, 1)）
 *   2. 对角元素 a_ii 设为"非对角绝对值之和 + 余量"
 *      余量越大 -> 对角占优越强 -> 收敛越快
 *   3. 由已知精确解 x_exact 反算 b = A * x_exact
 *      这样我们可以在求解后验证误差 ||x - x_exact||
 *
 * @param A        系数矩阵（host 端，行主序，大小 N*N）
 * @param b        右端项向量（host 端，大小 N）
 * @param x_exact  精确解向量（host 端，大小 N）
 * @param N        矩阵维度
 * @param seed     随机数种子（用于可重复性）
 */
void generate_system(float* A, float* b, const float* x_exact, int N, unsigned int seed) {
    srand(seed);

    for (int i = 0; i < N; i++) {
        float off_diag_sum = 0.0f;   // 第 i 行非对角元素绝对值之和

        for (int j = 0; j < N; j++) {
            if (i != j) {
                // 随机非对角元素：[0, 1) / N，缩小后保证对角强占优
                // 缩放原因：N 很大时若不缩放，off_diag_sum 约为 N/2，
                // 而对角只比它大 1 -> 谱半径接近 1 -> 收敛极慢（上万次迭代）
                float val = (float)(rand() % 100) / 100.0f / N;
                A[i * N + j] = val;
                off_diag_sum += fabsf(val);
            }
        }

        // 对角元素：非对角和 + 1.0，保证严格对角占优且收敛快
        // 缩放后 off_diag_sum 约 0.5，对角约 1.5 -> 谱半径约 0.33
        // -> 收敛仅需约 12 次迭代（tolerance = 1e-6）
        A[i * N + i] = off_diag_sum + 1.0f;
    }

    // 由精确解计算右端项：b = A * x_exact
    for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x_exact[j];
        }
        b[i] = sum;
    }
}

/**
 * @brief CPU 雅可比求解器（用于正确性验证和性能对比）
 *
 * 与 GPU 版本算法完全相同，但单线程串行执行。
 * 仅运行少量迭代用于验证 GPU 结果的正确性。
 *
 * @param A          系数矩阵
 * @param b          右端项
 * @param x          解向量（输入初始猜测，输出结果）
 * @param N          维度
 * @param max_iter   最大迭代次数
 * @param tolerance  收敛阈值
 * @return 实际迭代次数
 */
int cpu_jacobi(const float* A, const float* b, float* x, int N,
               int max_iter, float tolerance) {
    std::vector<float> x_old(N, 0.0f);
    std::vector<float> x_new(N);
    for (int i = 0; i < N; i++) x_old[i] = x[i];

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < N; i++) {
            float dot_product = 0.0f;
            float diag = 0.0f;
            for (int j = 0; j < N; j++) {
                if (j == i) {
                    diag = A[i * N + j];
                } else {
                    dot_product += A[i * N + j] * x_old[j];
                }
            }
            x_new[i] = (b[i] - dot_product) / diag;
        }

        // 计算最大差（无穷范数）
        float max_diff = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = fabsf(x_new[i] - x_old[i]);
            if (diff > max_diff) max_diff = diff;
        }

        // 交换缓冲区
        x_old.swap(x_new);

        if (max_diff < tolerance) {
            for (int i = 0; i < N; i++) x[i] = x_old[i];
            return iter + 1;
        }
    }

    for (int i = 0; i < N; i++) x[i] = x_old[i];
    return max_iter;
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // ==== 问题参数 ====
    const int N = 2048;              // 矩阵维度（2048 x 2048）
    const int max_iter = 10000;      // 最大迭代次数
    const float tolerance = 1e-6f;   // 收敛阈值（无穷范数）

    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Jacobi Iterative Solver" << std::endl;
    std::cout << "  Matrix size: " << N << " x " << N << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 步骤 1：在 CPU 上构造问题（对角占优矩阵 + 已知精确解）====

    std::vector<float> h_A((size_t)N * N);
    std::vector<float> h_b(N);
    std::vector<float> h_x_exact(N);

    // 精确解设为全 1，便于直观验证收敛结果
    for (int i = 0; i < N; i++) {
        h_x_exact[i] = 1.0f;
    }

    generate_system(h_A.data(), h_b.data(), h_x_exact.data(), N, 42);
    std::cout << "[1] Diagonally dominant system constructed" << std::endl;

    // ==== 步骤 2：分配设备内存 ====

    float *d_A, *d_b, *d_x0, *d_x1, *d_max_diff;

    CHECK(cudaMalloc(&d_A,        (size_t)N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_b,        (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_x0,       (size_t)N * sizeof(float)));   // 缓冲区 0
    CHECK(cudaMalloc(&d_x1,       (size_t)N * sizeof(float)));   // 缓冲区 1
    CHECK(cudaMalloc(&d_max_diff, sizeof(float)));               // 收敛判定

    // ==== 步骤 3：拷贝数据到设备 ====

    CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // 初始猜测：x = 0（全零）
    CHECK(cudaMemset(d_x0, 0, N * sizeof(float)));
    std::cout << "[2] Data transferred to GPU" << std::endl;

    // ==== 步骤 4：配置线程组织 ====

    const int block_size = 256;   // 每 block 256 个线程
    const int grid_size = (N + block_size - 1) / block_size;   // 向上取整

    // 归约 kernel 的共享内存大小（每线程一个 float）
    const int smem_size = block_size * sizeof(float);

    std::cout << "[3] Thread config: " << grid_size << " blocks x "
              << block_size << " threads" << std::endl;

    // ==== 步骤 5：迭代求解（GPU）====

    // CUDA Event 用于精确测量 GPU 执行时间
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));   // 计时开始

    int iter;
    float h_max_diff = tolerance + 1.0f;   // 初始值确保至少进入循环
    float* d_x_old = d_x0;                  // 当前指向"旧"缓冲区
    float* d_x_new = d_x1;                  // 当前指向"新"缓冲区

    for (iter = 0; iter < max_iter; iter++) {
        // 5a. 执行一次雅可比迭代：用 x_old 计算 x_new
        jacobi_iteration<<<grid_size, block_size>>>(d_A, d_b, d_x_old, d_x_new, N);

        // 5b. 重置全局最大差为 0（int 0 == float 0.0f）
        CHECK(cudaMemsetAsync(d_max_diff, 0, sizeof(float)));

        // 5c. 计算无穷范数 ||x_new - x_old||_inf
        compute_max_diff<<<grid_size, block_size, smem_size>>>(
            d_x_new, d_x_old, d_max_diff, N);

        // 5d. 将单个标量拷回 CPU 做收敛判断
        CHECK(cudaMemcpy(&h_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost));

        // 5e. 交换双缓冲：下一轮的"旧值"就是本轮的"新值"
        float* tmp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = tmp;

        // 5f. 收敛判定
        if (h_max_diff < tolerance) {
            iter++;   // 迭代计数从 0 开始，实际迭代次数 = iter + 1
            break;
        }
    }

    CHECK(cudaEventRecord(stop));          // 计时结束
    CHECK(cudaEventSynchronize(stop));     // 等待 GPU 完成

    float gpu_time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    std::cout << "[4] GPU solve complete" << std::endl;

    // 注意：交换后 d_x_old 指向最新的解
    std::vector<float> h_x_gpu(N);
    CHECK(cudaMemcpy(h_x_gpu.data(), d_x_old, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ==== 步骤 6：结果验证 ====

    // 6a. 计算与精确解的误差（L2 范数和无穷范数）
    float l2_error = 0.0f;
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_x_gpu[i] - h_x_exact[i]);
        l2_error += err * err;
        if (err > max_error) max_error = err;
    }
    l2_error = sqrtf(l2_error) / N;   // 归一化 L2 误差

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Results" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Iterations:      " << iter << std::endl;
    std::cout << "  Final max diff:  " << h_max_diff << std::endl;
    std::cout << "  L2 error:        " << l2_error << std::endl;
    std::cout << "  Max error:       " << max_error << std::endl;
    std::cout << "  GPU time:        " << gpu_time_ms << " ms" << std::endl;

    // 6b. 吞吐量估算：每次迭代约 2*N^2 次浮点运算（N 个行向量点积）
    //     每行 N 次乘加 = 2N FLOP，共 N 行 -> 2N^2 FLOP/迭代
    double flops = 2.0 * (double)N * N * iter;
    double gflops = flops / (gpu_time_ms * 1e-3) / 1e9;
    std::cout << "  Throughput:      " << gflops << " GFLOP/s" << std::endl;

    // 6c. CPU 验证（相同迭代数，验证算法一致性）
    std::vector<float> h_x_cpu(N, 0.0f);
    int cpu_iters = cpu_jacobi(h_A.data(), h_b.data(), h_x_cpu.data(),
                               N, iter, tolerance);
    float cpu_gpu_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_x_gpu[i] - h_x_cpu[i]);
        if (diff > cpu_gpu_diff) cpu_gpu_diff = diff;
    }
    std::cout << "  CPU vs GPU diff: " << cpu_gpu_diff
              << " (CPU iters: " << cpu_iters << ")" << std::endl;

    // 6d. 打印前 5 个分量的对比（精确解 = 1.0）
    std::cout << std::endl;
    std::cout << "  First 5 components (exact = 1.0):" << std::endl;
    for (int i = 0; i < 5; i++) {
        printf("    x[%d] = %.10f\n", i, h_x_gpu[i]);
    }
    std::cout << "----------------------------------------" << std::endl;

    // ==== 步骤 7：释放资源 ====

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_x0));
    CHECK(cudaFree(d_x1));
    CHECK(cudaFree(d_max_diff));

    return 0;
}
