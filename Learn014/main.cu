/**
 * @file    main.cu
 * @brief   CUDA 共轭梯度法（Conjugate Gradient, CG）— 求解对称正定方程组 Ax = b
 *
 * 本程序是 Learn012（雅可比）、Learn013（高斯-赛德尔）的续篇。
 * 前两者是"固定格式"的分裂迭代法（A = D + L + U 的组合），
 * 共轭梯度法则完全不同：它是 Krylov 子空间方法，只要求矩阵以"乘法"出现
 * （matvec），特别适合大规模稀疏问题，是科学计算中求解 SPD 系统的主力。
 *
 * 适用条件（CG 的硬性要求）：
 *   A 必须对称正定（SPD, Symmetric Positive Definite）：
 *   对称：A = A^T；正定：任意 x != 0 有 x^T A x > 0。
 *
 * 数学原理：
 *   1) 等价的优化问题：当 A 对称正定时
 *        Ax = b  <=>  min f(x) = 1/2 * x^T A x - b^T x
 *      因为 grad f(x) = Ax - b = -r（r = b - Ax 是残差），
 *      解方程等价于找一个二次凸函数的极小点。
 *
 *   2) 最速下降（Steepest Descent, SD）—— CG 的铺垫：
 *      从 x_k 沿负梯度方向（即残差 r_k）走最优步长（精确线搜索）：
 *        alpha_k = (r_k . r_k) / (r_k . A r_k)
 *        x_{k+1} = x_k + alpha_k r_k
 *      缺点：相邻两步的梯度互相垂直，路线呈"锯齿形"，收敛因子
 *      (kappa-1)/(kappa+1)（kappa 为条件数），病态问题时极慢。
 *
 *   3) 共轭（A-正交）方向：
 *      若方向组满足 p_i^T A p_j = 0 (i != j)，则依次沿每个方向做精确
 *      线搜索，最多 N 步到达精确解（在 A-内积 <x,y>_A = x^T A y 意义下
 *      相当于"坐标下降"）。关键问题：如何廉价地得到一组共轭方向？
 *
 *   4) CG 的巧妙之处：
 *      - 残差天然两两正交：r_i . r_j = 0 (i != j)
 *      - 因此只需三项递推即可生成新的共轭方向（等价于对残差做
 *        Gram-Schmidt 正交化，但其余项全部自动为零）：
 *            p_0 = r_0
 *            p_{k+1} = r_{k+1} + beta_k p_k,   beta_k = (r_{k+1}.r_{k+1})/(r_k.r_k)
 *
 *   CG 算法（每迭代：1 次矩阵-向量乘 + 3 个点积 + 3 个 axpy）：
 *     x_0 = 0;  r_0 = b - A x_0;  p_0 = r_0
 *     for k = 0, 1, 2, ...
 *         Ap       = A p_k
 *         alpha_k  = (r_k . r_k) / (p_k . A p_k)      // 精确线搜索步长
 *         x_{k+1}  = x_k + alpha_k p_k
 *         r_{k+1}  = r_k - alpha_k A p_k               // 递推残差（免重算 matvec）
 *         （若 ||r_{k+1}|| < tol 则停止）
 *         beta_k   = (r_{k+1} . r_{k+1}) / (r_k . r_k)
 *         p_{k+1}  = r_{k+1} + beta_k p_k
 *
 *   收敛速度（A-范数误差）：
 *        ||x_k - x*||_A <= 2 * ((sqrt(kappa)-1)/(sqrt(kappa)+1))^k * ||x_0-x*||_A
 *     依赖 sqrt(kappa) 而非 kappa —— 病态问题上远快于最速下降；
 *     且精确算术下最多 N 步收敛到精确解（本例 N=3，实测 ~3 步）。
 *
 *   与 Learn012/013 的对比：
 *     - 雅可比 / GS：任何可分裂的方阵都能套用，收敛慢（线性、依赖谱半径）
 *     - CG：只适用 SPD，但收敛快（有限步 + 谱条件数平方根速度），
 *       且只需 matvec —— 稀疏矩阵无需存 L/U 分裂
 *     - GS 需要串行依赖链；CG 每步全是 BLAS1/2 + SpMV，天然可并行
 *
 * CUDA 实现策略（本程序）：
 *   - matvec：一行一线程的稠密矩阵-向量乘（Learn004 模式）
 *   - dot_product：块内共享内存树形归约 + atomicAdd 合并各 block 结果。
 *     注意 atomicAdd 原生支持 float（对照 Learn012 中 atomicMax 不支持
 *     float、必须用 __float_as_int 位技巧绕行的情形）
 *   - axpy / update_direction：一元素一线程的 BLAS1 操作
 *   - 主机驱动迭代循环：alpha、beta 是标量，本程序每迭代把点积拷回 CPU
 *     计算（每轮 3 次小拷贝 + 隐式同步）。生产实现（cuSPARSE/cuBLAS、
 *     PETSc）把标量留在设备端以避免同步；再配合预条件子即 PCG
 *     （预条件共轭梯度，可作为后续课题）
 *   - 计时窗口外预热一次 kernel，避免首次启动的模块加载 / PTX JIT
 *     一次性开销污染计时（Learn013 的经验）
 *
 * 收敛判据说明：
 *   CG 中自然的判据是残差 2-范数 ||r||_2 < tol（r 是 CG 递推量，
 *   免费可得）。注意递推残差与真实残差 b - Ax 在浮点下会漂移，
 *   程序末尾同时打印两者。
 *
 * 注意：字符串字面量使用英文，中文仅用于注释（GBK 代码页下 NVCC 的
 *       UTF-8 误配对问题，见 Learn012）。
 */

#include <cuda_runtime.h>   // CUDA Runtime API
#include <cmath>            // fabsf, sqrtf
#include <cstdio>           // printf
#include <cstdlib>          // EXIT_FAILURE
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
// 设备端（GPU）kernel —— CG 每迭代所需的全部原语
// ============================================================================

/**
 * @brief 稠密矩阵-向量乘 y = A * x（CG 每迭代唯一"重"的操作）
 *
 * 一行一线程：线程 i 计算 y[i] = sum_j A[i][j] * x[j]。
 * 行主序存储，线程 i 顺序读第 i 行（地址连续，缓存友好）。
 * 稀疏矩阵时此 kernel 替换为 SpMV（如 cuSPARSE 的 csr mv），
 * CG 其余部分完全不变 —— "矩阵只以乘法出现"的好处。
 *
 * @param A  系数矩阵（设备端，行主序，N*N，须对称正定）
 * @param x  输入向量（设备端，N）
 * @param y  输出向量（设备端，N）
 * @param N  维度
 */
__global__ void matvec(const float* A, const float* x, float* y, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;   // 当前线程负责的行号
    if (row >= N) return;                              // 边界外线程退出

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        sum += A[row * N + j] * x[j];
    }
    y[row] = sum;
}

/**
 * @brief 点积 result = x . y（块内树形归约 + atomicAdd 合并）
 *
 * 归约算法：
 *   1. 每个线程加载一个元素对之积到共享内存
 *   2. 树形归约：每轮线程数减半，相邻两个相加
 *   3. 每个 block 的线程 0 用 atomicAdd 把局部和累加到全局结果
 *
 * 与 Learn012 的 compute_max_diff 同构（max 换成加法）。
 * atomicAdd 原生支持 float（不像 atomicMax 需要位技巧），
 * 多个 block 并发累加是安全的；代价是结果按 block 完成顺序相加，
 * 浮点求和顺序不定 —— 对本例无影响。
 *
 * 调用前需将 *result 清零（cudaMemsetAsync）。
 *
 * @param x       输入向量 1（设备端，N）
 * @param y       输入向量 2（设备端，N）
 * @param result  点积结果（设备端，单个 float，调用前需清零）
 * @param N       向量长度
 */
__global__ void dot_product(const float* x, const float* y,
                            float* result, int N) {
    extern __shared__ float sdata[];   // 动态共享内存，大小由启动参数指定

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    // 加载元素：边界外线程贡献 0（不影响和）
    sdata[tid] = (i < N) ? x[i] * y[i] : 0.0f;
    __syncthreads();

    // 树形归约：步长从 blockDim.x/2 逐轮减半（要求 blockDim.x 为 2 的幂）
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();   // 确保该轮归约完成后再进入下一轮
    }

    // 每个 block 的线程 0 将局部和原子累加到全局结果
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

/**
 * @brief BLAS axpy：y = y + alpha * x（一元素一线程）
 *
 * 用于 CG 的两处更新：
 *   x = x + alpha * p   （解向量的修正）
 *   r = r - alpha * Ap  （残差的递推，免去重新计算 b - Ax）
 *
 * @param alpha  标量系数（host 端由点积计算得到，按值传入）
 * @param x      输入向量（设备端，N）
 * @param y      读改写向量（设备端，N）
 * @param N      维度
 */
__global__ void axpy(float alpha, const float* x, float* y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    y[i] += alpha * x[i];
}

/**
 * @brief 共轭方向更新：p = r + beta * p（一元素一线程）
 *
 * 这是 CG 生成共轭方向的三项递推（"Gram-Schmidt 只剩一项"的原因
 * 是残差天然正交）。beta = 0 时退化为最速下降的重启。
 *
 * @param beta  标量系数（host 端由相邻两次残差点积计算）
 * @param r     当前残差（设备端，N，只读）
 * @param p     搜索方向（设备端，N，就地读改写）
 * @param N     维度
 */
__global__ void update_direction(float beta, const float* r, float* p, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    p[i] = r[i] + beta * p[i];
}

// ============================================================================
// 主机端（CPU）函数
// ============================================================================

/// 迭代轨迹最多打印的前几步
const int TRACE_ITERS = 6;

/**
 * @brief CPU 共轭梯度求解器（参考实现，打印迭代轨迹）
 *
 * 与 GPU 版本算法完全相同。轨迹打印 alpha / beta / ||r||_2，
 * 观察 ||r|| 每步单调下降、精确算术下最多 N 步收敛到 0。
 *
 * @param A          系数矩阵（行主序，N*N，须 SPD）
 * @param b          右端项
 * @param x          解向量（输入初始猜测，输出结果）
 * @param N          维度
 * @param max_iter   最大迭代次数
 * @param tolerance  残差 2-范数收敛阈值
 * @param verbose    是否打印前 TRACE_ITERS 步轨迹
 * @return 实际迭代次数
 */
int cpu_conjugate_gradient(const float* A, const float* b, float* x, int N,
                           int max_iter, float tolerance, bool verbose) {
    std::vector<float> r(N), p(N), Ap(N);

    // 初始化：r_0 = b - A x_0，p_0 = r_0（x 输入为初始猜测）
    for (int i = 0; i < N; i++) {
        float ax = 0.0f;
        for (int j = 0; j < N; j++) ax += A[i * N + j] * x[j];
        r[i] = b[i] - ax;
        p[i] = r[i];
    }

    float rr = 0.0f;                      // r . r
    for (int i = 0; i < N; i++) rr += r[i] * r[i];

    for (int iter = 0; iter < max_iter; iter++) {
        // Ap = A p
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) sum += A[i * N + j] * p[j];
            Ap[i] = sum;
        }

        // pAp = p . Ap
        float pAp = 0.0f;
        for (int i = 0; i < N; i++) pAp += p[i] * Ap[i];

        // 数值保护：pAp 接近 0 说明已收敛到机器精度（p = 0 或方向退化）
        if (pAp == 0.0f) return iter;

        // 精确线搜索步长
        float alpha = rr / pAp;

        // x = x + alpha p；r = r - alpha Ap
        for (int i = 0; i < N; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // rr_new = r . r（新残差的范数平方）
        float rr_new = 0.0f;
        for (int i = 0; i < N; i++) rr_new += r[i] * r[i];

        if (verbose && iter < TRACE_ITERS) {
            printf("    iter %2d: alpha = %.6f, ||r||_2 = %.6e\n",
                   iter + 1, alpha, sqrtf(rr_new));
        }

        if (rr_new < tolerance * tolerance) {
            return iter + 1;
        }

        // beta = rr_new / rr；p = r + beta p
        float beta = rr_new / rr;
        for (int i = 0; i < N; i++) p[i] = r[i] + beta * p[i];

        rr = rr_new;
    }
    return max_iter;
}

/**
 * @brief CPU 最速下降求解器（CG 的铺垫，用于对比收敛速度）
 *
 * 沿负梯度（残差）方向做精确线搜索。可见其收敛明显慢于 CG：
 * 收敛因子 (kappa-1)/(kappa+1)，且相邻两步方向互相垂直，
 * 迭代路线呈"锯齿形"。
 */
int cpu_steepest_descent(const float* A, const float* b, float* x, int N,
                         int max_iter, float tolerance, bool verbose) {
    std::vector<float> r(N), Ar(N);

    for (int i = 0; i < N; i++) {
        float ax = 0.0f;
        for (int j = 0; j < N; j++) ax += A[i * N + j] * x[j];
        r[i] = b[i] - ax;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) sum += A[i * N + j] * r[j];
            Ar[i] = sum;
        }

        float rr = 0.0f, rAr = 0.0f;
        for (int i = 0; i < N; i++) {
            rr += r[i] * r[i];
            rAr += r[i] * Ar[i];
        }
        float alpha = rr / rAr;

        for (int i = 0; i < N; i++) {
            x[i] += alpha * r[i];
            r[i] -= alpha * Ar[i];
        }

        float rr_new = 0.0f;
        for (int i = 0; i < N; i++) rr_new += r[i] * r[i];

        if (verbose && iter < TRACE_ITERS) {
            printf("    iter %2d: alpha = %.6f, ||r||_2 = %.6e\n",
                   iter + 1, alpha, sqrtf(rr_new));
        }

        if (rr_new < tolerance * tolerance) {
            return iter + 1;
        }
    }
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
    const int N = 3;                 // 矩阵维度
    const int max_iter = 100;        // 最大迭代次数（CG 理论上 <= N 步收敛）
    const float tolerance = 1e-6f;   // 收敛阈值（残差 2-范数 ||r||_2）

    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Conjugate Gradient (CG) Solver" << std::endl;
    std::cout << "  Matrix size: " << N << " x " << N << " (SPD required)" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 步骤 1：构造对称正定（SPD）问题 ====
    // CG 硬性要求 A 对称正定。本例矩阵：
    //   对称：A = A^T（a_01 = a_10 = 1, a_02 = a_20 = 1, a_12 = a_21 = 1）
    //   严格对角占优 + 正对角元 => 正定（对称矩阵的 Gershgorin 圆盘
    //         全部落在正半轴：lambda >= min_i (a_ii - sum_{j!=i} |a_ij|) > 0）
    //     行 0: 5 > 1+1 = 2；行 1: 6 > 1+1 = 2；行 2: 7 > 1+1 = 2
    // 右端项取 b = A * (1,1,1) = (7, 8, 9)，精确解即 (1, 1, 1)，便于验证：
    //   5x + 1y + 1z =  7
    //   1x + 6y + 1z =  8
    //   1x + 1y + 7z =  9
    std::vector<float> h_A = { 5.0f, 1.0f, 1.0f,   // 第 0 行
                               1.0f, 6.0f, 1.0f,   // 第 1 行
                               1.0f, 1.0f, 7.0f }; // 第 2 行
    std::vector<float> h_b = { 7.0f, 8.0f, 9.0f };
    const float exact[3] = { 1.0f, 1.0f, 1.0f };

    std::cout << "[1] SPD system constructed:" << std::endl;
    std::cout << "    5x + 1y + 1z =  7" << std::endl;
    std::cout << "    1x + 6y + 1z =  8" << std::endl;
    std::cout << "    1x + 1y + 7z =  9" << std::endl;
    std::cout << "    symmetric, strictly diagonally dominant" << std::endl;
    std::cout << "    => positive definite, CG applies" << std::endl;

    // ==== 步骤 2：CPU 求解（带轨迹打印）====

    // 2a. 最速下降（CG 的铺垫：同样的线搜索，但方向选残差本身）
    std::cout << std::endl;
    std::cout << "[2] CPU steepest descent trace (zigzag, slow):" << std::endl;
    std::vector<float> h_x_sd(N, 0.0f);
    int sd_iters = cpu_steepest_descent(h_A.data(), h_b.data(), h_x_sd.data(),
                                        N, max_iter, tolerance, /*verbose=*/true);
    std::cout << "    -> converged in " << sd_iters << " iterations" << std::endl;

    // 2b. 共轭梯度（观察：最多 N 步收敛到机器精度）
    std::cout << std::endl;
    std::cout << "[3] CPU conjugate gradient trace (conjugate directions):" << std::endl;
    std::vector<float> h_x_cg(N, 0.0f);
    int cg_iters = cpu_conjugate_gradient(h_A.data(), h_b.data(), h_x_cg.data(),
                                          N, max_iter, tolerance, /*verbose=*/true);
    std::cout << "    -> converged in " << cg_iters
              << " iterations (<= N = " << N << " in exact arithmetic)" << std::endl;

    // ==== 步骤 3：分配设备内存并拷贝数据 ====

    float *d_A, *d_x, *d_r, *d_p, *d_Ap, *d_dot;

    CHECK(cudaMalloc(&d_A,   (size_t)N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_x,   (size_t)N * sizeof(float)));   // 解
    CHECK(cudaMalloc(&d_r,   (size_t)N * sizeof(float)));   // 残差
    CHECK(cudaMalloc(&d_p,   (size_t)N * sizeof(float)));   // 搜索方向
    CHECK(cudaMalloc(&d_Ap,  (size_t)N * sizeof(float)));   // A*p 临时向量
    CHECK(cudaMalloc(&d_dot, sizeof(float)));               // 点积结果

    CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));

    // ==== 步骤 4：配置线程组织 ====

    const int block_size = 256;   // 每 block 256 个线程（2 的幂，归约要求）
    const int grid_size = (N + block_size - 1) / block_size;   // 向上取整
    const int smem_size = block_size * sizeof(float);          // 归约共享内存

    std::cout << std::endl;
    std::cout << "[4] Thread config: " << grid_size << " blocks x "
              << block_size << " threads" << std::endl;
    std::cout << "    per iteration: 1 matvec + 3 dots + 3 axpy kernels" << std::endl;

    // CUDA Event 用于精确测量 GPU 执行时间
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // ==== 步骤 5：GPU 共轭梯度求解 ====

    // 预热（warm-up）：首次 kernel 启动触发模块加载 / PTX JIT，
    // 一次性开销可达上百毫秒，必须在计时窗口外排除（Learn013 的经验）
    matvec<<<grid_size, block_size>>>(d_A, d_p, d_Ap, N);
    CHECK(cudaDeviceSynchronize());

    // 初始化：x_0 = 0，r_0 = b - A x_0 = b，p_0 = r_0
    CHECK(cudaMemset(d_x, 0, N * sizeof(float)));
    CHECK(cudaMemcpy(d_r, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_p, d_r, N * sizeof(float), cudaMemcpyDeviceToDevice));

    // rr = r . r（进入循环前的初始残差范数平方）
    CHECK(cudaMemsetAsync(d_dot, 0, sizeof(float)));
    dot_product<<<grid_size, block_size, smem_size>>>(d_r, d_r, d_dot, N);
    float rr = 0.0f;
    CHECK(cudaMemcpy(&rr, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(start));   // 计时开始

    int iter;
    float final_rnorm = sqrtf(rr);

    for (iter = 0; iter < max_iter; iter++) {
        // 5a. Ap = A * p（每迭代唯一的矩阵乘）
        matvec<<<grid_size, block_size>>>(d_A, d_p, d_Ap, N);

        // 5b. pAp = p . Ap
        CHECK(cudaMemsetAsync(d_dot, 0, sizeof(float)));
        dot_product<<<grid_size, block_size, smem_size>>>(d_p, d_Ap, d_dot, N);
        float pAp = 0.0f;
        CHECK(cudaMemcpy(&pAp, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

        // 数值保护：pAp == 0 说明已到机器精度（p 退化），无法继续
        if (pAp == 0.0f) {
            std::cout << "    breakdown at iter " << iter
                      << " (pAp == 0), stopping" << std::endl;
            break;
        }

        // 5c. alpha = (r.r) / (p.Ap) —— host 端标量运算
        float alpha = rr / pAp;

        // 5d. x = x + alpha * p；r = r - alpha * Ap
        axpy<<<grid_size, block_size>>>(alpha, d_p, d_x, N);
        axpy<<<grid_size, block_size>>>(-alpha, d_Ap, d_r, N);

        // 5e. rr_new = r . r
        CHECK(cudaMemsetAsync(d_dot, 0, sizeof(float)));
        dot_product<<<grid_size, block_size, smem_size>>>(d_r, d_r, d_dot, N);
        float rr_new = 0.0f;
        CHECK(cudaMemcpy(&rr_new, d_dot, sizeof(float), cudaMemcpyDeviceToHost));

        final_rnorm = sqrtf(rr_new);

        // 5f. 收敛判定：||r||_2 < tol
        if (final_rnorm < tolerance) {
            iter++;
            break;
        }

        // 5g. beta = rr_new / rr；p = r + beta * p；rr = rr_new
        float beta = rr_new / rr;
        update_direction<<<grid_size, block_size>>>(beta, d_r, d_p, N);
        rr = rr_new;
    }

    CHECK(cudaEventRecord(stop));        // 计时结束
    CHECK(cudaEventSynchronize(stop));   // 等待 GPU 完成

    float gpu_time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));

    std::vector<float> h_x_gpu(N);
    CHECK(cudaMemcpy(h_x_gpu.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << std::endl;
    std::cout << "[5] GPU CG solve complete" << std::endl;

    // ==== 步骤 6：结果验证 ====

    // 6a. 真实残差 ||b - A x||_2（对照递推残差：浮点下两者有微小漂移）
    float true_res = 0.0f;
    for (int i = 0; i < N; i++) {
        float ax = 0.0f;
        for (int j = 0; j < N; j++) ax += h_A[i * N + j] * h_x_gpu[j];
        float res = h_b[i] - ax;
        true_res += res * res;
    }
    true_res = sqrtf(true_res);

    // 6b. 与理论精确解、CPU 参考解的偏差
    float exact_err = 0.0f;
    float cpu_gpu_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float e1 = fabsf(h_x_gpu[i] - exact[i]);
        float e2 = fabsf(h_x_gpu[i] - h_x_cg[i]);
        if (e1 > exact_err) exact_err = e1;
        if (e2 > cpu_gpu_diff) cpu_gpu_diff = e2;
    }

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Results (Conjugate Gradient)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Iterations:            " << iter
              << " (theory: <= N = " << N << ")" << std::endl;
    std::cout << "  Recursive ||r||_2:     " << final_rnorm << std::endl;
    std::cout << "  True ||b - Ax||_2:     " << true_res << std::endl;
    std::cout << "  Max |x - x_exact|:     " << exact_err << std::endl;
    std::cout << "  CPU vs GPU diff:       " << cpu_gpu_diff
              << " (CPU iters: " << cg_iters << ")" << std::endl;
    std::cout << "  GPU time:              " << gpu_time_ms << " ms" << std::endl;

    // 6c. 最速下降 vs 共轭梯度 —— 本程序的核心结论
    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Steepest Descent vs Conjugate Gradient" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  SD iterations: " << sd_iters << " (zigzag on gradient)" << std::endl;
    std::cout << "  CG iterations: " << cg_iters << " (conjugate directions," << std::endl;
    std::cout << "                 exact in at most N steps)" << std::endl;
    std::cout << "  -> CG picks each new direction A-orthogonal to ALL" << std::endl;
    std::cout << "     previous ones, so no step is ever undone." << std::endl;
    std::cout << "  -> For sparse SPD systems (e.g. PDE discretizations)," << std::endl;
    std::cout << "     CG + preconditioning (PCG) is the workhorse solver." << std::endl;

    // 6d. 打印解向量
    std::cout << std::endl;
    std::cout << "  Solution:" << std::endl;
    int print_count = N < 5 ? N : 5;   // 不超过 N，避免越界读取
    for (int i = 0; i < print_count; i++) {
        printf("    %c = %.10f   (exact: %.10f)\n", 'x' + i, h_x_gpu[i], exact[i]);
    }
    std::cout << "----------------------------------------" << std::endl;

    // ==== 步骤 7：释放资源 ====

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_r));
    CHECK(cudaFree(d_p));
    CHECK(cudaFree(d_Ap));
    CHECK(cudaFree(d_dot));

    return 0;
}
