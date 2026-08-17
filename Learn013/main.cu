/**
 * @file    main.cu
 * @brief   CUDA 高斯-赛德尔（Gauss-Seidel）迭代法 — 求解线性方程组 Ax = b
 *
 * 本程序是 Learn012（雅可比迭代）的续篇，使用 CUDA 实现高斯-赛德尔迭代，
 * 并在同一方程组上对比两种方法的收敛速度。
 *
 * 数学原理：
 *   雅可比迭代用"上一步的所有旧值"计算本步新值：
 *     xi^(k+1) = (b_i - sum_{j!=i} a_ij * xj^(k)) / a_ii
 *   高斯-赛德尔的改进只有一句话：一旦算出新值，立即投入使用。
 *   按未知数编号从小到大依次计算，则计算 xi 时，编号更小的未知数的
 *   新值已经算出 —— 直接用新值代替旧值：
 *     xi^(k+1) = (b_i - sum_{j<i} a_ij * xj^(k+1)     <- 本轮新值（已算出）
 *                     - sum_{j>i} a_ij * xj^(k) ) / a_ii  <- 上轮旧值（还没算到）
 *
 *   矩阵分裂视角（A = L + D + U：严格下三角 + 对角 + 严格上三角）：
 *     雅可比：        x^(k+1) = D^-1 * (b - (L+U) * x^(k))
 *     高斯-赛德尔：   x^(k+1) = (D+L)^-1 * (b - U * x^(k))
 *   迭代法收敛 <=> 迭代矩阵谱半径 rho < 1。
 *
 *   实现上 GS 只需一个数组就地（in-place）更新：算出 xi 立刻写回 x[i]，
 *   后续读取 x[j] (j<i) 时自然读到新值 —— 不需要雅可比那样的双缓冲。
 *
 * 收敛条件与速度：
 *   - 严格对角占优（|a_ii| > sum_{j!=i} |a_ij|）时，GS 与雅可比都必收敛；
 *   - 对称正定（SPD）矩阵：GS 必收敛（雅可比反而不保证）；
 *   - 对一大类矩阵（相容排序，如三对角阵）有定量关系 rho_GS = rho_J^2，
 *     因此 GS 的迭代次数通常约为雅可比的一半 —— 本程序输出可验证。
 *
 * GPU 并行化的核心困难（本例的课题）：
 *   雅可比"各未知数互不依赖"，一个线程管一行，天然并行；
 *   GS 却存在循环携带依赖（loop-carried dependency）：
 *   xi^(k+1) 依赖 xj^(k+1) (j<i)，必须等前面的未知数算完才能算后面的。
 *   实际工程中的对策有：
 *     1. 红黑/多色排序（PDE 五点 stencil 的经典解法：按依赖图着色，
 *        同色未知数互不依赖可并行，逐色更新）
 *     2. 分块 GS：块间串行、块内退化为雅可比并行
 *     3. 异步/混沌迭代：就地乱序更新，对角占优时仍收敛（但不确定）
 *
 * 本程序采用的方案（教学上最清晰的确定性实现）：
 *   "按行串行、行内并行" 的单 block kernel：
 *     - kernel 内 for 循环按行号 i 从 0 到 N-1 串行推进（保证 GS 依赖顺序），
 *       每算完一行由 __syncthreads() 保证写入对全 block 可见后再进入下一行；
 *     - 每行内部，block 中全部线程用共享内存树形归约（复用 Learn012 的
 *       归约模式）并行计算点积 sum_{j!=i} a_ij * x_j；
 *     - 必须只启动 1 个 block：__syncthreads() 只在 block 内有效，
 *       多 block 之间无法维持行间依赖顺序。
 *   这正是稀疏三角方程组求解（SpTRSV）中 GS 求解器的经典结构：
 *   "外层按行串行、内层按列并行"。代价是无法利用多 block，
 *   大规模问题上通常让位于红黑 GS 或分块方法 —— 详见程序末尾输出。
 *
 * 程序结构：
 *   1. CPU 上先跑 GS 与雅可比（带迭代轨迹打印，观察 GS 用新值后收敛更快）
 *   2. GPU 上跑单 block 的 GS kernel（时间用 CUDA Event 测量）
 *   3. GPU 上跑雅可比（Learn012 的方法）作迭代次数 / 耗时对比
 *   4. 残差 ||Ax-b|| 验证 + 与理论精确解对比
 *
 * 注意：字符串字面量使用英文，中文仅用于注释。
 *       原因：NVCC 预处理器在中文 Windows（GBK 代码页）上可能将 UTF-8
 *       多字节序列误配对，导致字符串的闭合引号被吞掉（missing closing quote）。
 *       注释不受影响，因为注释在词法分析早期即被移除。
 */

#include <cuda_runtime.h>   // CUDA Runtime API
#include <cmath>            // fabsf, sqrtf, fmaxf
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
// 设备端（GPU）kernel
// ============================================================================

/**
 * @brief 高斯-赛德尔迭代 kernel — 核心计算（必须只启动 1 个 block）
 *
 * 完成一次完整的 GS 扫描：按行号 i 从 0 到 N-1 依次更新
 *   x[i] = (b[i] - sum_{j!=i} A[i][j] * x[j]) / A[i][i]
 * 其中 x 就地更新 —— 计算第 i 行时读到的 x[j] (j<i) 是本轮刚写入的新值，
 * x[j] (j>i) 是上一轮的旧值，这正是 GS 语义。
 *
 * 线程组织：单个 block，blockDim.x >= N（本例 256 >= 3）。
 *
 * 行内并行：blockDim.x 个线程合作计算第 i 行的非对角点积：
 *   1. 线程 tid 负责第 j = tid 列：贡献 a_ij * x[j]（j 越界或 j == i 时贡献 0）
 *   2. 共享内存树形归约求和（与 Learn012 的 max 归约同构，只是换成加法）
 *   3. 线程 0 根据归约结果写回 x[i]
 *
 * 同步说明（本 kernel 正确性的关键）：
 *   - 行间串行由 __syncthreads() 保证：写 x[i] 之后、进入下一行之前必须
 *     同步一次，否则其他线程可能先读到尚未更新的 x[i]；
 *   - __syncthreads() 会保证 barrier 之前的全局内存写入对 block 内所有
 *     线程可见，因此行间数据依赖（GS 的灵魂）在单 block 内是安全的；
 *   - 所有线程必须执行到每一个 __syncthreads()，因此本 kernel 不允许
 *     "越界线程提前 return"（Learn012 的 jacobi_iteration 可以，这里不行）。
 *
 * @param A  系数矩阵（设备端，行主序，N*N）
 * @param b  右端项向量（设备端，N）
 * @param x  解向量（设备端，N，就地读改写）
 * @param N  矩阵维度
 */
__global__ void gauss_seidel_iteration(const float* A, const float* b,
                                       float* x, int N) {
    extern __shared__ float sdata[];   // 动态共享内存，大小由启动参数指定

    unsigned int tid = threadIdx.x;

    // 外层：按行串行推进 —— GS 的依赖顺序（后面的行要用前面的新值）
    for (int i = 0; i < N; i++) {
        // 内层：行内并行。线程 tid 负责第 j = tid 列的点积贡献
        int j = (int)tid;
        float contribution = 0.0f;
        if (j < N && j != i) {
            contribution = A[i * N + j] * x[j];   // j<i 时是本轮新值，j>i 时是旧值
        }
        sdata[tid] = contribution;
        __syncthreads();

        // 树形归约：步长从 blockDim.x/2 逐轮减半（要求 blockDim.x 为 2 的幂）
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();   // 确保该轮归约完成后再进入下一轮
        }

        // 线程 0 独占写回：x[i] = (b_i - 非对角加权和) / 对角元素
        if (tid == 0) {
            x[i] = (b[i] - sdata[0]) / A[i * N + i];
        }
        __syncthreads();   // 写入必须对下一行的所有读取线程可见
    }
}

/**
 * @brief 计算两个向量的最大绝对差（无穷范数），用于收敛判定
 *
 * 使用共享内存树形归约（reduction）求 max|x_new[i] - x_old[i]|。
 * 与 Learn012 完全相同，此处复用以对比就地更新前后解向量的差异。
 *
 * 关于 atomicMax 处理 float 的技巧：
 *   CUDA 的 atomicMax 仅支持 int/unsigned int，不支持 float。
 *   但对于非负浮点数，IEEE 754 的位表示满足"值越大，整数表示越大"，
 *   因此可以用 __float_as_int 转为 int 后做 atomicMax，
 *   结果再 __int_as_float 转回 float，等价于 atomicMax for float。
 *   前提：所有值 >= 0（绝对值满足此条件）。
 *
 * @param x_new  新解向量（设备端，N）—— GS 中是就地更新后的 x
 * @param x_old  旧解向量（设备端，N）—— GS 中是迭代前的快照
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

/**
 * @brief 雅可比迭代 kernel（来自 Learn012，用于对比）
 *
 * 每个线程负责矩阵的一行，用 x_old 的全体旧值计算 x_new 的一个分量：
 *   x_new[i] = (b[i] - sum_{j!=i} A[i][j] * x_old[j]) / A[i][i]
 * 与 GS 相反：各分量互不依赖，可任意多 block 并行，但收敛更慢。
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

    for (int j = 0; j < N; j++) {
        float a_ij = A[row * N + j];
        if (j == row) {
            diag = a_ij;        // 记录对角元素
        } else {
            dot_product += a_ij * x_old[j];
        }
    }

    x_new[row] = (b[row] - dot_product) / diag;
}

// ============================================================================
// 主机端（CPU）函数
// ============================================================================

/// 迭代轨迹最多打印的前几步（观察 GS "即用新值" 的效果）
const int TRACE_ITERS = 6;

/**
 * @brief 在控制台打印一步迭代的解向量与最大差
 */
static void print_trace(int iter, const float* x, float max_diff, int N) {
    printf("    iter %2d: x = (", iter);
    for (int i = 0; i < N; i++) {
        printf("%s%.6f", i ? ", " : "", x[i]);
    }
    printf("),  max diff = %.6e\n", max_diff);
}

/**
 * @brief CPU 高斯-赛德尔求解器（参考实现，可打印迭代轨迹）
 *
 * 与 GPU 版本算法语义相同：就地更新，算出 xi 立刻写回 x[i]。
 * 注意内层读 x[j] 时：j<i 已是新值，j>i 仍是旧值 —— 一行代码体现 GS 灵魂。
 *
 * @param A          系数矩阵（行主序，N*N）
 * @param b          右端项
 * @param x          解向量（输入初始猜测，输出结果）
 * @param N          维度
 * @param max_iter   最大迭代次数
 * @param tolerance  收敛阈值
 * @param verbose    是否打印前 TRACE_ITERS 步的迭代轨迹
 * @return 实际迭代次数
 */
int cpu_gauss_seidel(const float* A, const float* b, float* x, int N,
                     int max_iter, float tolerance, bool verbose) {
    for (int iter = 0; iter < max_iter; iter++) {
        float max_diff = 0.0f;

        // 一次完整扫描：按 i 从小到大就地更新所有分量
        for (int i = 0; i < N; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sum += A[i * N + j] * x[j];   // j<i 新值，j>i 旧值
                }
            }
            float x_new = (b[i] - sum) / A[i * N + i];

            float diff = fabsf(x_new - x[i]);
            if (diff > max_diff) max_diff = diff;

            x[i] = x_new;   // 立即写回 —— GS 的灵魂
        }

        if (verbose && iter < TRACE_ITERS) {
            print_trace(iter + 1, x, max_diff, N);
        }

        if (max_diff < tolerance) {
            return iter + 1;
        }
    }
    return max_iter;
}

/**
 * @brief CPU 雅可比求解器（来自 Learn012，增加轨迹打印，用于对比）
 *
 * 双缓冲：x_new 的所有分量都从 x_old 计算，算完一整轮后才交换。
 */
int cpu_jacobi(const float* A, const float* b, float* x, int N,
               int max_iter, float tolerance, bool verbose) {
    std::vector<float> x_old(N);
    std::vector<float> x_new(N);
    for (int i = 0; i < N; i++) x_old[i] = x[i];

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < N; i++) {
            float dot_product = 0.0f;
            float diag = 0.0f;
            for (int j = 0; j < N; j++) {
                if (j == i) {
                    diag = A[i * N + j];
                } else {
                    dot_product += A[i * N + j] * x_old[j];   // 全部用旧值
                }
            }
            x_new[i] = (b[i] - dot_product) / diag;
        }

        float max_diff = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = fabsf(x_new[i] - x_old[i]);
            if (diff > max_diff) max_diff = diff;
        }

        x_old.swap(x_new);   // 整轮算完才交换（GS 则是逐个立即写回）

        if (verbose && iter < TRACE_ITERS) {
            print_trace(iter + 1, x_old.data(), max_diff, N);
        }

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
    const int N = 3;                 // 矩阵维度
    const int max_iter = 100;        // 最大迭代次数
    const float tolerance = 1e-6f;   // 收敛阈值（无穷范数）

    std::cout << "========================================" << std::endl;
    std::cout << "  CUDA Gauss-Seidel Iterative Solver" << std::endl;
    std::cout << "  Matrix size: " << N << " x " << N << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 步骤 1：在 CPU 上构造问题 ====
    // 沿用 Learn012 的方程组，便于直接对比两种方法的迭代次数：
    //   5x + 1y + 1z =  8
    //   1x + 6y + 2z = 10
    //   2x + 1y + 7z = 12
    // 严格对角占优（|a_ii| > sum_{j!=i}|a_ij|）：
    //   行 0: 5 > 1+1 = 2；行 1: 6 > 1+2 = 3；行 2: 7 > 2+1 = 3
    // 因此 GS 与雅可比都必收敛。
    // 理论精确解（手算克拉默法则可得）：
    //   x = 106/93 = 1.1397849462...
    //   y = 33/31  = 1.0645161290...
    //   z = 115/93 = 1.2365591398...
    std::vector<float> h_A = { 5.0f, 1.0f, 1.0f,   // 第 0 行
                               1.0f, 6.0f, 2.0f,   // 第 1 行
                               2.0f, 1.0f, 7.0f }; // 第 2 行
    std::vector<float> h_b = { 8.0f, 10.0f, 12.0f };
    const float exact[3] = { 106.0f / 93.0f, 33.0f / 31.0f, 115.0f / 93.0f };

    std::cout << "[1] System constructed:" << std::endl;
    std::cout << "    5x + 1y + 1z =  8" << std::endl;
    std::cout << "    1x + 6y + 2z = 10" << std::endl;
    std::cout << "    2x + 1y + 7z = 12" << std::endl;

    // ==== 步骤 2：CPU 求解（带轨迹打印）====
    // 观察要点：GS 第 1 步的 y 用了刚算出的 x（雅可比没有），
    // 误差下降明显更快；GS 所需迭代次数约为雅可比的一半。
    std::cout << std::endl;
    std::cout << "[2] CPU Gauss-Seidel trace (in-place, use new values):" << std::endl;
    std::vector<float> h_x_cpu_gs(N, 0.0f);
    int cpu_gs_iters = cpu_gauss_seidel(h_A.data(), h_b.data(), h_x_cpu_gs.data(),
                                        N, max_iter, tolerance, /*verbose=*/true);
    std::cout << "    -> converged in " << cpu_gs_iters << " iterations" << std::endl;

    std::cout << std::endl;
    std::cout << "[3] CPU Jacobi trace (double buffer, old values only):" << std::endl;
    std::vector<float> h_x_cpu_jac(N, 0.0f);
    int cpu_jac_iters = cpu_jacobi(h_A.data(), h_b.data(), h_x_cpu_jac.data(),
                                   N, max_iter, tolerance, /*verbose=*/true);
    std::cout << "    -> converged in " << cpu_jac_iters << " iterations" << std::endl;

    // ==== 步骤 3：分配设备内存并拷贝数据 ====

    float *d_A, *d_b;
    float *d_x, *d_x_prev;      // GS：就地更新的 x + 迭代前快照（收敛判定用）
    float *d_x0, *d_x1;         // 雅可比：双缓冲
    float *d_max_diff;

    CHECK(cudaMalloc(&d_A,        (size_t)N * N * sizeof(float)));
    CHECK(cudaMalloc(&d_b,        (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_x,        (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_x_prev,   (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_x0,       (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_x1,       (size_t)N * sizeof(float)));
    CHECK(cudaMalloc(&d_max_diff, sizeof(float)));

    CHECK(cudaMemcpy(d_A, h_A.data(), (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // ==== 步骤 4：配置线程组织 ====

    const int block_size = 256;   // 每 block 256 个线程（2 的幂，归约要求）
    const int grid_size = (N + block_size - 1) / block_size;   // 向上取整
    const int smem_size = block_size * sizeof(float);          // 归约共享内存

    std::cout << std::endl;
    std::cout << "[4] Thread config:" << std::endl;
    std::cout << "    GS kernel:     1 block x " << block_size
              << " threads (row-serial, column-parallel)" << std::endl;
    std::cout << "    Jacobi kernel: " << grid_size << " blocks x "
              << block_size << " threads (one row per thread)" << std::endl;

    // CUDA Event 用于精确测量 GPU 执行时间
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // ==== 步骤 5：GPU 高斯-赛德尔求解 ====

    // 预热（warm-up）：程序中首次启动 kernel 会触发驱动加载模块 / PTX JIT，
    // 一次性开销可达上百毫秒。提前在计时窗口外启动一次同款 kernel
    // （写到无关的临时缓冲 d_x_prev 上），避免污染后续计时。
    gauss_seidel_iteration<<<1, block_size, smem_size>>>(d_A, d_b, d_x_prev, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemset(d_x, 0, N * sizeof(float)));   // 初始猜测：x = 0

    CHECK(cudaEventRecord(start));   // 计时开始

    int gs_iter;
    float gs_max_diff = tolerance + 1.0f;   // 初始值确保至少进入循环

    for (gs_iter = 0; gs_iter < max_iter; gs_iter++) {
        // 5a. 快照当前解：GS 就地更新会覆盖 x，收敛判定需要"更新前"的值
        //     （同一流内的 DtoD 拷贝与后续 kernel 保持顺序）
        CHECK(cudaMemcpyAsync(d_x_prev, d_x, N * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        // 5b. 一次完整的 GS 扫描：kernel 内按行串行、行内并行归约
        gauss_seidel_iteration<<<1, block_size, smem_size>>>(d_A, d_b, d_x, N);

        // 5c. 重置全局最大差为 0（int 0 == float 0.0f）
        CHECK(cudaMemsetAsync(d_max_diff, 0, sizeof(float)));

        // 5d. 计算无穷范数 ||x - x_prev||_inf
        compute_max_diff<<<grid_size, block_size, smem_size>>>(
            d_x, d_x_prev, d_max_diff, N);

        // 5e. 将单个标量拷回 CPU 做收敛判断
        CHECK(cudaMemcpy(&gs_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost));

        // 5f. 收敛判定
        if (gs_max_diff < tolerance) {
            gs_iter++;
            break;
        }
    }

    CHECK(cudaEventRecord(stop));        // 计时结束
    CHECK(cudaEventSynchronize(stop));   // 等待 GPU 完成

    float gs_time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&gs_time_ms, start, stop));

    std::vector<float> h_x_gs(N);
    CHECK(cudaMemcpy(h_x_gs.data(), d_x, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << std::endl;
    std::cout << "[5] GPU Gauss-Seidel solve complete" << std::endl;

    // ==== 步骤 6：GPU 雅可比求解（Learn012 方法，对比用）====

    CHECK(cudaMemset(d_x0, 0, N * sizeof(float)));   // 初始猜测：x = 0

    CHECK(cudaEventRecord(start));   // 计时开始

    int jac_iter;
    float jac_max_diff = tolerance + 1.0f;
    float* d_x_old = d_x0;   // 当前指向"旧"缓冲区
    float* d_x_new = d_x1;   // 当前指向"新"缓冲区

    for (jac_iter = 0; jac_iter < max_iter; jac_iter++) {
        // 用 x_old 计算 x_new（全体旧值）
        jacobi_iteration<<<grid_size, block_size>>>(d_A, d_b, d_x_old, d_x_new, N);

        CHECK(cudaMemsetAsync(d_max_diff, 0, sizeof(float)));
        compute_max_diff<<<grid_size, block_size, smem_size>>>(
            d_x_new, d_x_old, d_max_diff, N);
        CHECK(cudaMemcpy(&jac_max_diff, d_max_diff, sizeof(float), cudaMemcpyDeviceToHost));

        // 交换双缓冲：下一轮的"旧值"就是本轮的"新值"
        float* tmp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = tmp;

        if (jac_max_diff < tolerance) {
            jac_iter++;
            break;
        }
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float jac_time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&jac_time_ms, start, stop));

    std::vector<float> h_x_jac(N);
    CHECK(cudaMemcpy(h_x_jac.data(), d_x_old, N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[6] GPU Jacobi solve complete" << std::endl;

    // ==== 步骤 7：结果验证与对比 ====

    // 7a. 计算残差 ||Ax - b||（无穷范数与 L2 范数）
    float l2_residual = 0.0f;
    float max_residual = 0.0f;
    for (int i = 0; i < N; i++) {
        float ax = 0.0f;
        for (int j = 0; j < N; j++) {
            ax += h_A[i * N + j] * h_x_gs[j];
        }
        float res = fabsf(ax - h_b[i]);
        l2_residual += res * res;
        if (res > max_residual) max_residual = res;
    }
    l2_residual = sqrtf(l2_residual);

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Results (Gauss-Seidel)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Iterations:      " << gs_iter << std::endl;
    std::cout << "  Final max diff:  " << gs_max_diff << std::endl;
    std::cout << "  L2 residual:     " << l2_residual << std::endl;
    std::cout << "  Max residual:    " << max_residual << std::endl;
    std::cout << "  GPU time:        " << gs_time_ms << " ms" << std::endl;

    // 7b. 与理论精确解对比
    float exact_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_x_gs[i] - exact[i]);
        if (err > exact_err) exact_err = err;
    }
    std::cout << "  Max |x - x_exact|: " << exact_err << std::endl;

    // 7c. GPU 与 CPU 一致性
    //     注意：GPU 行内点积用树形归约（两两相加），CPU 按列号顺序累加，
    //     浮点加法结合顺序不同会带来 ~1e-7 量级的舍入差异，属正常现象。
    float cpu_gpu_diff = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_x_gs[i] - h_x_cpu_gs[i]);
        if (diff > cpu_gpu_diff) cpu_gpu_diff = diff;
    }
    std::cout << "  CPU vs GPU diff: " << cpu_gpu_diff
              << " (CPU iters: " << cpu_gs_iters << ")" << std::endl;

    // 7d. GS 与雅可比对比 —— 本程序的核心结论
    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Gauss-Seidel vs Jacobi (same system)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  GS     iterations: " << gs_iter
              << ", time: " << gs_time_ms << " ms (1 block)" << std::endl;
    std::cout << "  Jacobi iterations: " << jac_iter
              << ", time: " << jac_time_ms << " ms" << std::endl;
    std::cout << "  Iteration ratio (Jacobi/GS): "
              << (double)jac_iter / (double)gs_iter << std::endl;
    std::cout << "  -> GS converges with about half the iterations," << std::endl;
    std::cout << "     but its row dependency limits GPU parallelism" << std::endl;
    std::cout << "     to a single block; Jacobi scales to all SMs." << std::endl;
    std::cout << "     For large sparse systems from PDE stencils, the fix is" << std::endl;
    std::cout << "     red-black / multi-colored Gauss-Seidel ordering." << std::endl;

    // 7e. 打印解向量
    std::cout << std::endl;
    std::cout << "  Solution:" << std::endl;
    int print_count = N < 5 ? N : 5;   // 不超过 N，避免越界读取
    for (int i = 0; i < print_count; i++) {
        printf("    %c = %.10f   (exact: %.10f)\n", 'x' + i, h_x_gs[i], exact[i]);
    }
    std::cout << "----------------------------------------" << std::endl;

    // ==== 步骤 8：释放资源 ====

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_x_prev));
    CHECK(cudaFree(d_x0));
    CHECK(cudaFree(d_x1));
    CHECK(cudaFree(d_max_diff));

    return 0;
}
