# Learn013 — CUDA 高斯-赛德尔（Gauss-Seidel）迭代法求解线性方程组

承接 [Learn012](../Learn012)（雅可比迭代）。本程序用 CUDA 实现高斯-赛德尔迭代求解
Ax = b，并在**同一方程组**上对比两种方法的收敛速度。

```
5x +  y +  z =  8          精确解：x = 106/93 ≈ 1.1398
 x + 6y + 2z = 10                   y = 33/31  ≈ 1.0645
2x +  y + 7z = 12                   z = 115/93 ≈ 1.2366
```

---

## 1. 直觉：雅可比法的"浪费"

雅可比迭代用**上一步的全部旧值**计算本步的全部新值：

```
xi^(k+1) = (b_i - Σ_{j≠i} a_ij * xj^(k)) / a_ii
```

设想按 x1 → x2 → x3 的顺序逐个计算：算 x2 时，x1 的新值明明**已经算出来了**，
雅可比却还用旧值——信息被浪费了。

高斯-赛德尔方法的改进只有一句话：

> **一旦算出新值，立即投入使用。**

## 2. 迭代公式

```
xi^(k+1) = (b_i - Σ_{j<i} a_ij * xj^(k+1)      ← 本轮新值（已算出）
                - Σ_{j>i} a_ij * xj^(k) ) / a_ii   ← 上轮旧值（还没算到）
```

- `j < i` 的项用本轮刚算出的新值；
- `j > i` 的项还只能用上一轮的旧值；
- 实现上只需**就地（in-place）更新**一个数组 `x`：按 i 从小到大扫描，
  算出 `xi` 立刻写回 `x[i]`，后面读 `x[j]`（j<i）时自然读到新值。
  **不需要雅可比那样的双缓冲。**

### 伪代码

```
x = 初始猜测（如全零）
repeat
    max_diff = 0
    for i = 0 .. N-1                 // 从小到大扫描
        sum = Σ_{j≠i} a_ij * x[j]    // j<i 读到的是新值，j>i 是旧值
        x_new = (b_i - sum) / a_ii
        max_diff = max(max_diff, |x_new - x[i]|)
        x[i] = x_new                 // 立即写回 —— GS 的灵魂
until max_diff < tolerance
```

## 3. 矩阵分裂视角

把 A 拆成 `A = L + D + U`（严格下三角 + 对角 + 严格上三角）：

| 方法 | 迭代公式 | 迭代矩阵 |
|---|---|---|
| 雅可比 | `x^(k+1) = D⁻¹ (b − (L+U) x^(k))` | `B_J = −D⁻¹(L+U)` |
| 高斯-赛德尔 | `x^(k+1) = (D+L)⁻¹ (b − U x^(k))` | `B_GS = −(D+L)⁻¹U` |

迭代法收敛 ⇔ 迭代矩阵谱半径 `ρ(B) < 1`（越小收敛越快）。

直观理解 GS 公式：把 `xi^(k+1) = (b_i − Σ_{j<i} a_ij xj^(k+1) − Σ_{j>i} a_ij xj^(k)) / a_ii`
整理成 `a_ii·xi^(k+1) + Σ_{j<i} a_ij·xj^(k+1) = b_i − Σ_{j>i} a_ij·xj^(k)`，
左边恰好是 `(D+L)·x^(k+1)` 的第 i 行，右边是 `(b − U·x^(k))` 的第 i 行。

## 4. 手算示例（初值 (0,0,0)）

**高斯-赛德尔**（后面的方程用刚算出的新值）：

```
第 1 步：  x = (8 − 0 − 0)/5        = 1.6
           y = (10 − 1.6 − 2·0)/6   = 1.4      ← 用了新的 x = 1.6
           z = (12 − 2·1.6 − 1.4)/7 ≈ 1.0571   ← 用了新的 x、y
第 2 步：  x ≈ 1.1086,  y ≈ 1.1295,  z ≈ 1.2362   （z 已贴近精确解 1.2366）
```

**雅可比**（全部用旧值 0）：

```
第 1 步：  x = 1.6,  y ≈ 1.6667,  z ≈ 1.7143
第 2 步：  x ≈ 0.9238,  y ≈ 0.8286,  z ≈ 1.0190   （离精确解还远）
```

GS 每"扫"一遍矩阵，信息就从方程组头部传到尾部一次，所以误差下降更快。

## 5. 收敛条件与速度

| 条件 | 雅可比 | 高斯-赛德尔 |
|---|---|---|
| A 严格对角占优（`|a_ii| > Σ_{j≠i}|a_ij|`） | 必收敛 | 必收敛 |
| A 对称正定（SPD） | 不保证 | 必收敛 |
| 一般情形 | `ρ(B_J) < 1` ⇔ 收敛 | `ρ(B_GS) < 1` ⇔ 收敛 |

定量规律：对"相容排序"的矩阵（如三对角阵）有 `ρ_GS = ρ_J²`。
谱半径小于 1 时平方更小，因此 **GS 的迭代次数通常约为雅可比的一半**。

本程序实测（容差 1e-6）：

```
GS     9 次迭代收敛
Jacobi 19 次迭代收敛
比值 Jacobi/GS = 2.11
```

## 6. GPU 并行化的核心困难（本例的课题）

雅可比"各未知数互不依赖"，一个线程管一行，天然并行。
GS 却存在**循环携带依赖**（loop-carried dependency）：

```
xi^(k+1) 依赖 xj^(k+1) (j < i)  →  必须等前面的未知数算完
```

实际工程中的对策：

1. **红黑 / 多色排序**（PDE 五点 stencil 的经典解法）：把未知数按依赖图
   着色，同色未知数互不依赖、可并行更新，然后逐色串行推进；
2. **分块 GS**：块间串行、块内退化为雅可比并行；
3. **异步 / 混沌迭代**：直接就地乱序更新，对角占优时仍收敛（但结果不确定）。

## 7. 本程序的 CUDA 实现

采用教学上最清晰的确定性方案——**"按行串行、行内并行"的单 block kernel**
（`gauss_seidel_iteration`，见 [main.cu](main.cu)）：

```cpp
__global__ void gauss_seidel_iteration(const float* A, const float* b,
                                       float* x, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;

    for (int i = 0; i < N; i++) {              // 外层：按行串行（GS 依赖顺序）
        int j = (int)tid;                       // 内层：行内并行
        sdata[tid] = (j < N && j != i) ? A[i*N + j] * x[j] : 0.0f;
        __syncthreads();

        for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {   // 树形归约求和
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) x[i] = (b[i] - sdata[0]) / A[i*N + i];   // 就地写回
        __syncthreads();                        // 写入对下一行的读取可见
    }
}
```

要点：

- **必须只启动 1 个 block**：`__syncthreads()` 只在 block 内有效，
  多 block 之间无法维持行间依赖顺序；
- **行间串行**由两处 `__syncthreads()` 保证（写回后、进入下一行前），
  barrier 之前的全局内存写入对 block 内所有线程可见，因此 GS 的
  数据依赖在单 block 内是安全的；
- **行内并行**复用 Learn012 的共享内存树形归约模式（max 归约换成加法归约）；
- 所有线程必须执行到每一个 `__syncthreads()`，因此**不允许**像
  `jacobi_iteration` 那样让越界线程提前 `return`；
- 收敛判定：每轮先用 DtoD 拷贝对 `x` 做快照（GS 就地更新会覆盖旧值），
  再复用 Learn012 的 `compute_max_diff` 归约 kernel 求 `‖x − x_prev‖∞`。

这正是稀疏三角方程组求解（SpTRSV）中 GS 求解器的经典结构：
**外层按行串行、内层按列并行**。

## 8. 实测输出摘录

```
[2] CPU Gauss-Seidel trace (in-place, use new values):
    iter  1: x = (1.600000, 1.400000, 1.057143),  max diff = 1.600000e+00
    iter  2: x = (1.108571, 1.129524, 1.236191),  max diff = 4.914286e-01
    ...
    -> converged in 9 iterations

[3] CPU Jacobi trace (double buffer, old values only):
    iter  1: x = (1.600000, 1.666667, 1.714286),  max diff = 1.714286e+00
    iter  2: x = (0.923810, 0.828571, 1.019048),  max diff = 8.380952e-01
    ...
    -> converged in 19 iterations

  Results (Gauss-Seidel)
    L2 residual:     9.53674e-07
    Max |x - x_exact|: 1.19209e-07
    CPU vs GPU diff: 0

  Gauss-Seidel vs Jacobi (same system)
    GS     iterations: 9,   time: 0.949408 ms (1 block)
    Jacobi iterations: 19,  time: 1.28442 ms
    Iteration ratio (Jacobi/GS): 2.11111
```

一个值得注意的现象：即便迭代次数少一半，GS 的单次迭代成本更高
（快照拷贝 + 单 block 内多轮 barrier），小规模问题上两者耗时同一量级；
**大规模问题上 GS 的串行依赖才是真正瓶颈**——工程出路是红黑/多色排序，
可作为后续课题（如 Learn014）。

细节：程序在计时窗口外启动了一次同款 kernel 做**预热**——首次启动 kernel
会触发驱动加载模块 / PTX JIT，一次性开销可达上百毫秒，不排除会污染计时。

## 9. 构建与运行

```bash
# 在仓库根目录
cmake -S . -B build
cmake --build build --config Debug --target Learn013
./build/Learn013/Debug/Learn013.exe
```

## 参考

- Learn012 — 雅可比迭代（双缓冲、共享内存 max 归约、atomicMax 处理 float 的位技巧）
- 《数值分析》任意教材的"迭代法解线性方程组"章节（矩阵分裂、谱半径收敛判据）
- Saad, *Iterative Methods for Sparse Linear Systems* — 不定方程组的 splitting 方法与并行排序
