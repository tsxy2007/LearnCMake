# Learn014 — CUDA 共轭梯度法（CG）求解对称正定线性方程组

承接 [Learn012](../Learn012)（雅可比迭代）、[Learn013](../Learn013)（高斯-赛德尔迭代）。
前两者是"固定格式"的分裂迭代法，本节实现完全不同的一类方法——
**Krylov 子空间迭代法**的代表：共轭梯度法（Conjugate Gradient, CG）。

```
5x + 1y + 1z =  7          精确解：x = y = z = 1（构造 b = A·(1,1,1)）
1x + 6y + 1z =  8
1x + 1y + 7z =  9          对称 + 严格对角占优 + 正对角 ⇒ 正定（SPD）
```

实测：**最速下降 12 步收敛，CG 恰好 3 步（= N）收敛到机器精度**。

---

## 1. 适用条件：A 必须对称正定（SPD）

CG 的硬性要求：

- **对称**：`A = Aᵀ`；
- **正定**：任意 `x ≠ 0` 有 `xᵀAx > 0`（等价于所有特征值 > 0）。

本例矩阵对称、严格对角占优且对角元为正——对称矩阵的 Gershgorin 圆盘
全部落在正半轴（`λ ≥ min_i(a_ii − Σ_{j≠i}|a_ij|) > 0`），故为 SPD。
（Learn012/013 的矩阵不对称，CG 不适用——所以本节换了一个对称矩阵。）

## 2. 出发点：解方程 = 求二次凸函数的极小点

A 对称正定时，定义二次函数

```
f(x) = 1/2 · xᵀAx − bᵀx
```

其梯度 `∇f(x) = Ax − b = −r`（`r = b − Ax` 正是残差）。于是

```
Ax = b  ⇔  ∇f = 0  ⇔  min f(x)
```

解方程组变成了"下山找极小点"——这是 CG 全部直觉的来源。
f 的等值线是一族同心椭圆（由 A 的特征向量定向、特征值定形），
A 病态（条件数 κ 大）时椭圆极扁，"下山"就会变得困难。

## 3. 铺垫：最速下降（Steepest Descent）及其锯齿

最朴素的下山法：每步沿负梯度方向（即残差 `r`）走"最优步长"
（精确线搜索，使 f 沿该方向最小）：

```
alpha = (rᵀr) / (rᵀAr)
x     = x + alpha · r
r     = r − alpha · A r
```

**死穴**：精确线搜索使相邻两步的方向互相垂直（本次梯度 ⊥ 上次行进方向），
路线呈**锯齿形**，同一方向反复来回。收敛因子 `(κ−1)/(κ+1)`——只依赖 κ。

本程序实测（同一方程组，容差 1e-6）：

```
iter 1: ||r|| = 8.6e-1
iter 2: ||r|| = 1.3e-1
iter 3: ||r|| = 3.7e-2     ← 每步只降一个固定倍数，
iter 4: ||r|| = 1.1e-2        迟迟到不了机器精度
...
12 步后收敛
```

## 4. 核心思想：共轭（A-正交）方向

**定义**：方向组 `p_0, p_1, ...` 称为 A-共轭（A-正交），若

```
p_iᵀ A p_j = 0    (i ≠ j)
```

即它们在 **A-内积** `<x, y>_A = xᵀAy` 的意义下两两正交。
当 `A = I` 时退化为普通正交。

**关键定理**：依次沿一组共轭方向做精确线搜索，则

1. 每一步把该方向上的误差分量**彻底清零**，且以后永不复活
   （不像最速下降那样走回头路）；
2. **最多 N 步到达精确解**（A-内积意义下的"坐标下降"，N 个方向张满全空间）。

剩下的问题：怎么廉价地构造一组共轭方向？直接对 N 个单位向量做
A-内积下的 Gram-Schmidt 需要 O(N²) 次向量运算并存储所有历史方向——太贵。

## 5. CG 的巧妙之处：三项递推生成共轭方向

CG 发现（Hestenes & Stiefel, 1952）：如果把"基"取为迭代中产生的**残差**，
则残差天然两两正交（`r_iᵀ r_j = 0, i ≠ j`），Gram-Schmidt 正交化的
绝大部分项自动为零，只剩**相邻一项**——于是共轭方向只需三项递推：

```
p_0 = r_0
p_{k+1} = r_{k+1} + beta_k · p_k        ← 只用到上一步的方向！
beta_k  = (r_{k+1}ᵀ r_{k+1}) / (r_kᵀ r_k)
```

（β 的这个形式即 Fletcher–Reeves 型；对 CG 还有等价的
`β = −r_{k+1}ᵀAp_k / p_kᵀAp_k` 等变体。）

## 6. 完整算法

```
x_0 = 0;  r_0 = b − A x_0;  p_0 = r_0
for k = 0, 1, 2, ...
    Ap      = A p_k                                // 每迭代唯一的矩阵乘
    alpha   = (r_kᵀ r_k) / (p_kᵀ A p_k)            // 精确线搜索步长
    x_{k+1} = x_k + alpha · p_k
    r_{k+1} = r_k − alpha · Ap                      // 递推残差，免重新算 b−Ax
    若 ||r_{k+1}||_2 < tol：停止
    beta    = (r_{k+1}ᵀ r_{k+1}) / (r_kᵀ r_k)
    p_{k+1} = r_{k+1} + beta · p_k
```

每迭代的计算量：**1 次矩阵-向量乘 + 3 个点积 + 3 个 axpy**——
矩阵只以"乘法"出现（不需要 L/U 分裂、不需要对角元非零），
稀疏矩阵时成本仅 O(nnz)，这是 CG 成为大规模稀疏问题主力的根本原因。

## 7. 收敛速度

A-范数误差满足

```
||x_k − x*||_A ≤ 2 · ((√κ − 1)/(√κ + 1))^k · ||x_0 − x*||_A
```

- 依赖 **√κ** 而非 κ——与最速下降的 `(κ−1)/(κ+1)` 相比，
  病态问题（κ 大）下优势是数量级的；
- 精确算术下 **≤ N 步收敛**（浮点下会在 N 步附近达到机器精度量级）。

本程序实测对比（同方程组、同容差 1e-6）：

```
最速下降：12 步
CG：       3 步（= N），第 3 步 ||r|| 从 9.9e-2 直落到 9.5e-8

两法轨迹的对比还可见一个细节：
  第 1 步两者完全相同（首方向都是 r_0，alpha = 0.122940），
  第 2 步开始分道扬镳（0.199938 vs 0.201175）——
  正是"用不用共轭方向"开始起作用的时刻。
```

## 8. GPU 实现（[main.cu](main.cu)）

CG 一步 = **1 个 SpMV + 3 个 dot + 3 个 axpy**，全是可并行的
BLAS1/2 原语——不像高斯-赛德尔有行间串行依赖，CG 天然适合 GPU。

| kernel | 功能 | 并行策略 |
|---|---|---|
| `matvec` | `y = A·x` | 一行一线程（Learn004 模式） |
| `dot_product` | `x·y` | 块内共享内存树形归约 + `atomicAdd` 合并 |
| `axpy` | `y += α·x` | 一元素一线程 |
| `update_direction` | `p = r + β·p` | 一元素一线程 |

实现要点：

- **atomicAdd 原生支持 float**——对照 Learn012 中 `atomicMax` 不支持
  float、必须用 `__float_as_int` 位技巧绕行的情形；代价是各 block
  的累加顺序不定（浮点加法不满足结合律），对本例无影响；
- **主机驱动迭代循环**：α、β 是标量，本程序每迭代把点积拷回 CPU 计算
  （每轮 3 次 4 字节拷贝 + 隐式同步）。N=3 的教学场景无妨，但大规模
  问题上这些同步是瓶颈——生产实现（cuSPARSE/cuBLAS、PETSc）用
  **设备端标量 + cublasDot 把结果留在显存**，彻底消除 host 同步；
- 稀疏矩阵时把 `matvec` 换成 SpMV（如 cuSPARSE 的 CSR mv）即可，
  CG 其余部分一行不改；
- 计时窗口外预热一次 kernel，排除首次启动的模块加载 / PTX JIT
  一次性开销（Learn013 的经验）；
- **递推残差 vs 真实残差**：循环里的 `r` 是递推量（`r -= α·Ap`），
  与真实残差 `b − Ax` 在浮点下会漂移；程序末尾同时打印两者验证。

## 9. 实测输出摘录

```
[3] CPU conjugate gradient trace (conjugate directions):
    iter  1: alpha = 0.122940, ||r||_2 = 8.563175e-01
    iter  2: alpha = 0.201175, ||r||_2 = 9.891419e-02
    iter  3: alpha = 0.208416, ||r||_2 = 9.548668e-08
    -> converged in 3 iterations (<= N = 3 in exact arithmetic)

  Results (Conjugate Gradient)
    Iterations:            3 (theory: <= N = 3)
    Recursive ||r||_2:     1.5957e-07
    True ||b - Ax||_2:     0
    Max |x - x_exact|:     0
    CPU vs GPU diff:       0 (CPU iters: 3)
    GPU time:              0.680928 ms

  Steepest Descent vs Conjugate Gradient
    SD iterations: 12 (zigzag on gradient)
    CG iterations: 3 (conjugate directions,
                   exact in at most N steps)

  Solution:
    x = 1.0000000000   (exact: 1.0000000000)
    y = 1.0000000000   (exact: 1.0000000000)
    z = 1.0000000000   (exact: 1.0000000000)
```

## 10. 三种迭代法横向对比（Learn012 ~ Learn014）

| | 雅可比 | 高斯-赛德尔 | 共轭梯度 |
|---|---|---|---|
| 适用矩阵 | 任意（对角非零） | 任意（对角非零） | **仅对称正定** |
| 每迭代核心 | SpMV（旧值） | SpMV（就地混用新旧值） | SpMV + 3 dot + 3 axpy |
| 收敛速度 | 因子 `ρ_J` | `ρ_GS ≈ ρ_J²` | **≤ N 步；因子 `(√κ−1)/(√κ+1)`** |
| 数据依赖 | 无（天然并行） | 行间串行链 | 无（BLAS 原语全并行） |
| GPU 上的形态 | 任意多 block | 单 block / 红黑排序 | 任意多 block |

注：三个例子用的矩阵不同（014 必须对称），迭代次数不能跨课直接比较。
工程实践：大规模稀疏 SPD 系统（PDE 离散等）的标准答案是
**预条件共轭梯度（PCG）**——先用一个近似可逆的预条件子把 κ 压小，
再跑 CG；非对称系统则换 BiCGSTAB / GMRES 等 Krylov 方法。

## 11. 构建与运行

```bash
# 在仓库根目录
cmake -S . -B build
cmake --build build --config Debug --target Learn014
./build/Learn014/Debug/Learn014.exe
```

## 参考

- Learn012 — 雅可比迭代（双缓冲、共享内存 max 归约、atomicMax 位技巧）
- Learn013 — 高斯-赛德尔迭代（单 block 按行串行、行内归约）
- Shewchuk, *An Introduction to the Conjugate Gradient Method Without
  the Agonizing Pain* —— 公认最好的 CG 入门讲义（椭圆/正交的几何直觉）
- Trefethen & Bau, *Numerical Linear Algebra* Lecture 38 —— Krylov 方法
- Saad, *Iterative Methods for Sparse Linear Systems* —— PCG 与生产实现
