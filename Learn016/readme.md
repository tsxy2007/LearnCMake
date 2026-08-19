# Learn016 — 策略模式（Strategy Pattern）：封装一族可替换的算法

GoF 23 种设计模式中的**行为型**模式。一句话：

> 定义一系列算法，把它们一个个封装起来，并使它们可以**互相替换**。
> 策略模式使算法的变化**独立于使用算法的客户**。

承接 [Learn015](../Learn015)（桥接模式）的口诀：
**两边都有类层次 → Bridge；只有一边可变 → Strategy** —— 本例正是
"只有一边"的典型。场景与仓库数值计算系列呼应：把 Learn012（雅可比）、
Learn013（高斯-赛德尔）、Learn014（共轭梯度）的三种解法合成一个
可在运行时整体切换的**算法族**。

---

## 1. 问题：switch 的蔓延（违反开闭原则）

同一个程序要支持多种解法时，最直觉的写法是函数内部分支：

```cpp
int solve(int method, const Matrix& A, const Vector& b, Vector& x, ...) {
    switch (method) {              // ← 坏味道在这里
        case JACOBI: /* 雅可比迭代 */ break;
        case GS:     /* 高斯-赛德尔 */ break;
        case CG:     /* 共轭梯度 */   break;
        // 新增 SOR？PCG？→ 必须回来修改这个 switch
    }
}
```

每加一种算法都要**修改**（而非新增）已测试的代码；算法细节与读题、
初始化、统计报表等上下文逻辑搅在一起；算法无法作为对象传递、
横向对比、运行中整体替换 —— 这些都违反**开闭原则（OCP）**：
对扩展开放，对修改关闭。

## 2. 核心思想：把"算法"提升为对象

```
     Context                          Strategy
     ───────                          ────────
┌────────────────────┐          ┌─────────────────────────┐
│ EquationSolver     │          │ Solver（算法族的唯一抽象）│
│  A_, b_, x_  问题  │          │  + name()               │
│  ◇ strategy_       ├─────────>│  + solve(A, b, x, ...)  │
│  + set_strategy()  │   委托   └───────────△─────────────┘
│  + run()           │                      │
└────────────────────┘          ┌───────────┼──────────────┐
                         JacobiSolver GaussSeidelSolver ConjugateGradientSolver
```

- **Context 管"事"**：保存题目、复位初值、委托求解、统计误差 ——
  与算法无关，写一遍即可；
- **Strategy 管"术"**：怎么迭代收敛，一算法一类；
- 新增算法 = 新增一个类，Context 零改动（开闭原则达成）。

## 3. 三个角色（GoF）

| 角色 | 本例对应 | 职责 |
|---|---|---|
| Strategy | `Solver` | 算法的公共接口（name / brief / solve） |
| ConcreteStrategy | `JacobiSolver` / `GaussSeidelSolver` / `ConjugateGradientSolver` | 各自实现一种算法 |
| Context | `EquationSolver` | 持有问题与当前策略，把求解委托给它 |

代码骨架（对应 `main.cpp`）：

```cpp
// Strategy：接口刻意收窄到"给题还解"
class Solver {
public:
    virtual ~Solver() = default;
    virtual std::string name() const = 0;
    virtual SolveStats solve(const std::vector<float>& A,
                             const std::vector<float>& b,
                             std::vector<float>& x,
                             int max_iter, float tol) const = 0;
};

// ConcreteStrategy：算法知识（怎么收敛）留在自己类里
class ConjugateGradientSolver final : public Solver { /* ... */ };

// Context：只见接口，不见任何具体算法
class EquationSolver {
public:
    void set_strategy(std::shared_ptr<Solver> s);   // 运行时换策略
    SolveStats run(int max_iter, float tol) {       // 委托
        x_.assign(N_, 0.0f);
        return strategy_->solve(A_, b_, x_, max_iter, tol);
    }
private:
    std::shared_ptr<Solver> strategy_;              // ★ 当前策略
};
```

## 4. 运行时换策略

与 Learn015 的"运行时换桥"同源 —— 都是组合 + 指针的能力：

```cpp
EquationSolver ctx(N, A, b, jacobi);   // 构造时注入策略
ctx.run(100, 1e-6f);                   // 16 iterations（雅可比）
ctx.set_strategy(cg);                  // ★ 整体平替
ctx.run(100, 1e-6f);                   // 3 iterations（CG）
```

同一个 Context 对象、同一道题、同一段委托代码 —— 换的只是策略。
对比表（程序实测，tol = 1e-6，三种解法均收敛到精确解 (1,1,1)）：

| Strategy | iters | 特点 |
|---|---|---|
| Jacobi | 16 | 全用旧值，双缓冲，可并行（GPU 友好） |
| Gauss-Seidel | 7 | 新值立即用，就地更新，天然串行 |
| CG | 3 | Krylov 子空间，仅限 SPD，≤ N 步精确 |

## 5. 与桥接模式的对比（呼应 Learn015）

| | Bridge（Learn015） | Strategy（本例） |
|---|---|---|
| 分类 | 结构型 | 行为型 |
| 变化维度 | **两个**（Shape 树 × Renderer 树） | **一个**（算法族） |
| Context/Abstraction 侧 | 自己也有继承层次 | 平铺一个类，不建树 |
| Implementor/Strategy 侧 | 平台/机制，偏结构 | 算法/行为，偏过程 |
| 代码形态 | 持有指针 + 委托 | 完全相同 |

> 口诀：**两边都有层次 → Bridge；只有一边 → Strategy。**

## 6. 现代 C++ 的三种"策略"形态

| 形态 | 绑定时机 | 开销 | 何时用 |
|---|---|---|---|
| 虚函数 + 指针（本例） | 运行时 | 一次间接调用 | 算法有状态/多方法/需运行时换 |
| `std::function` / lambda | 运行时 | 可能堆分配 | 签名简单的轻量策略（比较器等） |
| 模板参数（policy，如 `std::vector` 的 `Allocator`） | 编译期 | 零虚函数开销 | 性能敏感、策略在编译期已知 |

```cpp
// 轻量版：std::function 也是策略模式的运用
template <typename T>
void sort_by(std::vector<T>& v, std::function<bool(T, T)> less);  // 策略作参数

// 编译期版：模板 policy —— 标准库容器/智能指针的标准做法
template <typename T, typename Alloc = std::allocator<T>>
class vector;
```

## 7. 何时使用

✔ 适合：

- 同一任务存在多种算法/做法，需要按场景选择或运行时切换；
- 算法分支（switch/if-else）在多处重复出现；
- 想在不改动已有代码的前提下增加新做法（开闭原则）；
- 需要把"做法"作为对象传递、比较、组合。

✘ 不适合：

- 只有一种实现且可预见的将来也不会有第二种（直接写函数即可）；
- 算法间的差异只是几个参数（用配置/参数对象，不必上类）；
- 策略与上下文需要大量私有状态交互（接口会变得臃肿，考虑重构边界）。

## 8. 构建与运行

```bash
cmake --build build --target Learn016 --config Debug
./build/Learn016/Debug/Learn016.exe
```

程序依次演示：

1. 同一道 3×3 SPD 方程组 × 三种策略的对比表（迭代数、残差、误差）；
2. 各策略自述算法特点（算法知识留在策略类里）；
3. 在同一个 Context 对象上 `set_strategy` 平替算法；
4. 新增第 4 种解法（SOR/PCG）的成本：switch 版要改旧代码，
   策略版只加一个类。

## 9. 实战中的使用案例

以下都是"一族可替换算法"的真实系统。每个案例先点出"谁在换算法
（Context）× 换的是什么（Strategy）"，认出这个结构就能认出策略模式。

### 案例 1：C++ 标准库 —— 策略无处不在

| Context | Strategy（模板参数） |
|---|---|
| `std::sort` / `std::map` | 比较器 `Compare`（默认 `std::less`） |
| `std::vector` / `std::list` | 分配器 `Allocator`（policy-based design） |
| `std::unique_ptr` | 删除器 `Deleter`（默认 `default_delete`） |
| `std::unordered_map` | 哈希与相等判据 `Hash` / `KeyEqual` |

标准库用的是**编译期策略**（模板参数）：零虚函数开销，实例化时锁定，
不能再运行时换 —— 与本例的运行期策略互为补充。

### 案例 2：支付渠道 —— 电商结算

| Context | Strategy |
|---|---|
| 订单结算 `CheckoutContext`（金额、订单号、重试对账逻辑） | `PaymentStrategy` → 支付宝 / 微信支付 / 银联 / Apple Pay |

结算流程（验单 → 调支付渠道 → 记账 → 回调处理）只写一遍，
新增渠道 = 新增一个策略类 + 注册，结算主干零改动。

### 案例 3：导航路线规划

| Context | Strategy |
|---|---|
| 地图服务 `RoutePlanner`（起终点、偏好） | `RouteStrategy` → 驾车 / 公交 / 步行 / 骑行 / 避高速 |

同一份"找路 → 估算时长 → 序列化成步骤"的骨架，不同交通方式的
路网、约束、算法（Dijkstra 变体 × 交通模式）整体可换。

### 案例 4：机器学习优化器 —— PyTorch / TensorFlow

| Context | Strategy |
|---|---|
| 训练循环（前向 → 反传 → `optimizer.step()`） | `torch.optim.SGD` / `Adam` / `RMSprop` / `AdamW` |

训练循环代码不感知优化器内部（动量、自适应学习率各有各的递推），
一行构造参数换优化器即可对比收敛曲线 —— 和本例三解法对比表同构。

### 案例 5：数值库的方法选择 —— SciPy / MATLAB

| Context | Strategy |
|---|---|
| `scipy.optimize.minimize(fun, x0, method=...)` | `"BFGS"` / `"CG"` / `"Nelder-Mead"` / `"L-BFGS-B"` |
| `scipy.sparse.linalg` 稀疏求解 | `cg` / `gmres` / `bicgstab`（还有各自的预条件子） |
| MATLAB `ode45` / `ode15s` | 常微分方程的 RK45 / 刚性 BDF 策略 |

`method=` 字符串在运行时挑算法 —— 正是本目录
`set_strategy(Jacobi / Gauss-Seidel / CG)` 的工业形态；
PETSc 的 `KSPCreate` + `KSPSetType` 亦是同一结构。

### 练习建议

给本例补第 4 个策略 `SorSolver`（逐次超松弛，Learn013 的 GS 推广：
`x_new = x + w * (GS 本轮值 - x)`，`0 < w < 2`）或带对角预条件的
`PcgSolver`：只加一个类、注册一行，Context 与三个旧策略零改动 ——
亲手验证开闭原则。
