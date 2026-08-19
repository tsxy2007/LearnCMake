# Learn017 — 模板方法模式（Template Method）：骨架固定，步骤可变

GoF 23 种设计模式中的**行为型**模式。一句话：

> 在基类中定义算法的**骨架**（骨架内的步骤顺序固定、不可改写），
> 把其中**会变化的步骤**声明为虚函数，延迟到子类实现。

系列口诀链（承接 [Learn015](../Learn015)、[Learn016](../Learn016)）：

| 场景 | 模式 | 手段 |
|---|---|---|
| 两个维度都要独立扩展（Shape × Renderer） | Bridge | 组合，两边各一棵树 |
| 整个算法可替换（换一种解法器） | Strategy | 组合，运行时可换 |
| 骨架相同、只有个别步骤不同 | **Template Method** | **继承，子类填步骤** |

---

## 1. 问题：骨架重复

雅可比 / 高斯-赛德尔 / CG（Learn012–014）的主循环骨架完全同构：

```
初始化工作区 → 循环{ 做一步迭代 → 查收敛 } → 汇报统计
```

三种解法各自为战时，这段骨架被抄了三遍：判据从绝对残差换成相对残差、
或统计口径一变，就要同时改三处、测三处。

## 2. 核心思想：骨架上收基类，步骤下放子类

```
        AbstractClass                      ConcreteClass
        ─────────────                      ─────────────
  ┌───────────────────────────┐     ┌────────────────────────────┐
  │ IterativeSolver           │     │ JacobiSolver               │
  │  A_ b_ x_ stats_          │     │   initialize() 分配双缓冲  │
  │                           │◄────┤   iterate()    全用旧值    │
  │ + run()   ← 模板方法(非虚)│ 继承 │ GaussSeidelSolver          │
  │ # initialize() = 0  原语  │     │   iterate()    就地更新    │
  │ # iterate()      = 0 原语 │     │ ConjugateGradientSolver   │
  │ # on_iteration() {}  钩子 │     │   iterate()    三项递推    │
  └───────────────────────────┘     └────────────────────────────┘
```

`run()` 的骨架（**非虚** —— 子类不许改写流程本身）：

```cpp
SolveStats run(int max_iter, float tol) {
    initialize();                                  // 步骤 1：原语
    for (int iter = 0; iter < max_iter; ++iter) {
        iterate();                                 // 步骤 2：原语
        stats_.final_residual = residual_norm(...);// 步骤 3：公共判据
        on_iteration(iter);                        // 步骤 4：钩子
        if (stats_.final_residual < tol || stop_requested_)
            break;                                 // 步骤 5：公共终止
    }
    return stats_;
}
```

**好莱坞原则**（Don't call us, we'll call you）：子类不驱动流程，只被
骨架回调。库是你调它，框架是它调你 —— 模板方法是框架的最小雏形。

## 3. 角色与"三种方法"

| 角色 / 方法 | 本例对应 | 说明 |
|---|---|---|
| AbstractClass | `IterativeSolver` | 骨架 + 公共状态 + 公共判据 |
| ConcreteClass | `JacobiSolver` / `GaussSeidelSolver` / `ConjugateGradientSolver` | 只实现原语 |
| **模板方法** | `run()` | 非虚；骨架是**不变式** |
| **原语操作** | `initialize()` / `iterate()` | 纯虚；protected，只有骨架能调 |
| **钩子方法** | `on_iteration()` | 默认空实现；子类**可选**覆盖 |

钩子 vs 原语：原语是必填步骤，钩子是可选的插入点 ——
`GaussSeidelSolver::initialize()` 是空实现、`TracedGaussSeidel` 只覆盖
`on_iteration()` 打印残差轨迹，都不需要动骨架一行。

## 4. 控制权在框架：`stop_requested_`

步骤函数无权 `break` 循环（循环属于骨架）。CG 遇到 `pAp == 0`
（方向退化到机器精度）时，通过 protected 成员 `stop_requested_`
**请求**骨架终止 —— "请假条"式的单向通信，控制权始终在基类。

## 5. 与策略模式对比（Learn016 同题两种解法）

实测数字完全一致（Jacobi 16 / GS 7 / CG 3 次收敛），分解方式不同：

| | Template Method（本例） | Strategy（Learn016） |
|---|---|---|
| 关系 | 继承（is-a） | 组合（has-a） |
| 变化单位 | 固定骨架中的**步骤** | **整个算法** |
| 运行时切换 | 否（类型构造时定型） | 是（`set_strategy`） |
| 不变式 | 基类强制骨架不可变 | Context 对算法一无所知 |
| 步骤间共享状态 | protected 成员直接共享 | 需经接口参数传递 |

二者可叠加：一个策略对象内部完全可以用模板方法搭骨架。
选择依据：**连循环结构、判据都可能不同 → Strategy；
循环骨架稳定、只有"一步"不同 → Template Method。**

## 6. 现代 C++ 关联

- **NVI（Non-Virtual Interface）惯用法**：公有接口非虚、虚函数收进
  protected/private —— 本例 `run()` 公有非虚 + 原语受保护，正是 NVI
  的应用。它比"全 public virtual"更能守住不变式（虚函数即子类可改点，
  能少暴露就少暴露）。
- **CRTP（奇异递归模板）**：模板方法的编译期版本 ——
  `class Derived : public Base<Derived>`，基类静态调用
  `derived().step()`，零虚函数开销、可内联；代价同样是失去运行时
  多态（类型编译期定型）。
- C++ 没有语言级 "final method on non-virtual" 问题：非虚天然不可改写；
  若骨架本就是虚函数，可标 `final` 防子类再覆盖。

## 7. 何时使用

✔ 适合：

- 多个算法的流程骨架相同，仅个别步骤不同（本例的三种迭代法）；
- 想把"判据 / 统计 / 生命周期管理"这类横切逻辑收敛到一处；
- 做框架：固定生命周期（初始化→循环→清理），把业务步骤留给使用者。

✘ 不适合：

- 各实现连循环结构都不同（硬套会把 iterate() 做成万能大杂烩，
  改用 Strategy）；
- 需要运行时切换实现（继承关系编译期定型，换不了 —— 用 Strategy）；
- 步骤数量因实现而异（模板方法假设骨架的步骤是固定的）。

## 8. 构建与运行

本目录含**同一模式的两个示例**（数值版 + 游戏版，见第 10 节）：

```bash
cmake --build build --target Learn017 --config Debug          # 数值版
./build/Learn017/Debug/Learn017.exe

cmake --build build --target Learn017Weapon --config Debug    # 游戏版
./build/Learn017/Debug/Learn017Weapon.exe
```

程序依次演示：

1. 固定在基类的五步骨架；
2. 三个具体类只写 `initialize` + `iterate`，产出与 Learn016 相同的
   对比表（Jacobi 16 / GS 7 / CG 3 次收敛，均命中精确解）；
3. 钩子方法：`TracedGaussSeidel` 只覆盖 `on_iteration` 打印残差轨迹；
4. 与策略模式的逐项对照。

## 9. 实战中的使用案例

以下都是"框架定骨架、用户填步骤"的真实系统。识别标志：
**是它在调你，不是你在调它**（好莱坞原则）。

### 案例 1：单元测试框架 —— GoogleTest / pytest

| 骨架（框架持有） | 步骤（用户填写） |
|---|---|
| `RUN_ALL_TESTS` → 对每个用例：`SetUp()` → `TestBody()` → `TearDown()` | 继承 `::testing::Test`，写 `SetUp` / 测试体 / `TearDown` |

用户从不写 `main` 循环（除非定制 runner）；失败统计、超时控制、
用例注册全部由骨架统一维护 —— 模板方法 + 注册表的经典组合。

### 案例 2：Web 框架的生命周期 —— Java Servlet / Unity

| 骨架 | 步骤 |
|---|---|
| `HttpServlet.service()`：解析请求 → 分发 → 渲染响应 | `doGet()` / `doPost()` |
| Unity 引擎：`Awake → Start → Update(每帧) → OnDestroy` | 组件脚本覆盖任意阶段的回调 |

用户代码是"被调用者"：引擎/容器持有主循环，业务只提供插入点。

### 案例 3：编译器与构建流水线

| 骨架 | 步骤 |
|---|---|
| 编译流水线：词法 → 语法 → 语义 → IR 生成 → 代码生成 | 各目标平台后端实现 `codegen()` 步骤（LLVM 的各 Target） |
| CMake/构建系统：configure → generate → build → test | `CMAKE_..._HOOK`、各语言工具链模块填规则 |

前端骨架稳定多年，后端步骤随目标平台无限扩展 ——
"骨架不变式"在这里价值最大。

### 案例 4：C++ 工程实践 —— NVI 与 CRTP

- **NVI**：公有非虚 `draw()` 做参数校验/日志，再调私有虚 `do_draw()`
  —— 大量 C++ 代码规范（含部分标准库实现、Chromium style）推荐形态；
- **CRTP 静态多态**：Eigen 表达式模板、`std::enable_shared_from_this`
  都是"编译期模板方法"：基类调 `static_cast<Derived*>(this)` 的步骤，
  零虚调用开销，广泛用于数值/游戏等热路径。

### 练习建议

给本例加第 4 个具体类 `SorSolver`（逐次超松弛，GS 的推广：
`x_new = x + w * (gs_value - x)`，`0 < w < 2`）：只需实现
`initialize()`（存 relax factor）与 `iterate()` 两个函数，骨架、判据、
统计一行不碰 —— 亲手体会"填空式扩展"；再试把 `w` 扫过 0.5~1.8，
在钩子里收集收敛步数，观察 w=1 附近退化为 GS、w 过冲发散的现象。

## 10. 同一模式的第二个示例：武器攻击流程（weapon.cpp）

游戏是模板方法的高产领域 —— **武器攻击流程**（`Learn017Weapon`）
与迭代解法器是同一个模式、另一个领域：

```
门槛检查(冷却+资源) → 瞄准 → 消耗资源 → 攻击动作
                    → 伤害掷骰 → 结算扣血 → 命中钩子 → 进入冷却
```

| 角色 / 方法 | 武器版（weapon.cpp） | 数值版（main.cpp） |
|---|---|---|
| AbstractClass | `Weapon`（门槛/结算/冷却统一管理） | `IterativeSolver` |
| ConcreteClass | `Sword` / `Bow` / `MagicWand` | `Jacobi` / `GS` / `CG` |
| 模板方法（非虚） | `attack(target)` | `run(max_iter, tol)` |
| 纯虚原语 | `has_resource` / `consume_resource` / `fire` | `initialize` / `iterate` |
| 带默认的钩子 | `aim` / `roll_damage` | — |
| 默认空的钩子 | `on_hit` | `on_iteration` |
| 公共控制 | 冷却、资源门槛、扣血结算 | 残差判据、统计、终止 |

游戏版额外体现了三个点：

1. **门槛即不变式**：冷却与资源检查是骨架第 0 步，任何武器子类都
   无法绕过 —— 游戏服防"跳过冷却开火"式外挂的第一道门就是唯一的
   公有入口 `attack()`（NVI 惯用法）；
2. **抽象出"资源"统一三种武器**：近战的"无弹药"（恒真）、弓的箭、
   法杖的法力，藏在同一个纯虚原语 `has_resource()` 后面；
3. **钩子的三种用法各占一类**：`Sword` 覆盖 `aim`（踏步近身）与
   `roll_damage`（20% 暴击 ×2），`MagicWand` 覆盖 `on_hit`
   （灼烧溅射追加伤害），`Bow` 一个钩子都不覆盖 —— 全都合法。

随机数用固定种子的 `mt19937`，演示输出可复现。
