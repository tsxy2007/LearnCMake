/**
 * @file    main.cpp
 * @brief   策略模式（Strategy Pattern）—— 封装一族可替换的算法，运行时可整体换
 *
 * 定义（GoF 行为型模式）：
 *   定义一系列算法，把它们一个个封装起来，并且使它们可以互相替换。
 *   策略模式使算法的变化独立于使用算法的客户。
 *
 * 本例场景（与仓库数值计算系列呼应）：
 *   Learn012（雅可比）、Learn013（高斯-赛德尔）、Learn014（共轭梯度）
 *   分别用三个独立程序解同一个 Ax = b。若想在一个程序里支持多种解法，
 *   最直觉的写法是把算法做成函数内部分支：
 *
 *     int solve(int method, ...) {
 *         switch (method) {              // ← 坏味道在这里
 *             case JACOBI: ... break;
 *             case GS:     ... break;
 *             case CG:     ... break;
 *             // 新增 SOR？PCG？→ 必须回来改这个 switch
 *         }
 *     }
 *
 *   问题（违反开闭原则 OCP —— 对扩展开放、对修改关闭）：
 *     - 每加一种算法都要"修改"已有函数，而不是只"新增"代码；
 *     - 算法迭代细节与读题、初始化、统计、报表等上下文逻辑搅在一起；
 *     - 算法无法作为对象传递、横向比较、在运行中整体替换。
 *
 * 策略模式的做法 —— 把"算法"提升为对象：
 *
 *        Context                        Strategy
 *        ───────                        ────────
 *   ┌────────────────────┐        ┌─────────────────────────┐
 *   │ EquationSolver     │        │ Solver（算法族的唯一抽象）│
 *   │  A_, b_, x_  问题  │        │  + name()               │
 *   │  ◇ strategy_       ├───────>│  + solve(A, b, x, ...)  │
 *   │  + set_strategy()  │  委托  └───────────△─────────────┘
 *   │  + run()           │                    │
 *   └────────────────────┘        ┌───────────┼──────────────┐
 *                          JacobiSolver GaussSeidelSolver ConjugateGradientSolver
 *
 *   - Context 持有问题（矩阵、右端项）与当前策略，只负责委托与统计；
 *   - 每个算法是一个独立的类：新增算法 = 新增一个类，Context 零改动；
 *   - 两次 run() 之间可 set_strategy() 整体换算法 —— 与 Learn015 的
 *     "运行时换桥"同源，都是"组合 + 指针"带来的能力。
 *
 * 三个角色（GoF）：
 *   Strategy          —— Solver：所有算法的公共接口
 *   ConcreteStrategy  —— JacobiSolver / GaussSeidelSolver / ConjugateGradientSolver
 *   Context           —— EquationSolver：持有策略，把求解工作委托给它
 *
 * 与桥接模式的对比（呼应 Learn015 的口诀）：
 *   两边都有类层次 → Bridge；只有一边可变 → Strategy。
 *   本例 Context（EquationSolver）没有也不需要为算法建继承树，
 *   "解法"是唯一变化的维度 —— 所以是策略而非桥接；
 *   Learn015 的 Shape 树与 Renderer 树各自生长，才是两个维度。
 *
 * 现代 C++ 中"策略"的三种形态（详见 readme）：
 *   1) 虚函数 + 指针（本例）：运行时可换，代价是一次间接调用；
 *   2) std::function / lambda：轻量策略，适合签名简单的场合；
 *   3) 模板参数（policy-based design，如 std::vector 的 Allocator）：
 *      编译期绑定、零虚函数开销，但不能再运行时换。
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <cmath>      // sqrt
#include <cstdio>     // printf（表格式输出）
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>  // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

// ============================================================================
// 公共例程 —— 与具体算法无关的统计工具（各策略共用，保证对比公平）
// ============================================================================

/// 一次求解的统计信息（与算法无关的"统一报表"）
struct SolveStats {
    int iterations = 0;          // 实际迭代次数
    float final_residual = 0.0f; // 收敛时的真实残差 ||b - A x||_2
};

/**
 * @brief 真实残差 ||b - A x||_2
 *
 * 三种策略统一用它做收敛判据（CG 内部本有免费的递推残差，
 * Learn014；这里为公平对比改查真实残差，N=3 时开销可忽略）。
 */
static float residual_norm(const std::vector<float>& A,
                           const std::vector<float>& b,
                           const std::vector<float>& x) {
    const int N = static_cast<int>(b.size());
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float ax = 0.0f;
        for (int j = 0; j < N; ++j) ax += A[i * N + j] * x[j];
        float r = b[i] - ax;
        sum += r * r;
    }
    return std::sqrt(sum);
}

// ============================================================================
// Strategy —— 策略接口（算法族的唯一抽象）
// ============================================================================

/**
 * @brief 迭代解法器接口：所有"解 Ax = b 的算法"的共同契约
 *
 * 接口刻意收窄到"给题还解"：算法内部状态（残差、方向向量等）
 * 不出现在接口上 —— Context 对算法内部一无所知，才能整体替换。
 * 策略无状态，故 solve 为 const，可被多个 Context 共享。
 */
class Solver {
public:
    virtual ~Solver() = default;   // 虚析构：经基类指针删除派生对象所必需

    virtual std::string name() const = 0;   // 报表用短名
    virtual std::string brief() const = 0;  // 一句话算法特点
    virtual SolveStats solve(const std::vector<float>& A,   // 系数矩阵（行主序）
                             const std::vector<float>& b,   // 右端项
                             std::vector<float>& x,         // 入：初始猜测；出：解
                             int max_iter, float tol) const = 0;
};

// ============================================================================
// ConcreteStrategy —— 具体策略（每个算法一个类，互不引用）
// ============================================================================

/**
 * @brief 雅可比迭代（Learn012）：本轮全部用上一步的旧值
 *
 * xi^(k+1) = (b_i - Σ_{j≠i} a_ij * xj^(k)) / a_ii
 * 必须双缓冲（x 与 x_new 交替）：新值要等整轮算完才能启用。
 * 收敛慢（线性，依赖迭代矩阵谱半径），但每行独立可并行 —— GPU 友好。
 */
class JacobiSolver final : public Solver {
public:
    std::string name() const override { return "Jacobi"; }
    std::string brief() const override {
        return "all old values, double buffer, GPU-parallel";
    }

    SolveStats solve(const std::vector<float>& A, const std::vector<float>& b,
                     std::vector<float>& x, int max_iter,
                     float tol) const override {
        const int N = static_cast<int>(b.size());
        std::vector<float> x_new(N);
        SolveStats st;

        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (j != i) sum += A[i * N + j] * x[j];
                }
                x_new[i] = (b[i] - sum) / A[i * N + i];
            }
            x = x_new;   // 整轮算完才整体切换（双缓冲的同步点）

            st.iterations = iter + 1;
            st.final_residual = residual_norm(A, b, x);
            if (st.final_residual < tol) break;
        }
        return st;
    }
};

/**
 * @brief 高斯-赛德尔迭代（Learn013）：算出新值立即投入使用
 *
 * xi^(k+1) = (b_i - Σ_{j<i} a_ij * xj^(k+1)   ← 本轮新值
 *                 - Σ_{j>i} a_ij * xj^(k) ) / a_ii  ← 上轮旧值
 * 就地更新单个数组即可（无需双缓冲），收敛约比雅可比快一倍
 * （理论上 ρ_GS = ρ_Jacobi^2），但行间有依赖、天然串行。
 */
class GaussSeidelSolver final : public Solver {
public:
    std::string name() const override { return "Gauss-Seidel"; }
    std::string brief() const override {
        return "use new values at once, in-place, serial";
    }

    SolveStats solve(const std::vector<float>& A, const std::vector<float>& b,
                     std::vector<float>& x, int max_iter,
                     float tol) const override {
        const int N = static_cast<int>(b.size());
        SolveStats st;

        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (j != i) sum += A[i * N + j] * x[j];  // j<i 已是新值
                }
                x[i] = (b[i] - sum) / A[i * N + i];          // 立即写回
            }

            st.iterations = iter + 1;
            st.final_residual = residual_norm(A, b, x);
            if (st.final_residual < tol) break;
        }
        return st;
    }
};

/**
 * @brief 共轭梯度法（Learn014）：Krylov 子空间方法，要求 A 对称正定
 *
 * 残差天然两两正交，只需三项递推生成 A-正交方向：
 *   p_0 = r_0；alpha = (r.r)/(p.Ap)；x += alpha p；r -= alpha Ap；
 *   beta = (r+.r+)/(r.r)；p = r + beta p
 * 精确算术下最多 N 步收敛到精确解（本例 N=3 → 3 步即到机器精度）。
 * 每步全是 matvec + 点积 + axpy，只要求矩阵"以乘法出现"——稀疏友好。
 */
class ConjugateGradientSolver final : public Solver {
public:
    std::string name() const override { return "CG"; }
    std::string brief() const override {
        return "Krylov, SPD only, exact in <= N steps";
    }

    SolveStats solve(const std::vector<float>& A, const std::vector<float>& b,
                     std::vector<float>& x, int max_iter,
                     float tol) const override {
        const int N = static_cast<int>(b.size());
        std::vector<float> r(N), p(N), Ap(N);

        // r_0 = b - A x_0；p_0 = r_0
        for (int i = 0; i < N; ++i) {
            float ax = 0.0f;
            for (int j = 0; j < N; ++j) ax += A[i * N + j] * x[j];
            r[i] = b[i] - ax;
            p[i] = r[i];
        }
        auto dot = [](const std::vector<float>& u, const std::vector<float>& v) {
            float s = 0.0f;
            for (size_t k = 0; k < u.size(); ++k) s += u[k] * v[k];
            return s;
        };
        float rr = dot(r, r);

        SolveStats st;
        for (int iter = 0; iter < max_iter; ++iter) {
            // Ap = A p；pAp = p . Ap
            for (int i = 0; i < N; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) sum += A[i * N + j] * p[j];
                Ap[i] = sum;
            }
            float pAp = dot(p, Ap);
            if (pAp == 0.0f) break;   // 数值保护：方向退化（已到机器精度）

            float alpha = rr / pAp;   // 精确线搜索步长
            float rr_new = 0.0f;
            for (int i = 0; i < N; ++i) {
                x[i] += alpha * p[i];      // 解修正
                r[i] -= alpha * Ap[i];     // 残差递推（免重算 matvec）
                rr_new += r[i] * r[i];
            }

            st.iterations = iter + 1;
            st.final_residual = residual_norm(A, b, x);
            if (st.final_residual < tol) break;

            float beta = rr_new / rr;      // 新共轭方向
            for (int i = 0; i < N; ++i) p[i] = r[i] + beta * p[i];
            rr = rr_new;
        }
        return st;
    }
};

// ============================================================================
// Context —— 上下文：持有问题与当前策略，把求解委托给策略
// ============================================================================

/**
 * @brief 线性方程组求解器（上下文）
 *
 * 职责边界是策略模式的关键：
 *   - Context 管"事"：保存题目（A、b）、复位初始猜测、委托求解、
 *     统计误差 —— 这些逻辑与算法无关，写一遍即可；
 *   - Strategy 管"术"：怎么迭代收敛，一算法一类。
 * Context 对具体算法零依赖（只见 Solver 接口），换算法不动本类。
 */
class EquationSolver {
public:
    EquationSolver(int N, std::vector<float> A, std::vector<float> b,
                   std::shared_ptr<Solver> strategy)
        : N_(N), A_(std::move(A)), b_(std::move(b)),
          strategy_(std::move(strategy)) {}

    /// 运行时换策略（整体替换，题目与统计字段不动）
    void set_strategy(std::shared_ptr<Solver> s) { strategy_ = std::move(s); }

    /// 当前策略名（报表用；name() 按值返回，这里也必须按值接收，
    /// 返回 const& 会绑定到临时对象造成悬垂引用）
    std::string strategy_name() const { return strategy_->name(); }

    /**
     * @brief 用当前策略解一次题
     *
     * 每次都从零初始猜测出发（对比公平），委托给策略后返回统计。
     * 注意本函数与"用哪种算法"完全解耦 —— 这就是策略模式的收益。
     */
    SolveStats run(int max_iter, float tol) {
        x_.assign(N_, 0.0f);                       // 复位初始猜测
        SolveStats st = strategy_->solve(A_, b_, x_, max_iter, tol);
        return st;
    }

    /// 与精确解的最大偏差（验证用）
    float max_error_vs(const std::vector<float>& exact) const {
        float e = 0.0f;
        for (int i = 0; i < N_; ++i) {
            float d = std::fabs(x_[i] - exact[i]);
            if (d > e) e = d;
        }
        return e;
    }

private:
    int N_;                              // 维度
    std::vector<float> A_;               // 系数矩阵（行主序，N*N）
    std::vector<float> b_;               // 右端项
    std::vector<float> x_;               // 最近一次的解
    std::shared_ptr<Solver> strategy_;   // ★ 当前策略（组合替代分支）
};

// ============================================================================
// 主函数 —— 同一道题 × 三种策略：对比、运行时切换、扩展成本
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Strategy Pattern: interchangeable" << std::endl;
    std::cout << "  solvers for the same  Ax = b" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 题目：Learn014 的 3x3 对称正定方程组 ====
    // 对称 + 严格对角占优 + 正对角元 => 正定（CG 的硬性要求满足），
    // 且对角占优保证雅可比 / GS 收敛 —— 三种策略在同一题上都合法。
    //   5x +  y +  z = 7
    //    x + 6y +  z = 8        精确解 x = y = z = 1
    //    x +  y + 7z = 9
    const int N = 3;
    const int max_iter = 100;
    const float tol = 1e-6f;
    const std::vector<float> exact = { 1.0f, 1.0f, 1.0f };

    // ==== 演示 1：三策略同题对比 ====

    std::cout << std::endl;
    std::cout << "[1] Same system, three strategies (runtime selection," << std::endl;
    std::cout << "    context code never changes):" << std::endl;
    std::cout << "    5x +  y +  z = 7" << std::endl;
    std::cout << "     x + 6y +  z = 8    exact solution: (1, 1, 1)" << std::endl;
    std::cout << "     x +  y + 7z = 9" << std::endl;

    // 注意：不能写成 { make_shared<A>, make_shared<B>, ... } 初始化列表
    // —— 各元素模板类型不同，无法推导共同类型（Learn015 踩过的坑），
    // 必须逐个 push_back 到显式类型的容器。
    std::vector<std::shared_ptr<Solver>> strategies;
    strategies.push_back(std::make_shared<JacobiSolver>());
    strategies.push_back(std::make_shared<GaussSeidelSolver>());
    strategies.push_back(std::make_shared<ConjugateGradientSolver>());

    // Context 持题 + 初始策略；之后只换策略，不换 Context
    EquationSolver ctx(N,
                       { 5.0f, 1.0f, 1.0f,    // A 第 0 行
                         1.0f, 6.0f, 1.0f,    // A 第 1 行
                         1.0f, 1.0f, 7.0f },  // A 第 2 行
                       { 7.0f, 8.0f, 9.0f },  // b
                       strategies[0]);

    // 表头与数据行共用同一格式串，保证列对齐
    printf("\n    %-13s | %5s | %10s | %10s\n", "Strategy", "iters",
           "residual", "max error");
    printf("    %s\n", "---------------+-------+------------+------------");
    for (const auto& s : strategies) {
        ctx.set_strategy(s);                       // ★ 运行时换策略
        SolveStats st = ctx.run(max_iter, tol);    // 同一段委托代码
        printf("    %-13s | %5d | %10.2e | %10.2e\n",
               s->name().c_str(), st.iterations, st.final_residual,
               ctx.max_error_vs(exact));
    }
    std::cout << "    (tolerance: ||b - Ax||_2 < " << tol << ")" << std::endl;

    // ==== 演示 2：各策略特点（算法知识留在策略类里）====

    std::cout << std::endl;
    std::cout << "[2] Each strategy carries its own know-how:" << std::endl;
    for (const auto& s : strategies) {
        std::cout << "    " << s->name() << ": " << s->brief() << std::endl;
    }

    // ==== 演示 3：换策略前后 —— 同一个 Context 对象 ====

    std::cout << std::endl;
    std::cout << "[3] Swap the algorithm on a live context object:" << std::endl;
    ctx.set_strategy(strategies[0]);
    SolveStats a = ctx.run(max_iter, tol);
    std::cout << "    strategy = " << ctx.strategy_name() << " -> "
              << a.iterations << " iterations" << std::endl;
    ctx.set_strategy(strategies[2]);   // 平替：Context、题目、统计全不动
    SolveStats b = ctx.run(max_iter, tol);
    std::cout << "    strategy = " << ctx.strategy_name() << " -> "
              << b.iterations << " iterations" << std::endl;

    // ==== 演示 4：扩展成本对比（开闭原则）====

    std::cout << std::endl;
    std::cout << "[4] Cost of adding a 4th solver (e.g. SOR, PCG):" << std::endl;
    std::cout << "    switch version : edit the existing solve() body" << std::endl;
    std::cout << "                     (modify old, tested code)" << std::endl;
    std::cout << "    strategy version: write one new class, register it" << std::endl;
    std::cout << "                     (add-only; context untouched)" << std::endl;

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. An algorithm becomes an object; the context" << std::endl;
    std::cout << "     only sees the Strategy interface." << std::endl;
    std::cout << "  2. New algorithm = one new class (open-closed)," << std::endl;
    std::cout << "     no switch/if-else to maintain." << std::endl;
    std::cout << "  3. Strategies are swappable at runtime, like the" << std::endl;
    std::cout << "     bridge swap in Learn015." << std::endl;
    std::cout << "  4. Rule of thumb (Learn015): hierarchies on both" << std::endl;
    std::cout << "     sides -> Bridge; one varying side -> Strategy." << std::endl;

    return 0;
}
