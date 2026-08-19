/**
 * @file    main.cpp
 * @brief   模板方法模式（Template Method）—— 骨架固定在基类，步骤延迟到子类
 *
 * 定义（GoF 行为型模式）：
 *   定义一个操作中算法的骨架（skeleton），而将一些步骤延迟到子类中。
 *   模板方法使子类可以在不改变算法结构的前提下，重定义该算法的某些步骤。
 *
 * 本例场景（与 Learn016 同一道题、同一组解法，换一种分解方式）：
 *   雅可比 / 高斯-赛德尔 / CG 的主循环骨架其实完全同构：
 *
 *       初始化工作区 → 循环{ 做一步迭代 → 查收敛 } → 汇报统计
 *
 *   三种解法各自为战时，这段骨架会在每个实现里抄一遍 —— 将来改判据
 *   （比如换成相对残差）要同时改三处。模板方法把骨架提升到基类并
 *   固定下来（非虚的 run()），只把"做一步迭代"这个真正不同的步骤
 *   留成纯虚函数，由子类填写：
 *
 *        AbstractClass                      ConcreteClass
 *        ─────────────                      ─────────────
 *   ┌───────────────────────────┐     ┌────────────────────────────┐
 *   │ IterativeSolver           │     │ JacobiSolver               │
 *   │  A_ b_ x_ stats_          │     │   initialize() 分配双缓冲  │
 *   │                           │◄────┤   iterate()    全用旧值    │
 *   │ + run()   ← 模板方法(非虚)│ 继承 │ GaussSeidelSolver          │
 *   │ # initialize() = 0  原语  │     │   iterate()    就地更新    │
 *   │ # iterate()      = 0 原语 │     │ ConjugateGradientSolver   │
 *   │ # on_iteration() {}  钩子 │     │   iterate()    三项递推    │
 *   └───────────────────────────┘     └────────────────────────────┘
 *
 *   run() 的骨架（子类不可改写）：
 *     initialize();                     // 步骤 1：准备工作区（纯虚）
 *     for (iter = 0..max_iter) {
 *         iterate();                    // 步骤 2：一步迭代（纯虚）
 *         stats = residual_norm(...);   // 步骤 3：公共判据（公共代码）
 *         on_iteration(iter);           // 步骤 4：钩子（默认空，可选）
 *         if (收敛 || stop_requested_)  // 步骤 5：公共终止条件
 *             break;
 *     }
 *
 * 好莱坞原则（Hollywood Principle）：
 *   "别调用我们，我们会调用你"（Don't call us, we'll call you）。
 *   子类不驱动流程，只被基类骨架回调 —— 控制权在框架手里。
 *   这正是框架（framework）与库（library）的本质区别：库是你调它，
 *   框架是它调你。模板方法是这个原则最小的化身。
 *
 * 角色与三种"方法"（GoF）：
 *   AbstractClass  —— IterativeSolver：骨架 + 公共状态 + 公共判据
 *   ConcreteClass  —— Jacobi / GaussSeidel / ConjugateGradientSolver
 *   - 模板方法（template method）：run()，非虚 —— 骨架不变式
 *   - 原语操作（primitive op）    ：initialize() / iterate()，纯虚
 *   - 钩子方法（hook method）     ：on_iteration()，默认实现，可选覆盖
 *
 * 与策略模式（Learn016）的对比 —— 同一道题的两种分解：
 *   - Strategy：组合（has-a），替换"整个算法"，运行时 set_strategy 可换；
 *   - Template Method：继承（is-a），固定"骨架"只变"步骤"，子类类型
 *     构造时定型、运行中不可换；换来的是零配置 + 骨架不变式被强制。
 *   经验法则：整个算法可换 → Strategy；骨架相同、个别步骤不同 →
 *   Template Method。二者可叠加：一个策略对象内部可以用模板方法搭骨架。
 *   （Learn015 的口诀链：两边都有层次 → Bridge；一边整个可换 →
 *    Strategy；一边只是步骤不同 → Template Method。）
 *
 * 现代 C++ 关联：
 *   - NVI（Non-Virtual Interface）惯用法：公有接口非虚、虚函数收进
 *     protected/private —— 本例 run() 公有非虚 + 步骤函数受保护虚，
 *     正是 NVI 的标准应用（Herb Sutter: "Prefer to make virtual
 *     functions private" 的精神）；
 *   - CRTP（奇异递归模板）是它的编译期版本：零虚函数开销，
 *     但同样失去运行时多态（见 readme 案例）。
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <cmath>      // sqrt, fabs
#include <cstdio>     // printf（表格式输出）
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>  // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

// ============================================================================
// 公共例程 —— 与具体算法无关的统计工具
// ============================================================================

/// 一次求解的统计信息（与算法无关的"统一报表"）
struct SolveStats {
    int iterations = 0;           // 实际迭代次数
    float final_residual = 0.0f;  // 收敛时的真实残差 ||b - A x||_2
};

/// 真实残差 ||b - A x||_2 —— 骨架第 3 步的公共判据（Learn016 同款）
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

/// 点积（CG 策略内部递推用）
static float dot(const std::vector<float>& u, const std::vector<float>& v) {
    float s = 0.0f;
    for (size_t k = 0; k < u.size(); ++k) s += u[k] * v[k];
    return s;
}

/// 与精确解的最大偏差（验证用）
static float max_error_vs(const std::vector<float>& x,
                          const std::vector<float>& exact) {
    float e = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        float d = std::fabs(x[i] - exact[i]);
        if (d > e) e = d;
    }
    return e;
}

// ============================================================================
// AbstractClass —— 抽象类：算法骨架 + 公共状态 + 公共判据
// ============================================================================

/**
 * @brief 迭代解法器基类：把"怎么解"的同构骨架固定下来
 *
 * 骨架（run）只写一遍；判据、统计、终止逻辑全在此维护 ——
 * 将来把判据换成相对残差，只改本类一处，三个子类自动受益。
 * 子类唯一要做的事：告诉骨架"一步迭代"怎么走。
 */
class IterativeSolver {
public:
    IterativeSolver(int N, std::vector<float> A, std::vector<float> b,
                    std::string name)
        : N_(N), A_(std::move(A)), b_(std::move(b)),
          x_(static_cast<size_t>(N), 0.0f), name_(std::move(name)) {}

    virtual ~IterativeSolver() = default;   // 虚析构：经基类指针删除派生对象

    const std::string& name() const { return name_; }        // 报表用
    const std::vector<float>& solution() const { return x_; } // 最近一次的解

    /**
     * @brief 模板方法：算法骨架 —— 刻意非虚！
     *
     * 若声明为 virtual，子类就能改写流程本身，"初始化后再迭代、
     * 每步查收敛"的不变式随之失守。骨架必须由基类独裁。
     * 这同时是 NVI 惯用法：公有接口非虚，虚步骤收在 protected 里。
     */
    SolveStats run(int max_iter, float tol) {
        initialize();                       // 步骤 1：准备工作区（原语）

        stats_ = SolveStats{};
        stop_requested_ = false;

        for (int iter = 0; iter < max_iter; ++iter) {
            iterate();                      // 步骤 2：一步迭代（原语）

            stats_.iterations = iter + 1;
            stats_.final_residual = residual_norm(A_, b_, x_);  // 步骤 3：公共判据

            on_iteration(iter);             // 步骤 4：钩子（默认空）

            if (stats_.final_residual < tol || stop_requested_) {
                break;                      // 步骤 5：公共终止条件
            }
        }
        return stats_;
    }

protected:
    // ---- 原语操作：protected —— 外部不得乱序调用，只有骨架能调 ----

    /// 原语 1：初始化各自的额外工作区（x_ 已由基类清零）
    virtual void initialize() = 0;

    /// 原语 2：从当前 x_ 出发推进一步（骨架的核心变化点）
    virtual void iterate() = 0;

    // ---- 钩子方法：带默认实现，子类"可选"覆盖（不覆盖也不出错）----
    // 区别于纯虚原语：钩子提供骨架的"插入点"，而非必填步骤

    /// 钩子：每步迭代完成后的回调（默认什么都不做）
    virtual void on_iteration(int /*iter*/) {}

    // ---- 公共状态：子类经 protected 直接访问（继承带来的亲近）----
    int N_;                          // 维度
    std::vector<float> A_;           // 系数矩阵（行主序，N*N）
    std::vector<float> b_;           // 右端项
    std::vector<float> x_;           // 解向量（骨架第 3 步统一检查它）
    SolveStats stats_;               // 骨架统一维护的统计
    bool stop_requested_ = false;    // 原语"请假"通道：请求骨架终止
                                      //（循环控制权在骨架，步骤无权 break）

private:
    std::string name_;
};

// ============================================================================
// ConcreteClass —— 具体类：只填写骨架留给自己的步骤
// ============================================================================

/**
 * @brief 雅可比迭代（Learn012）：本轮全部用上一步的旧值
 *
 * 全类只有两个函数：initialize（分配双缓冲）+ iterate（一步扫描）。
 * 循环、判据、统计全部继承自骨架 —— 这就是模板方法的收益。
 */
class JacobiSolver final : public IterativeSolver {
public:
    JacobiSolver(int N, std::vector<float> A, std::vector<float> b)
        : IterativeSolver(N, std::move(A), std::move(b), "Jacobi") {}

protected:
    void initialize() override {
        x_new_.assign(static_cast<size_t>(N_), 0.0f);   // 第二缓冲区
    }

    void iterate() override {
        for (int i = 0; i < N_; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N_; ++j) {
                if (j != i) sum += A_[i * N_ + j] * x_[j];
            }
            x_new_[i] = (b_[i] - sum) / A_[i * N_ + i];
        }
        x_.swap(x_new_);   // 双缓冲切换：整轮算完才整体启用新值
    }

private:
    std::vector<float> x_new_;
};

/**
 * @brief 高斯-赛德尔迭代（Learn013）：算出新值立即投入使用
 *
 * 工作区不需要任何额外成员（就地更新），initialize 为空实现 ——
 * 钩子式的宽容在这里体现：不是每个子类都必须做点什么。
 */
class GaussSeidelSolver : public IterativeSolver {
public:
    GaussSeidelSolver(int N, std::vector<float> A, std::vector<float> b)
        : IterativeSolver(N, std::move(A), std::move(b), "Gauss-Seidel") {}

protected:
    void initialize() override { /* 就地更新：无需额外工作区 */ }

    void iterate() override {
        for (int i = 0; i < N_; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < N_; ++j) {
                if (j != i) sum += A_[i * N_ + j] * x_[j];  // j<i 已是新值
            }
            x_[i] = (b_[i] - sum) / A_[i * N_ + i];         // 立即写回
        }
    }
};

/**
 * @brief 共轭梯度法（Learn014）：Krylov 子空间方法，要求 A 对称正定
 *
 * iterate() = 完整的一步 CG 三项递推（Ap → alpha → x/r 修正 → beta → p）。
 * 数值细节：pAp == 0（方向退化到机器精度）时无法继续，但步骤无权
 * break —— 经 stop_requested_ 请求骨架终止（见骨架第 5 步），
 * 这是"控制权在框架"的一个真实工程体现。
 */
class ConjugateGradientSolver final : public IterativeSolver {
public:
    ConjugateGradientSolver(int N, std::vector<float> A, std::vector<float> b)
        : IterativeSolver(N, std::move(A), std::move(b), "CG") {}

protected:
    void initialize() override {
        // x_0 = 0（基类已清零）=> r_0 = b - A*0 = b；p_0 = r_0
        r_ = b_;
        p_ = r_;
        Ap_.assign(static_cast<size_t>(N_), 0.0f);
        rr_ = dot(r_, r_);
    }

    void iterate() override {
        for (int i = 0; i < N_; ++i) {          // Ap = A p
            float sum = 0.0f;
            for (int j = 0; j < N_; ++j) sum += A_[i * N_ + j] * p_[j];
            Ap_[i] = sum;
        }
        float pAp = dot(p_, Ap_);
        if (pAp == 0.0f) {                      // 机器精度内方向退化
            stop_requested_ = true;             // 向骨架"请假"终止
            return;
        }

        float alpha = rr_ / pAp;                // 精确线搜索步长
        float rr_new = 0.0f;
        for (int i = 0; i < N_; ++i) {
            x_[i] += alpha * p_[i];             // 解修正
            r_[i] -= alpha * Ap_[i];            // 残差递推
            rr_new += r_[i] * r_[i];
        }

        float beta = rr_new / rr_;              // 新共轭方向
        for (int i = 0; i < N_; ++i) p_[i] = r_[i] + beta * p_[i];
        rr_ = rr_new;
    }

private:
    std::vector<float> r_, p_, Ap_;             // CG 专属工作区
    float rr_ = 0.0f;                           // r . r
};

// ============================================================================
// 钩子方法演示 —— 不动骨架、不加新算法，只挂一个"观察哨"
// ============================================================================

/// 追踪版 GS：唯一目的是覆盖 on_iteration 钩子，打印前几步的残差轨迹
class TracedGaussSeidel final : public GaussSeidelSolver {
public:
    using GaussSeidelSolver::GaussSeidelSolver;   // 继承构造函数

protected:
    /// 钩子默认是空实现；这里插入"观察"逻辑 —— 骨架一行未改
    void on_iteration(int iter) override {
        if (iter < TRACE_ITERS) {
            printf("    iter %2d: ||b - Ax||_2 = %.6e\n",
                   iter + 1, stats_.final_residual);
        }
    }

private:
    static constexpr int TRACE_ITERS = 4;   // 最多打印的前几步
};

// ============================================================================
// 主函数 —— 骨架演示、三实现对比、钩子演示、与策略模式对照
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Template Method: fixed skeleton in the" << std::endl;
    std::cout << "  base class, steps filled by subclasses" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 题目：Learn014/016 的 3x3 对称正定方程组，精确解 (1,1,1) ====
    const int N = 3;
    const int max_iter = 100;
    const float tol = 1e-6f;
    const std::vector<float> A = { 5.0f, 1.0f, 1.0f,    // 第 0 行
                                   1.0f, 6.0f, 1.0f,    // 第 1 行
                                   1.0f, 1.0f, 7.0f };  // 第 2 行
    const std::vector<float> b = { 7.0f, 8.0f, 9.0f };
    const std::vector<float> exact = { 1.0f, 1.0f, 1.0f };

    // ==== 演示 1：骨架 —— 只在基类写一遍 ====

    std::cout << std::endl;
    std::cout << "[1] The skeleton, written once in the base class:" << std::endl;
    std::cout << "    initialize()                 <- primitive (pure virtual)" << std::endl;
    std::cout << "    loop {" << std::endl;
    std::cout << "        iterate()                <- primitive (pure virtual)" << std::endl;
    std::cout << "        residual = ||b - Ax||_2  <- common criterion (base)" << std::endl;
    std::cout << "        on_iteration(iter)       <- hook (default: no-op)" << std::endl;
    std::cout << "        break if converged       <- common control (base)" << std::endl;
    std::cout << "    }" << std::endl;

    // ==== 演示 2：三个具体类 —— 各自只实现 initialize + iterate ====

    std::cout << std::endl;
    std::cout << "[2] Three concrete classes fill in the steps" << std::endl;
    std::cout << "    (loop / criterion / stats inherited):" << std::endl;
    std::cout << "    5x +  y +  z = 7" << std::endl;
    std::cout << "     x + 6y +  z = 8    exact solution: (1, 1, 1)" << std::endl;
    std::cout << "     x +  y + 7z = 9" << std::endl;

    // 注意：不能写成 { make_unique<A>(...), ... } 初始化列表 ——
    // 各元素模板类型不同无法推导共同类型（Learn015 踩过的坑），
    // 必须逐个 push_back 到显式类型的容器
    std::vector<std::unique_ptr<IterativeSolver>> solvers;
    solvers.push_back(std::make_unique<JacobiSolver>(N, A, b));
    solvers.push_back(std::make_unique<GaussSeidelSolver>(N, A, b));
    solvers.push_back(std::make_unique<ConjugateGradientSolver>(N, A, b));

    // 表头与数据行共用同一格式串，保证列对齐（与 Learn016 同格式，便于对照）
    printf("\n    %-13s | %5s | %10s | %10s\n", "Concrete", "iters",
           "residual", "max error");
    printf("    %s\n", "---------------+-------+------------+------------");
    for (const auto& s : solvers) {
        SolveStats st = s->run(max_iter, tol);   // 调的都是基类的同一个 run
        printf("    %-13s | %5d | %10.2e | %10.2e\n",
               s->name().c_str(), st.iterations, st.final_residual,
               max_error_vs(s->solution(), exact));
    }
    std::cout << "    (tolerance: ||b - Ax||_2 < " << tol << ")" << std::endl;

    // ==== 演示 3：钩子方法 —— 不改骨架，插入观察逻辑 ====

    std::cout << std::endl;
    std::cout << "[3] Hook method: TracedGaussSeidel overrides" << std::endl;
    std::cout << "    on_iteration() only (skeleton untouched):" << std::endl;

    TracedGaussSeidel traced(N, A, b);
    SolveStats st = traced.run(max_iter, tol);
    std::cout << "    ... converged in " << st.iterations << " iterations" << std::endl;

    // ==== 演示 4：与策略模式（Learn016）对照 ====

    std::cout << std::endl;
    std::cout << "[4] Template Method (this) vs Strategy (Learn016):" << std::endl;
    std::cout << "    relation     : inheritance (is-a)   | composition (has-a)" << std::endl;
    std::cout << "    what varies  : steps of a fixed     | the whole algorithm" << std::endl;
    std::cout << "                  skeleton" << std::endl;
    std::cout << "    runtime swap : no (type fixed at    | yes (set_strategy)" << std::endl;
    std::cout << "                  construction)" << std::endl;
    std::cout << "    invariant    : base enforces the    | context knows nothing" << std::endl;
    std::cout << "                  skeleton" << std::endl;
    std::cout << "    -> same problem, same numbers, two decompositions;" << std::endl;
    std::cout << "       they also compose: a strategy can be built on a" << std::endl;
    std::cout << "       template-method skeleton internally." << std::endl;

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. run() is NON-virtual: the skeleton is an" << std::endl;
    std::cout << "     invariant subclasses cannot change." << std::endl;
    std::cout << "  2. Primitives (initialize/iterate) are the" << std::endl;
    std::cout << "     only things a concrete class writes." << std::endl;
    std::cout << "  3. Hooks (on_iteration) offer optional insertion" << std::endl;
    std::cout << "     points with default no-op bodies." << std::endl;
    std::cout << "  4. Hollywood principle: the base calls the derived," << std::endl;
    std::cout << "     never the reverse." << std::endl;
    std::cout << "  5. Series rule of thumb (Learn015/016): two-sided" << std::endl;
    std::cout << "     hierarchies -> Bridge; whole algorithm swaps ->" << std::endl;
    std::cout << "     Strategy; only steps differ -> Template Method." << std::endl;

    return 0;
}
