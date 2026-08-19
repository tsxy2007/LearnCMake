/**
 * @file    main.cpp
 * @brief   桥接模式（Bridge Pattern）—— 将"抽象"与"实现"分离，使二者独立演化
 *
 * 问题背景（继承的类爆炸）：
 *   图形程序里有两个**独立变化**的维度：
 *     维度 A —— 画什么：Circle / Rectangle / Triangle / ...
 *     维度 B —— 怎么画：OpenGL / Vulkan / CPU 软件光栅化 / ...
 *   若用继承把两个维度编织在一起，每个组合都得是一个类：
 *        class Circle_OpenGL    : public Circle,    public OpenGLAPI { ... };
 *        class Circle_Vulkan    : public Circle,    public VulkanAPI { ... };
 *        class Rectangle_OpenGL : public Rectangle, public OpenGLAPI { ... };
 *        ...
 *     M 个形状 × N 个后端 = M*N 个具体类（3×3 = 9，5×5 = 25），
 *     且后端调用代码在每个组合里重复一遍；新增一个形状要再写 N 个类，
 *     新增一个后端要再写 M 个类 —— 复杂度按**乘法**增长。
 *
 * 桥接模式的做法：
 *   识别出两个变化的维度后，让"抽象"维度（Abstraction，本例的 Shape）
 *   不再通过继承、而是通过一个指针持有"实现"维度（Implementor，
 *   本例的 Renderer）—— 这根指针就是"桥"：
 *
 *        Abstraction                    Implementor
 *        ───────────                    ───────────
 *        Shape ◇───────────────>        Renderer
 *        ├── Circle                     ├── OpenGLRenderer
 *        └── Rect                       ├── VulkanRenderer
 *                                       └── SoftwareRenderer
 *
 *   - Shape 只描述"高层语义"（draw / resize / 名称），把任何实际的
 *     绘制原语**委托**给 Renderer；
 *   - Renderer 只描述"底层原语"（画圆 / 画矩形），根本不知道 Shape
 *     的存在，更不知道有几个形状；
 *   - 两棵继承树各自独立生长：新增 Triangle 只加 1 个类，新增
 *     DirectXRenderer 后端也只加 1 个类 —— 复杂度从 M*N 降为 M+N。
 *
 * 四个角色（GoF 结构型模式）：
 *   Abstraction         —— Shape：高层接口，持有 Implementor（桥）
 *   RefinedAbstraction  —— Circle / Rect：扩充 Abstraction 的层次
 *   Implementor         —— Renderer：底层原语接口，与 Abstraction 解耦
 *   ConcreteImplementor —— OpenGL/Vulkan/SoftwareRenderer：对接具体 API
 *
 * 与策略模式的区别（代码形状很像，意图不同）：
 *   - Bridge 是结构型：两个维度都长期演化、各自拥有继承层次，
 *     Implementor 通常代表"平台/机制"（渲染后端、消息通道、ORM）；
 *   - Strategy 是行为型：封装一族可替换的算法，通常只有一个维度，
 *     Context 不会为策略建继承树。
 *   判断口诀：两边都有类层次 → Bridge；只有一边有 → Strategy。
 *
 * 现代 C++ 要点：
 *   - 桥用 std::shared_ptr<Renderer>：渲染后端如同 GPU 上下文，
 *     天然被多个 Shape 共享（想表达独占可换 unique_ptr，桥照样成立）
 *   - 接口类必须给出虚析构（经由基类指针删除派生对象的前提）
 *   - "构造时注入 + set_renderer 运行时替换"即依赖注入的两种形态
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <cstdio>     // printf（表格式输出）
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>   // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

// ============================================================================
// Implementor —— 实现者接口（维度 B："怎么画"）
// ============================================================================

/**
 * @brief 渲染后端接口：只提供"原语"级操作
 *
 * 它刻意做得很"低级"：只有画圆 / 画矩形这类原子能力，绝不出现
 * draw_shape 之类的高层语义。高低层分离正是 Bridge 的关键：
 *   - Abstraction 站在"语义层"，负责编排这些原语；
 *   - Implementor 站在"机制层"，各自对接真实的图形 API。
 * 后端换代（OpenGL → Vulkan）时，Shape 那棵树一行都不用改。
 */
class Renderer {
public:
    virtual ~Renderer() = default;   // 虚析构：经基类指针删除派生对象所必需

    virtual void draw_circle(float cx, float cy, float radius) = 0;
    virtual void draw_rect(float x, float y, float w, float h) = 0;
    virtual std::string backend_name() const = 0;
};

// ============================================================================
// ConcreteImplementor —— 具体实现者（各对接一种真实 API）
// ============================================================================

/// OpenGL 后端：典型的"绑缓冲 + 发绘制命令"风格
class OpenGLRenderer final : public Renderer {
public:
    void draw_circle(float cx, float cy, float radius) override {
        // 真实实现：圆细分（tessellate）成 24 边形上传顶点后绘制；
        // 此处仅打印调用轨迹，演示"桥通到了哪家电厂的插座"
        std::cout << "    [OpenGL  ] tessellate -> 24-gon, glDrawArrays("
                  << "GL_TRIANGLE_FAN)  circle@(" << cx << ", " << cy
                  << ") r=" << radius << "\n";
    }

    void draw_rect(float x, float y, float w, float h) override {
        std::cout << "    [OpenGL  ] glDrawArrays(GL_TRIANGLES)  rect@("
                  << x << ", " << y << ") " << w << "x" << h << "\n";
    }

    std::string backend_name() const override { return "OpenGL"; }
};

/// Vulkan 后端：命令缓冲（command buffer）风格
class VulkanRenderer final : public Renderer {
public:
    void draw_circle(float cx, float cy, float radius) override {
        std::cout << "    [Vulkan  ] vkCmdDraw(circle@(" << cx << ", " << cy
                  << ") r=" << radius << ", 24 verts, 1 instance)\n";
    }

    void draw_rect(float x, float y, float w, float h) override {
        std::cout << "    [Vulkan  ] vkCmdDraw(rect@(" << x << ", " << y
                  << ") " << w << "x" << h << ", 6 verts, 1 instance)\n";
    }

    std::string backend_name() const override { return "Vulkan"; }
};

/// CPU 软件光栅化后端：逐像素写入帧缓冲风格
class SoftwareRenderer final : public Renderer {
public:
    void draw_circle(float cx, float cy, float radius) override {
        std::cout << "    [Software] scanline rasterize circle@(" << cx
                  << ", " << cy << ") r=" << radius
                  << " -> 512x512 framebuffer\n";
    }

    void draw_rect(float x, float y, float w, float h) override {
        std::cout << "    [Software] fill rect@(" << x << ", " << y << ") "
                  << w << "x" << h << " -> 512x512 framebuffer\n";
    }

    std::string backend_name() const override { return "Software"; }
};

// ============================================================================
// Abstraction —— 抽象接口（维度 A："画什么"）
// ============================================================================

/**
 * @brief 形状基类：持有 Renderer 的指针 —— 这根指针就是"桥"
 *
 * Shape 层只关心形状自身的几何语义（位置、尺寸、缩放），任何实际
 * 绘制都委托给 renderer_。它对 OpenGL / Vulkan / 软件光栅化一无所知，
 * 后端增删不需要改动 Shape 这棵树的任何一行 —— 两个维度解耦了。
 */
class Shape {
public:
    /// 构造时"架桥"：注入具体后端（构造器注入，依赖注入的一种形态）
    explicit Shape(std::shared_ptr<Renderer> renderer)
        : renderer_(std::move(renderer)) {}

    virtual ~Shape() = default;

    virtual void draw() const = 0;           // 高层接口：语义层的"画自己"
    virtual void resize(float factor) = 0;   // 高层接口：纯几何语义，与后端无关
    virtual std::string name() const = 0;    // 高层接口：自述身份

    /// 运行时"换桥"：整体替换实现，Shape 对象本身的几何状态不动
    void set_renderer(std::shared_ptr<Renderer> r) { renderer_ = std::move(r); }

    /// 当前桥接的后端名（演示观察用）
    std::string backend_name() const { return renderer_->backend_name(); }

protected:
    /// 子类经此把原语操作委托过桥
    Renderer& impl() const { return *renderer_; }

private:
    std::shared_ptr<Renderer> renderer_;     // ★ 桥：组合替代继承
};

// ============================================================================
// RefinedAbstraction —— 扩充抽象（只编排原语，不碰任何具体后端）
// ============================================================================

/**
 * @brief 圆：只依赖"画圆原语"，不关心背后是哪家 API
 */
class Circle final : public Shape {
public:
    Circle(std::shared_ptr<Renderer> renderer,
           float cx, float cy, float radius)
        : Shape(std::move(renderer)), cx_(cx), cy_(cy), radius_(radius) {}

    void draw() const override {
        // 形状更复杂时，这里可以是多步原语的编排（如描边 + 填充 + 阴影），
        // 编排逻辑属于"语义"，放在抽象层；每步原语则过桥交给后端
        impl().draw_circle(cx_, cy_, radius_);
    }

    /// 缩放是纯几何操作：只改本层状态，后端对 resize 一无所知
    void resize(float factor) override { radius_ *= factor; }

    std::string name() const override { return "Circle"; }

private:
    float cx_, cy_, radius_;
};

/**
 * @brief 矩形：同样的套路 —— 编排 draw_rect 原语实现自己的语义
 *
 * 命名注意：类名用 Rect 而非 Rectangle —— windows.h（wingdi.h）在全局
 * 命名空间声明了 GDI 绘图函数 BOOL Rectangle(HDC,...)，同名非类型标识符
 * 会隐藏类名，使 make_unique<Rectangle> 解析失败（与 min/max 宏同类的
 * Windows SDK 名字污染问题）。
 */
class Rect final : public Shape {
public:
    Rect(std::shared_ptr<Renderer> renderer,
         float x, float y, float w, float h)
        : Shape(std::move(renderer)), x_(x), y_(y), w_(w), h_(h) {}

    void draw() const override { impl().draw_rect(x_, y_, w_, h_); }
    void resize(float factor) override { w_ *= factor; h_ *= factor; }
    std::string name() const override { return "Rectangle"; }

private:
    float x_, y_, w_, h_;
};

// ============================================================================
// 主函数 —— 三个演示：维度的乘积、运行时换桥、语义留在抽象层
// ============================================================================

/// 依次画出一组形状（演示"一批形状共享同一座桥"）
static void draw_all(const std::vector<std::unique_ptr<Shape>>& shapes) {
    for (const auto& s : shapes) {
        std::cout << "  " << s->name() << " @ " << s->backend_name() << ":";
        std::cout << "\n";
        s->draw();
    }
}

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Bridge Pattern: Shape (abstraction)" << std::endl;
    std::cout << "               x Renderer (implementor)" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 演示 1：类爆炸对比 —— 为什么需要桥接 ====
    // 继承方案把两个维度编死在一起，组合数按乘法增长；
    // 桥接方案两棵树独立生长，总数按加法增长。
    std::cout << std::endl;
    std::cout << "[1] Class count: inheritance (M*N) vs bridge (M+N)" << std::endl;
    // 表头与数据行用同一套 printf 格式串，保证列对齐
    printf("    %2s shapes x %2s backends | %11s | %6s\n", "M", "N",
           "inheritance", "bridge");
    printf("    %s\n", "------------------------+-------------+-------");
    const int dims[] = { 2, 3, 4, 5 };
    for (int m : dims) {
        printf("    %2d shapes x %2d backends | %11d | %6d\n", m, m, m * m, m + m);
    }
    std::cout << "    -> add 1 shape:  inheritance += N, bridge += 1" << std::endl;
    std::cout << "    -> add 1 backend: inheritance += M, bridge += 1" << std::endl;

    // ==== 演示 2：2 个形状 x 3 个后端 —— 6 种组合，零个"组合专用类" ====

    // 三个后端实例（如 GPU 上下文，可被多个形状共享，故用 shared_ptr）
    auto opengl = std::make_shared<OpenGLRenderer>();
    auto vulkan = std::make_shared<VulkanRenderer>();
    auto software = std::make_shared<SoftwareRenderer>();

    std::cout << std::endl;
    std::cout << "[2] Cross product: 2 shapes x 3 backends," << std::endl;
    std::cout << "    no class written per combination:" << std::endl;

    // 对每个后端：架桥 -> 画全部形状。组合是"配出来的"，不是"写出来的"。
    // 注：三个 shared_ptr<具体后端> 模板类型各异，必须先统一收进
    // shared_ptr<Renderer> 数组再遍历（花括号初始化列表无法自行推导共同类型）
    const std::shared_ptr<Renderer> backends[] = { opengl, vulkan, software };
    for (const auto& backend : backends) {
        std::cout << "  --- bridging all shapes to "
                  << backend->backend_name() << " ---" << std::endl;

        std::vector<std::unique_ptr<Shape>> shapes;
        shapes.push_back(std::make_unique<Circle>(backend, 10.0f, 20.0f, 5.0f));
        shapes.push_back(std::make_unique<Rect>(backend, 1.0f, 2.0f, 8.0f, 4.0f));

        draw_all(shapes);
    }

    // ==== 演示 3：运行时换桥 —— 同一个对象，整体换掉实现 ====

    std::cout << std::endl;
    std::cout << "[3] Re-bridge at runtime (set_renderer):" << std::endl;

    // 只有一个 Circle 对象，几何状态（圆心、半径）始终活在抽象层
    Circle circle(opengl, 10.0f, 20.0f, 5.0f);

    std::cout << "  bridge = " << circle.backend_name() << ":" << std::endl;
    circle.draw();

    circle.set_renderer(vulkan);   // 换桥：实现整体替换，对象不重建
    std::cout << "  bridge = " << circle.backend_name() << ":" << std::endl;
    circle.draw();

    circle.set_renderer(software);
    std::cout << "  bridge = " << circle.backend_name() << ":" << std::endl;
    circle.draw();

    // ==== 演示 4：resize 语义留在抽象层，后端无感 ====

    std::cout << std::endl;
    std::cout << "[4] resize(1.5) is pure geometry in the abstraction layer:"
              << std::endl;
    circle.resize(1.5f);           // 只改 radius_，桥两端的接口都没变
    std::cout << "  after resize, same bridge (" << circle.backend_name()
              << "):" << std::endl;
    circle.draw();

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. Two independent dimensions -> two class trees," << std::endl;
    std::cout << "     connected by a pointer (the bridge)." << std::endl;
    std::cout << "  2. Complexity grows M+N instead of M*N;" << std::endl;
    std::cout << "     each new shape/backend costs exactly 1 class." << std::endl;
    std::cout << "  3. The bridge can be swapped at runtime." << std::endl;
    std::cout << "  4. Rule of thumb: hierarchies on both sides -> Bridge;" << std::endl;
    std::cout << "     replaceable algorithm on one side -> Strategy." << std::endl;

    return 0;
}
