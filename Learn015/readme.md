# Learn015 — 桥接模式（Bridge Pattern）：把"抽象"与"实现"分离

GoF 23 种设计模式中的**结构型**模式。一句话：

> 当一个类型存在**两个独立变化的维度**时，把其中一个维度从"继承"
> 改为"组合"（一根指针连过去，即"桥"），让两棵类树各自独立演化，
> 复杂度从 **M×N** 降为 **M+N**。

本例场景：**形状（画什么）× 渲染后端（怎么画）**。

---

## 1. 问题：继承的类爆炸

图形程序里有两个维度各自膨胀：

- 维度 A —— 画什么：`Circle` / `Rectangle` / `Triangle` / …
- 维度 B —— 怎么画：OpenGL / Vulkan / CPU 软件光栅化 / …

如果用多重继承把两个维度编织在一起：

```cpp
class Circle_OpenGL    : public Circle,    public OpenGLAPI { /* OpenGL 调用 */ };
class Circle_Vulkan    : public Circle,    public VulkanAPI { /* Vulkan 调用 */ };
class Circle_Software  : public Circle,    public SoftAPI    { /* 光栅化代码 */ };
class Rect_OpenGL      : public Rectangle, public OpenGLAPI { /* 又写一遍 */ };
// ... M 个形状 x N 个后端 = M*N 个具体类
```

| M 形状 × N 后端 | 继承方案（M×N） | 桥接方案（M+N） |
|---|---|---|
| 2 × 2 | 4 | 4 |
| 3 × 3 | 9 | 6 |
| 4 × 4 | 16 | 8 |
| 5 × 5 | 25 | 10 |

更痛的不是类多，而是**每次扩展的代价**：

- 新增 1 个形状 → 继承方案再写 N 个类；桥接方案只加 1 个类；
- 新增 1 个后端 → 继承方案再写 M 个类；桥接方案也只加 1 个类；
- 且后端调用代码在继承方案里每个组合重复一遍，改一处 API 要动 M 处。

## 2. 核心思想：组合替代继承

识别出两个变化的维度后，让**抽象**（Abstraction）持有**实现**
（Implementor）的指针 —— 这根指针就是"桥"：

```
   Abstraction（维度 A：画什么）        Implementor（维度 B：怎么画）
   ─────────────────────────           ─────────────────────────
   ┌──────────────────┐                ┌──────────────────────┐
   │ Shape            │                │ Renderer             │
   │  ◇ renderer_     ├───────────────>│  + draw_circle()     │
   │  + draw()        │        桥      │  + draw_rect()       │
   │  + resize()      │                │  + backend_name()    │
   └───────△──────────┘                └─────────△────────────┘
           │                                     │
   ┌───────┴────────┐                ┌───────────┼─────────────┐
   │                │                │           │             │
 Circle         Rect         OpenGLRenderer VulkanRenderer SoftwareRenderer
```

- `Shape` 只描述**高层语义**（draw / resize），绘制原语全部委托过桥；
- `Renderer` 只描述**底层原语**，不知道 `Shape` 的存在，更不知道有几个形状；
- 两棵树互不引用，各自生长。

## 3. 四个角色（GoF）

| 角色 | 本例对应 | 职责 |
|---|---|---|
| Abstraction | `Shape` | 高层接口，持有 `Renderer`（桥） |
| RefinedAbstraction | `Circle` / `Rect` | 扩充抽象层，编排原语实现自己的语义 |
| Implementor | `Renderer` | 底层原语接口，与抽象解耦 |
| ConcreteImplementor | `OpenGL/Vulkan/SoftwareRenderer` | 对接具体 API |

代码骨架（对应 `main.cpp`）：

```cpp
// Implementor：刻意"低级"，只有原语
class Renderer {
public:
    virtual ~Renderer() = default;
    virtual void draw_circle(float cx, float cy, float r) = 0;
    virtual void draw_rect(float x, float y, float w, float h) = 0;
};

// Abstraction：持有桥，只管语义
class Shape {
public:
    explicit Shape(std::shared_ptr<Renderer> r) : renderer_(std::move(r)) {}
    virtual void draw() const = 0;
    void set_renderer(std::shared_ptr<Renderer> r);   // 运行时换桥
private:
    std::shared_ptr<Renderer> renderer_;              // ★ 桥
};

// RefinedAbstraction：编排原语，不碰任何具体后端
class Circle final : public Shape {
    void draw() const override { impl().draw_circle(cx_, cy_, radius_); }
};
```

> 命名小坑：矩形类叫 `Rect` 而不是 `Rectangle` —— `windows.h`（wingdi.h）
> 在全局命名空间声明了 GDI 函数 `BOOL Rectangle(HDC, ...)`，同名非类型
> 标识符会隐藏类名，导致 `make_unique<Rectangle>` 编译失败。
> 与 `min`/`max` 宏同属 Windows SDK 的名字污染问题。

## 4. 运行时换桥

桥是普通成员指针，因此**可以整体替换**而对象不重建：

```cpp
Circle circle(opengl, 10, 20, 5);   // 构造时架桥（构造器注入）
circle.draw();                      // 走 OpenGL
circle.set_renderer(vulkan);        // 运行时换桥
circle.draw();                      // 同一对象，走 Vulkan
```

几何状态（圆心、半径）始终活在抽象层 —— `resize(1.5)` 只改
`radius_`，桥两端的接口都没有任何变化。

## 5. 与相近模式的区别

| 模式 | 意图 | 与 Bridge 的区别 |
|---|---|---|
| **Strategy**（行为型） | 封装一族可替换算法 | 代码形状几乎一样；区别在意图：Strategy 只有**一边**有类层次（Context 不为策略建树），Bridge **两边都有**且都长期演化 |
| **Adapter**（结构型） | 事后让不兼容的接口合作 | Adapter 着眼"改接口"，Bridge 着眼"预先把两个维度拆开" |
| **Decorator**（结构型） | 动态叠加职责 | Decorator 与被装饰者**同接口**（is-a 且 has-a），Bridge 两端是**不同**的接口 |

> 口诀：**两边都有层次 → Bridge；只有一边 → Strategy。**

## 6. 何时使用

✔ 适合：

- 两个维度确有独立扩展的需求（形状×后端、控件×平台、消息×渠道）；
- 希望运行时切换实现（换后端、换通道）；
- 跨平台层（GUI 库的 Window × Win32/X11/Cocoa —— 这正是 GoF 原书例子）。

✘ 不适合：

- 只有一个维度会变（直接 Strategy，甚至一个函数参数就够）；
- 两个维度永远只有一种组合（继承/直接实现即可，别过度设计）。

## 7. 构建与运行

```bash
cmake --build build --target Learn015 --config Debug
./build/Learn015/Debug/Learn015.exe
```

程序依次演示：

1. `M×N` vs `M+N` 的类数对比表；
2. 2 形状 × 3 后端的 6 种组合 —— 组合是"配出来的"，不是"写出来的"；
3. `set_renderer` 运行时换桥；
4. `resize` 的几何语义留在抽象层，后端无感。

## 8. 实战中的使用案例

以下都是"两个维度各自膨胀"的真实系统。每个案例先点出两个维度（对应
M×N 中的 M 和 N），再看谁扮演四角色 —— 认出这个结构，就能在别人的
代码里认出桥接。

### 案例 1：跨平台 GUI 工具链（GoF 原书例子）

GoF 原书用的正是窗口系统：

| 维度 | 成员 |
|---|---|
| Abstraction（控件树） | `Window` → `IconWindow` / `Button` / `Menu` |
| Implementor（平台后端） | `WindowImp` → `XWindowImp`（X11）/ `PMWindowImp`（OS/2）/ `Win32WindowImp` |

工业实例 —— Qt 的 QPA（Qt Platform Abstraction）：`QWidget`/`QWindow`
控件树不感知平台，经 `QPlatformWindow` 桥接到 `QWindowsWindow`（Windows）、
`QXcbWindow`（Linux/X11）、`QCocoaWindow`（macOS）。移植一个新平台 =
写一个平台插件，控件代码零改动。

### 案例 2：数据库访问层 —— JDBC / ODBC / ADO.NET

| 维度 | 成员 |
|---|---|
| Abstraction（SQL API） | `java.sql.Connection` / `Statement` / `ResultSet` |
| Implementor（驱动） | MySQL / PostgreSQL / Oracle / SQLite 各厂商驱动 |

应用代码只面向 `java.sql.*` 接口编程，`DriverManager` 负责架桥；
换数据库 = 换驱动包 + 连接串，SQL 调用代码一行不动 ——
"面向抽象编程 + 运行时换桥"的标准形态。

### 案例 3：日志系统 —— Python logging / log4j

| 维度 | 成员 |
|---|---|
| Abstraction（记什么） | `Logger` 层级（root → 业务模块 logger：级别过滤、上下文） |
| Implementor（发到哪） | `Handler`/`Appender` → `StreamHandler`（控制台）/ `FileHandler` / `SocketHandler`（远端）/ `SyslogHandler` |

同一个 `Logger` 可同时架**多座**桥（一条日志既写文件又发网络），
还能运行时 `addHandler` 动态加桥 —— 桥接靠组合实现，天然支持一对多。

### 案例 4：C++ 的 PIMPL 惯用法 —— 桥接的退化形式

```cpp
// widget.h —— 头文件只剩一个指针，私有成员全部移进 .cpp
class Widget {
public:
    Widget();
    ~Widget();                      // 析构必须在 .cpp 定义（那里看得到完整 Impl）
    void draw();
private:
    struct Impl;                    // 前向声明
    std::unique_ptr<Impl> impl_;    // ★ 桥（Qt 中称 d-pointer）
};
```

- 相当于 **M = 1 的桥接**：抽象侧没有继承层次，只有一个类；
- Qt 几乎每个公有类都配一个 `XxxPrivate`（d-pointer）；
- 动机与完整桥接不同：**ABI 稳定**（增删私有成员不破坏二进制兼容）、
  **编译防火墙**（实现所需的头文件不再传染给使用者）、缩短全量编译时间。

### 案例 5：图形 API 后端 —— Skia / Dear ImGui

| 维度 | 成员 |
|---|---|
| Abstraction（绘图语义） | 画路径 / 文字 / 图片 / 图层合成 |
| Implementor（GPU 后端） | OpenGL / Vulkan / Metal / DirectX / CPU 软件光栅化 |

Chrome 的 Skia 用一套高层绘图 API 桥接到各 GPU 后端；Dear ImGui 同理
（`ImGui_ImplOpenGL3` / `ImGui_ImplVulkan` / `ImGui_ImplDX11` 各为一座
现成的桥）。本目录的 `Shape × Renderer` 正是它的教学缩影。

### 练习建议：消息系统

消息类型（普通 / 加急 / 带回执）× 发送渠道（Email / SMS / 微信推送）
是另一经典场景：四角色齐备，且"加急消息需同时走两条渠道"可以练习
一座抽象架多座桥的组合用法，适合作为第二个实现来练手。
