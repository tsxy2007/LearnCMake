# Learn019 — 工厂模式三兄弟：简单工厂 / 工厂方法 / 抽象工厂

GoF 创建型模式（其中工厂方法、抽象工厂在 23 式之内）。一句话：

> 把 **"new 哪个具体类"** 封装起来 —— 创建逻辑与使用逻辑分离，
> 使用方只依赖**抽象产品**接口。

系列口诀链（承接 [Learn015](../Learn015)–[Learn018](../Learn018)）：

| 场景 | 模式 | 手段 |
|---|---|---|
| 两个维度都要独立扩展 | Bridge | 组合，两边各一棵树 |
| 整个算法可替换 | Strategy | 组合，运行时可换 |
| 骨架相同、只有个别步骤不同 | Template Method | 继承，子类填步骤 |
| 复杂对象分步组装 | Builder | 分步接口 + 导演配方 |
| **创建对象/对象族，屏蔽 new** | **Factory 家族** | **创建函数抽象化** |

---

## 1. 问题：散落的 new + 条件判断

使用方代码直接 `new Sword()` / `new Bow()` 会带来两件事：

1. **依赖具体类型**：换产品、加产品，所有调用点都要改；
2. **条件逻辑蔓延**：`if (style == melee) ... else ...` 的创建分支
   在每个出生点重复。

工厂家族把创建收敛到一处/一类，使用方只见抽象产品。

## 2. 简单工厂（非 GoF，但人人从它开始）

```cpp
class SimpleWeaponFactory {
public:
    static std::unique_ptr<Weapon> create(WeaponType type) {
        switch (type) {                       // 所有 new 集中于此
        case WeaponType::Sword:     return std::make_unique<Sword>();
        case WeaponType::Bow:       return std::make_unique<Bow>();
        case WeaponType::MagicWand: return std::make_unique<MagicWand>();
        }
        throw std::runtime_error("unknown weapon type");
    }
};
```

- ✔ 最简单，一个函数解决；
- ✘ 新增产品必须**修改** switch —— 违反开闭原则
  （与 Learn016 策略模式里 `solve()` 的 switch 是同一个坏味道）。

## 3. 工厂方法（GoF）：把 create 声明为虚函数

```cpp
class WeaponFactory {                          // Creator
public:
    // 公共流程（非虚）—— 这就是 Learn017 的模板方法：
    // 骨架调用虚的原语，顺手"试一击"
    std::unique_ptr<Weapon> create_and_test() const {
        auto weapon = create_weapon();         // ★ 工厂方法
        weapon->attack();
        return weapon;
    }
    virtual std::unique_ptr<Weapon> create_weapon() const = 0;
    virtual std::string name() const = 0;
};

class AxeFactory final : public WeaponFactory {   // 新增 = 加类，不改旧的
    std::unique_ptr<Weapon> create_weapon() const override {
        return std::make_unique<Axe>();
    }
};
```

- ✔ 兑现 OCP：新增 Axe 产品 = `Axe` + `AxeFactory` 两个类，旧代码零修改；
- ✘ 代价：产品与工厂类数量 **2:1** 膨胀。

> GoF 细节：Creator 通常不只有 create —— 它还承载**使用产品**的公共
> 流程（本例 `create_and_test`）。工厂方法只是骨架中的创建步骤，
> 这正是 Learn017 模板方法的结构。

## 4. 抽象工厂（GoF）：一次生产一族配套产品

```cpp
class KitFactory {                             // 抽象工厂：一族产品的创建接口
    virtual std::unique_ptr<Weapon> make_weapon() const = 0;
    virtual std::unique_ptr<Armor>  make_armor()  const = 0;
    virtual std::unique_ptr<Potion> make_potion() const = 0;
};

class WarriorKitFactory final : public KitFactory { /* 剑+板甲+治疗药水 */ };
class MageKitFactory     final : public KitFactory { /* 杖+长袍+法力药水 */ };
```

- 关键词是**族**（family）：三个 `make_*` 产出互相配套的产品，
  战士套件里绝不会混进法杖 —— 配套性由具体工厂一手保证；
- 换风格 = 换**整个工厂对象**（近战全家桶 ↔ 法系全家桶）；
- Learn018 readme 的"工厂对比表"在此补全：抽象工厂 = "一族怎么配套"。

## 5. 注册表工厂：switch 的查表替代品

```cpp
using WeaponCreator = std::function<std::unique_ptr<Weapon>()>;
std::map<std::string, WeaponCreator> registry;

registry["sword"] = [] { return std::make_unique<Sword>(); };
registry["crossbow"] = [] { return std::make_unique<Bow>(); };  // 运行期注册

auto w = registry.at("crossbow")();            // 查表创建
```

- 兼取两家长处：像简单工厂一样**不膨胀类**，像工厂方法一样
  **不改旧代码**（注册一个 lambda 即新增产品）；
- 注册开放到**运行期** —— 动态库加载、插件初始化、脚本配置的
  标准姿势（工业实现常配全局单例 + 各产品在自己 .cpp 里的
  静态自注册对象）；
- 查不到名字立即抛错 —— 对照简单工厂 switch 漏分支的未定义行为。

## 6. 四者怎么选

| | new 的决策者 | 新增产品 | 类膨胀 | 特色 |
|---|---|---|---|---|
| 简单工厂 | 一个 switch | **改**旧代码 | 无 | 最省事 |
| 工厂方法 | 子类工厂 | 加 2 个类 | 2:1 | OCP 教科书形态 |
| 抽象工厂 | 一个族工厂 | 加整族产品 + 工厂 | 族级 | 产品配套一致性 |
| 注册表 | 查表 + lambda | 注册一条 | 无 | 运行期开放（插件） |

与 Builder（Learn018）的边界：**简单对象一步交付 → 工厂；
复杂对象分步组装、交付口校验 → 建造者**。二者常组合：
建造者的选择本身可以交给工厂。

## 7. 何时使用

✔ 适合：

- 使用方不应依赖具体类型（面向抽象编程的创建端）；
- 产品种类会增长，或按配置/平台/风格在运行期决定；
- 产品需要成套出现（抽象工厂）；
- 支持第三方插件注册新类型（注册表）。

✘ 不适合：

- 产品只有一个实现且可预见不会变（直接 `make_unique` 即可）；
- 对象创建只是几行平凡初始化、且不构成依赖问题 —— 过度设计。

## 8. 构建与运行

```bash
cmake --build build --target Learn019 --config Debug
./build/Learn019/Debug/Learn019.exe
```

程序依次演示：

1. 简单工厂创建三种武器 + OCP 问题点；
2. 工厂方法：三个工厂经 `create_and_test` 公共流程出招；
   后来新增的 `AxeFactory` 证明"加类不改旧"；
3. 抽象工厂：战士/法师两族装备套件，配套性演示；
4. 注册表：按名创建、运行期注册 `crossbow`、未知名字抛错被捕获；
5. 四形态选择指南 + 与系列前作的连接。

## 9. 实战中的使用案例

### 案例 1：跨平台集成 —— Qt QPA（抽象工厂）

Learn015 提过 Qt 的平台抽象：每个平台插件（Windows/X11/Cocoa）就是
一个**抽象工厂**，一次性产出该平台的一族对象（窗口、事件循环、
字体引擎、剪贴板）—— 换平台 = 换整个工厂，族内配套永不混搭。

### 案例 2：驱动与算法选择 —— JDBC / PETSc（工厂方法 + 注册表）

- Java `DriverManager.getConnection(url)`：按 URL 前缀查注册表，
  返回对应厂商的 `Connection` 实现（Learn016 readme 案例 2 的
  创建端就是这么来的）；
- PETSc `KSPCreate` + `KSPSetType("cg"/"gmres")`：按名字从注册表
  取求解器构造函数 —— Learn016 的 Solver 族用注册表工厂串起来。

### 案例 3：游戏内容生态 —— 刷怪器与模组（注册表工厂）

游戏引擎的 spawner 按配置名从注册表创建实体/武器；模组（mod）
加载动态库时向同一注册表**运行期注册**新类型 —— 本课演示 [4]
的 `registry["crossbow"] = ...` 就是这个机制的缩影。

### 案例 4：标准库与日常 C++

- `std::make_unique/make_shared` 是语言级的工厂函数（封装 new 的
  最小形态）；
- 序列化框架的反序列化（按类型名/标签重建对象）、单元测试里
  按名字创建 fixture，几乎都是注册表工厂。

### 练习建议

1. 给注册表加"自注册"形态：让每个武器类在自己的 .cpp 里用一个
   静态 `bool registered = (registry["sword"] = ..., true);` 完成登记；
2. 增加第三族 `AssassinKitFactory`（匕首 + 皮甲 + 迅捷药水），
   验证抽象工厂"加族不动使用方"；
3. 用注册表工厂改写 Learn016 的 Solver 选择、Learn015 的 Renderer
   选择 —— 体会"机制同源、意图不同"。
