# Learn018 — 建造者模式（Builder Pattern）：分步构建复杂对象

GoF 23 种设计模式中的**创建型**模式。一句话：

> 将一个复杂对象的**构建**与它的**表示**分离，使同样的构建过程
> 可以创建**不同的表示**。

系列口诀链（承接 [Learn015](../Learn015)–[Learn017](../Learn017)）：

| 场景 | 模式 | 手段 |
|---|---|---|
| 两个维度都要独立扩展（Shape × Renderer） | Bridge | 组合，两边各一棵树 |
| 整个算法可替换（换一种解法器） | Strategy | 组合，运行时可换 |
| 骨架相同、只有个别步骤不同 | Template Method | 继承，子类填步骤 |
| 复杂对象分步组装，过程与表示分离 | **Builder** | **分步接口 + 导演配方** |

本例场景：**游戏角色创建**（身份 + 六项属性 + 装备表 + 技能表）。

---

## 1. 问题：伸缩构造函数 + 流程散落

```cpp
// 痛点：这个调用没人读得懂 —— 哪个 13 是防御？哪个 5 是敏捷？
// 想跳过可选字段还得补一长串占位值
GameCharacter c("Kael", "", 1, 120, 10, 15, 13, 5,
                {"Iron Sword"}, {"Power Strike"});
```

而且"先定身份 → 按职业算属性 → 套默认装备 → 配技能表"这套组装
流程若写在调用方，每个出生点都要抄一遍，顺序也无约束。

## 2. 核心思想：把"装哪些部件、按什么顺序装"与"部件怎么造"分开

```
        Director（导演：持有配方）        Builder（抽象建造者）
        ─────────────────────           ─────────────────
  ┌──────────────────────────┐   ┌──────────────────────────────┐
  │ CharacterDirector        │   │ CharacterBuilder             │
  │  construct_newbie()      │──>│  build_identity() = 0        │
  │  construct_arena_champ() │驱动│  build_stats()    = 0        │
  └──────────────────────────┘   │  build_gear()     = 0        │
     同一配方（构建过程）          │  build_skills()   = 0        │
     × 不同建造者（表示）          │  + reset()/deliver() 公共流程 │
                                 └──────────△───────────────────┘
                                            │ 继承
                                 ┌──────────┴──────────┐
                                 │ WarriorBuilder      │ MageBuilder
                                 │  近战属性/铁剑/板甲  │  法师属性/木杖/长袍
                                 └──────────┬──────────┘
                                            │ 产出
                                 ┌──────────▼──────────┐
                                 │ GameCharacter（产品）│
                                 └─────────────────────┘
```

- **配方 × 建造者解耦**：`construct_arena_champion()` 这张配方
  套战士建造者出战士冠军，套法师建造者出法师冠军 ——
  "同样的构建过程，不同的表示"；
- **Director 是可复用的流程，不是必选项**：调用方也可以不用导演、
  手动分步驱动建造者（自定义等级/头衔/加一件私货装备）；
- **deliver() 是唯一出口**：交付时统一校验（有名字、有属性、有
  技能），半成品出不了厂。

## 3. 四个角色（GoF）

| 角色 | 本例对应 | 职责 |
|---|---|---|
| Product | `GameCharacter` | 被组装的复杂对象；默认构造私有，只能从建造者"出生" |
| Builder | `CharacterBuilder` | 分步构建接口 + 非虚的 reset/deliver/add_equipment 公共流程 |
| ConcreteBuilder | `WarriorBuilder` / `MageBuilder` | 两种"表示"：属性公式、默认装备与技能 |
| Director | `CharacterDirector` | 配方（构建顺序与追加件），只依赖抽象 Builder |

代码骨架（对应 `main.cpp`）：

```cpp
class GameCharacter {
private:
    friend class CharacterBuilder;   // 基类：reset/deliver/add_equipment
    friend class WarriorBuilder;     // 具体建造者逐个登记（friend 不继承）
    friend class MageBuilder;
    GameCharacter() = default;       // 私有：产品只能从建造者"出生"
    /* name/title/level/hp/... equipment/skills */
};

class CharacterBuilder {
public:
    void reset();                        // 非虚公共流程：清空毛坯
    GameCharacter deliver();             // 非虚公共流程：校验 + 交付 + 自动复位
    void add_equipment(const std::string&);   // 公共"加零件"
    virtual void build_identity(...) = 0;     // 分步原语，表示各异
    virtual void build_stats(int) = 0;
    virtual void build_gear() = 0;
    virtual void build_skills() = 0;
protected:
    GameCharacter product_;              // 毛坯
};

class CharacterDirector {                // 导演：配方
public:
    GameCharacter construct_arena_champion(CharacterBuilder& b, name) {
        b.reset();
        b.build_identity(name, "the Arena Champion");
        b.build_stats(20);
        b.build_gear();  b.build_skills();
        b.add_equipment("Champion's Cape");
        return b.deliver();
    }
};
```

## 4. 交付即门槛：`deliver()`

```cpp
GameCharacter deliver() {
    if (product_.name_.empty()) throw std::runtime_error("no name");
    if (product_.hp_ <= 0)      throw std::runtime_error("invalid stats");
    if (product_.skills_.empty()) throw std::runtime_error("no skills");
    GameCharacter out = std::move(product_);   // 成品出仓
    reset();                                    // 工位自动复位
    return out;
}
```

与 Learn017 的公共门槛同一精神：**不变式集中在唯一出口把关**，
调用方永远拿不到没有属性/技能的"残次品角色"。程序演示 [5] 里
只装了身份就 `deliver()` 的建造者被异常拒之门外。

## 5. 与工厂三兄弟的对比（创建型家族）

| | Factory Method | Abstract Factory | Builder |
|---|---|---|---|
| 产出 | 单个对象（子类决定哪个） | 一族相关对象 | 一个复杂对象 |
| 过程 | 一步到位 | 一步到位（多个工厂方法） | **分步组装** |
| 调用方能否干预过程 | 否 | 否 | **能**（手动驱动/插零件） |
| 中间状态 | 无 | 无 | 有（毛坯，但对外不可见） |
| 典型问题 | "new 谁" | "一族怎么配套" | "参数多、流程长" |

> 经验法则：对象能一次造好（哪怕按族）→ 工厂；
> 对象要**装很多步**、或**同一流程要出多种表示** → 建造者。

## 6. 现代 C++ 的"小写版"：链式调用

工业代码里更常见的是 Builder 的简化形态 —— 返回 `*this` 的链式
setter（named parameter idiom），常与"参数对象 + 一次执行"配合：

```cpp
// 把 Query().select("name").from("users").where("age > 18").build()
// 拆开看：每个 setter 就是 build_xxx 步骤，build() 就是 deliver()
Query q = Query().select("name").from("users").where("age > 18");
```

与 GoF 完整版的差别：没有 Director（流程由调用方链式表达）、
通常一个 Builder 只对应一个 Product（不追求"多表示"）。

## 7. 何时使用

✔ 适合：

- 构造参数多、可选项多（ telescoping constructor 已经出现）；
- 对象需要分阶段装配，且装配顺序有讲究；
- 同一套装配流程要产出多种表示（配方复用）；
- 想对外隐藏中间状态、只在最终交付时校验不变式。

✘ 不适合：

- 三五个必填参数的普通对象（普通构造函数/聚合初始化更直接）；
- 对象创建后仍会频繁变更（那是可变实体，不是"装配出厂"语义，
  建造者反而暗示"成品只读"）。

## 8. 构建与运行

本目录含**同一模式的三个示例**（标准四角色 + 参数不一致两种进阶，见第 10 节）：

```bash
cmake --build build --target Learn018 --config Debug          # 标准四角色
./build/Learn018/Debug/Learn018.exe

cmake --build build --target Learn018Params --config Debug    # 进阶：参数不一致三方案
./build/Learn018/Debug/Learn018Params.exe

cmake --build build --target Learn018Config --config Debug    # 进阶二：参数基类
./build/Learn018/Debug/Learn018Config.exe
```

程序依次演示：

1. 伸缩构造函数之痛（动机）；
2. 同一张新手配方 × 战士/法师两种表示的属性卡；
3. 换竞技场冠军配方（20 级 + 冠军饰品），仍是同一流程；
4. 不用导演手动分步组装自定义角色；
5. `deliver()` 拒绝交付半成品（异常被捕获展示）。

## 9. 实战中的使用案例

以下都是"分步配置 → 一次物化 → 之后只读"的真实系统。

### 案例 1：Rust 标准库 —— `std::process::Command`

```rust
Command::new("sort").arg("-n").current_dir("/tmp").spawn()
//   create → set → set → ... → build(deliver = spawn)
```

Rust 把 builder 当作 API 设计文化：凡是"多可选参数 + 最终执行"
的接口都长这样（`torch.optim`、`HttpServer::new().bind()` 同理）。

### 案例 2：Java 生态 —— StringBuilder / OkHttp / Lombok

| 库 | Builder 用法 |
|---|---|
| JDK | `StringBuilder.append().append().toString()`（deliver = toString） |
| OkHttp | `Request.Builder().url(...).header(...).build()` |
| Lombok | `@Builder` 注解自动生成 builder —— 说明该模式样板代码多到值得代码生成 |

### 案例 3：C API 的"builder 面孔" —— CUDA / Vulkan / cuDNN

无虚函数、无类层次，但结构一模一样：

```cpp
// cuDNN 张量描述符：create → 分步 set → 之后只读
cudnnCreateTensorDescriptor(&desc);
cudnnSetTensor4dDescriptor(desc, /*format*/ ..., /*type*/ ...,
                           n, c, h, w);      // ← build_* 步骤
// 之后 desc 作为"成品"传入每个计算调用，不再修改
```

Vulkan 的管线创建（填一串结构体再
`vkCreateGraphicsPipelines`）、CUDA 的 `cudaLaunchConfig_t`
同理：**分步配置、一次物化、成品只读** —— 这正是建造者要固化的
生命周期。仓库的 Learn004–Learn014 CUDA 程序里 kernel 启动参数
的组装，也可以用这个视角重读一遍。

### 案例 4：SQL 查询构建器 —— SQLAlchemy / Knex / TypeORM

```python
session.query(User).filter(User.age > 18).order_by(User.name).all()
#   select 部分 → where 部分 → order 部分 → deliver(all)
```

分步表达查询的各个部件，最后一次性编译执行 ——
`WHERE` 和 `ORDER BY` 的先后顺序由构建器内部保证，
调用方不可能拼出顺序错误的 SQL 骨架。

### 练习建议

1. 增加第三种表示 `TankBuilder`（血牛：hp 加成、嘲讽技能）——
   只写四个 `build_*` 函数 + 在 `GameCharacter` 登记一行 friend，
   两张既有配方立即能生产坦克；
2. 给 Director 加一张 `construct_raid_boss` 配方（40 级 +
   双倍装备 + 自定义头衔），体会"新增配方不动建造者"的另一条
   扩展轴 —— 配方与表示两张轴独立生长，正是本模式的收益。

## 10. 进阶：每种类型的参数不一致怎么办（params.cpp）

标准 Builder 隐含一个前提：所有表示共用同一套步骤签名。现实里
各类型常有**自己特有的参数**——战士建造者要"怒气上限 rage"，
法师建造者要"元素 element"（决定技能表）。

### 反模式：参数并集接口

```cpp
class CharacterBuilder {
    virtual void set_rage(int) = 0;           // 法师：只能空实现
    virtual void set_element(std::string) = 0;// 战士：只能空实现
};
```

三个坏处：接口膨胀（每加一类就加一组方法）；违反 LSP（一半实现
"收到调用但假装没发生"）；配方里充斥无意义调用或被迫转型猜类型。

### 三种正解

| 方案 | 手法 | 适用 |
|---|---|---|
| ① 配置与毛坯分离 | 差异参数是具体建造者（或族中间层）的自有配置成员 + setter；`build_*` 读配置算产品 | 参数只是"数值调调"，通用配方无需感知它们 |
| ② 接口分层 | 族中间层 `MeleeCharacterBuilder`（+`set_rage`）/ `CasterCharacterBuilder`（+`set_element`）；配方也分层重载 | 参数按**族**分组，配方需要按族特化 |
| ③ 泛型配方 + 能力检测 | 函数模板 `construct_custom<B>` + `void_t` 探测 + `if constexpr` 注入 | 想要**一份**配方编译期自适应任意建造者 |
| ④ **参数基类整体传入** | `BuildConfig` 基类装公共参数，派生 config 装特有参数；`build(const BuildConfig&)` + 守卫转型（config.cpp） | 配置需要**打包传递/拷贝/序列化**（存档、网络、回放），或想让建造者无状态 |

方案 ① 的关键细节 —— **reset() 只清毛坯，不清配置**：

```cpp
warrior_builder.set_rage(60);            // 配置一次
director.construct_newbie(warrior_builder, "Kael");   // 出厂 rage 60
director.construct_newbie(warrior_builder, "Kael");   // 仍是 60
// 配方内部的 reset() 清空的是毛坯产品；rage_ 是建造者配置，原封保留
```

方案 ② 的族配方吃中间层引用——传错族**编译失败**（编译期安全）：

```cpp
GameCharacter construct_furious(MeleeCharacterBuilder& b, name) {
    b.set_rage(200);                     // 名正言顺：近战族都有怒气
    return construct_newbie(b, name);
}
// director.construct_furious(mage_builder, ...)  // ✘ 编译不过
```

方案 ③ 用 C++17 检测惯用法把 Director 编译期化（C++20 写法更直白）：

```cpp
template <typename B, typename = void>
struct has_set_rage : std::false_type {};
template <typename B>
struct has_set_rage<B, std::void_t<decltype(std::declval<B&>().set_rage(0))>>
    : std::true_type {};

template <typename B>
GameCharacter construct_custom(B& b, const std::string& name, int level) {
    b.reset();
    if constexpr (has_set_rage<B>::value)   // 编译期裁剪：
        b.set_rage(180);                    //   近战族注入怒气
    b.build_identity(name, "the Custom");   //   法系族该分支不参与编译
    b.build_stats(level);
    b.build_skills();
    return b.deliver();
}
```

产品侧的配套做法：特有字段做成**可空**（`rage_max_ == 0` /
`element_` 为空 = 无此项），属性卡按需打印 —— 产品模式允许各表示
携带不同的附加属性。

### 方案 ④：参数基类（Parameter Object）整体传入（config.cpp）

把"配置"从建造者身上的 setter 状态变成**独立可传递的对象**：
基类 `BuildConfig` 装公共参数（name/title/level）+ 类型守卫 `kind()`；
派生 `WarriorConfig`（+rage）/ `MageConfig`（+element）装特有参数。

```
调用方: WarriorConfig{rage=60}        ← 派生对象，特有参数就位
   │
导演配方: 只在 BuildConfig 层面填公共参数（name/title/level）
   │        —— 特有参数随派生身份"搭顺风车"原样通过
   ▼
建造者: build(const BuildConfig&) → config_of<WarriorConfig>(cfg)
         守卫转型取回特有参数 → 装配 → deliver()
```

```cpp
// 守卫转型：裸 static_cast 错配时是未定义行为，先对暗号再转
template <typename C>
const C& config_of(const BuildConfig& cfg) {
    if (std::string(cfg.kind()) != C::static_kind())
        throw std::runtime_error("config mismatch: ...");
    return static_cast<const C&>(cfg);
}

// 通用配方：吃 (任意建造者, 任意配置)，只填基类公共参数
GameCharacter construct_newbie(CharacterBuilder& b, BuildConfig& cfg, name);

// 类型特化配方：对派生 config 重载 —— 特化的是数据类，不碰建造者接口
GameCharacter construct_furious(CharacterBuilder& b, WarriorConfig& cfg, name);
```

相比方案 ①–③ 的独特收益：

1. **参数一次打包成型**，调用点自文档（不用一串 setter）；
2. **建造者无状态**：配置不在建造者身上，同一实例可交替/并发装配
   不同 config，配置冲突被消灭；
3. **config 可拷贝、可序列化**：存档系统/网络同步/回放直接存
   config 对象 —— setter 方案做不到；
4. 特化配方只需对派生 config 重载，接口零膨胀（同方案 ②）。

代价（诚实标注）：基类引用进来必然**向下转型** —— 必须守卫
（`kind()` 标签或 `dynamic_cast`），且 builder × config 的错配要到
**运行期**才暴露（方案 ③ 是编译期）。C++20 可用模板 + concepts
消除转型。

### 进阶练习

给 `params.cpp` 加一个 `PriestBuilder`（法系族：元素默认 Holy、
另有族特有参数 `faith`）：分别用三种方案给它接入 —— 中间层加
`set_faith`（方案 ②）、泛型配方里再加一个 `has_set_faith` 分支
（方案 ③）—— 体会三种手法可以叠加使用。
