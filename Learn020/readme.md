# Learn020 — 工厂 × 建造者：创建型模式的组合拳

模式组合第一课。一句话：

> **工厂选"谁来装"，建造者管"怎么装"** —— 门面是工厂的
> 一步到位，内脏是建造者的分步装配与交付校验。

系列口诀链（承接 [Learn015](../Learn015)–[Learn019](../Learn019)）：

| 场景 | 模式 | 手段 |
|---|---|---|
| 两个维度都要独立扩展 | Bridge | 组合，两边各一棵树 |
| 整个算法可替换 | Strategy | 组合，运行时可换 |
| 骨架相同、只有个别步骤不同 | Template Method | 继承，子类填步骤 |
| 复杂对象分步组装 | Builder | 分步接口 + 导演配方 |
| 创建对象/对象族，屏蔽 new | Factory 家族 | 创建函数抽象化 |
| **两者结合：门面工厂 + 装配车间** | **Factory × Builder** | **注册表选建造者** |

---

## 1. 动机：一个角色诞生需要四份力量

只工厂：一步交付，但复杂产品的中间步骤无从干预；
只建造者：分步严谨，但"选哪个建造者 + 特有参数预设"散在调用方。
合起来正好补齐 —— 再嵌一个抽象工厂供应装备族：

```
调用方: forge.create("warrior", "champion", "Kael")
  │  一步到位（工厂门面）
  ▼
CharacterForge（注册表工厂，前门）
  ├── 类型表: "warrior"  → WarriorBuilder        ← 工厂选建造者
  │           "berserker" → 预配置(rage=200)      ← 工厂级参数化
  │           "frost-mage" → 预配置(Frost)
  └── 配方表: "newbie" / "champion" → Recipe     ← 导演的配方
  ▼
配方驱动建造者分步装配
  ├── build_identity  身份（公共）
  ├── build_stats     属性（读建造者配置 rage/element）
  ├── build_gear      装备 ←── KitFactory（抽象工厂：整族配套供应）
  ├── build_skills    技能
  └── deliver()       交付校验（半成品出不了厂）
  ▼
GameCharacter（产品）
```

## 2. 各模式在流水线上的职责

| 模式 | 本例角色 | 只回答一个问题 |
|---|---|---|
| 注册表工厂（Learn019） | `CharacterForge` 的类型表/配方表 | 选**谁来装**、按**哪张配方** |
| 建造者（Learn018） | `CharacterBuilder` 层次 + `deliver()` | **怎么装**、成品何时合格 |
| 抽象工厂（Learn019） | `KitFactory`（战士/法师装备族） | 部件从**哪族来**（配套供应） |
| 导演（Learn018） | `CharacterDirector` 的配方 | 步骤的**顺序与追加件** |
| 模板方法（Learn017） | `deliver/reset/build_gear` 非虚流程 | 骨架不变式谁说了算（基类） |

## 3. 关键代码骨架

```cpp
class CharacterForge {
    std::map<std::string, BuilderCreator> builders_;  // 类型名 → 建造者
    std::map<std::string, Recipe> recipes_;           // 配方名 → 装配流程
public:
    // 一步到位：工厂体验，内部全流程
    GameCharacter create(type, recipe, name) {
        CharacterBuilder& b = *builder_for(type);     // 选谁装
        return recipes_.at(recipe)(b, name);          // 怎么装
    }
    // 拆开用：拿建造者自己驱动（灵活模式）
    std::unique_ptr<CharacterBuilder> builder_for(type);
};

// 预配置变体：Learn018 的差异参数在工厂层安家
forge.register_builder("berserker", [] {
    auto b = std::make_unique<WarriorBuilder>();
    b->set_rage(200);          // 工厂级参数化
    return b;
});

// 建造者步骤里嵌套抽象工厂：工位向配件厂整族下单
void build_gear() {    // CharacterBuilder 公共实现
    const KitFactory& kit = kit_factory();   // 族绑定
    product_.equipment_.push_back(kit.make_weapon()->name());
    product_.equipment_.push_back(kit.make_armor()->name());
    product_.equipment_.push_back(kit.make_potion()->name());
}
```

## 4. 两种用法并存

| | 一步到位 `create()` | 拆开用 `builder_for()` |
|---|---|---|
| 体验 | 工厂：一个调用拿成品 | 建造者：自己分步驱动 |
| 适用 | 常规角色（配方够用） | 定制角色（中途干预、插私货件） |
| 特有参数 | 走注册变体（`"berserker"`） | Learn018 四方案的阵地 |

## 5. 何时组合 / 何时不组合

✔ 适合组合：

- 产品复杂（要分步装配 + 交付校验）**且**类型多/要配置化选择
  （注册表）；
- 部件需要成族供应（装备族、平台族）；
- 同一系统既要"常规一键产出"又要"高级定制入口"。

✘ 不必为组合而组合：

- 产品三两个字段、一步能造好 —— 工厂单用即可（建造者多余）；
- 类型唯一且固定 —— 建造者单用即可（注册表多余）。

> 经验法则：先把单模式的痛点列出来，**每个痛点引入一个模式**，
> 而不是照着"全家桶"菜单堆砌。

## 6. 构建与运行

```bash
cmake --build build --target Learn020 --config Debug
./build/Learn020/Debug/Learn020.exe
```

程序依次演示：

1. 流水线分工图（四个模式各司其职）；
2. 一步到位创建（warrior × newbie/champion、mage × newbie）；
3. 预配置变体（`berserker` rage 200、`frost-mage` Frost）；
4. 拆开用（`builder_for` + 手动分步 + 插私货装备）；
5. 两张表各守各的门（未知类型/未知配方分别抛错被捕获）。

## 7. 实战中的使用案例

### 案例 1：OkHttp / Retrofit —— 门面工厂 + Request.Builder

`OkHttpClient`（工厂：连接池、拦截器配置）与
`Request.Builder().url(...).header(...).build()`（建造者）共存于
同一次请求 —— 与本课 `Forge.create()` 内部跑装配线同构。

### 案例 2：SQLAlchemy —— sessionmaker 工厂 + Query 构建

`sessionmaker(engine)` 一步产出会话（工厂）；会话上
`query(User).filter(...).order_by(...)` 分步构建查询（建造者）。
数据库 ORM 几乎全是这个组合。

### 案例 3：Vulkan —— 工厂入口 + 描述符装配

`vkCreateDevice/vkCreateGraphicsPipelines` 是工厂入口（按配置
产出成品句柄），入口前填的那一串结构体（createInfo 层层嵌套、
逐字段设置再一次性物化）就是建造者的"C API 面孔"
（Learn018 案例 3 的深化：两者在同一个 API 里出现）。

### 案例 4：游戏角色系统 —— 本例的原型

暗黑/WoW 式 hero 系统：职业注册表（工厂）+ 天赋/装备装配
（建造者）+ 套装整族掉落（抽象工厂）+ 出生模板（导演配方）。
Spawner 按配置名一键造怪是 `create()`，角色编辑器逐项调整是
`builder_for()` 拆开用。

### 练习建议

1. 注册新类型 `"archer"`：加 `ArcherBuilder` + 一行
   `register_builder`（差异参数 `quiver` 容量）—— 验证"加类不改旧"；
2. 注册新配方 `"raid-boss"`（40 级 + 双倍装备 + 自定义头衔）——
   验证配方轴独立扩展；
3. 给 `CharacterForge::create` 加一个 `Learn018Config` 风格的
   参数基类入口（`create(type, recipe, name, const BuildConfig&)`），
   把"拆开用才能设参数"的剩余场景也收进门面。
