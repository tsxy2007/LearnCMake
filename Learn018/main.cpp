/**
 * @file    main.cpp
 * @brief   建造者模式（Builder Pattern）—— 分步构建复杂对象，构建过程与表示分离
 *
 * 定义（GoF 创建型模式）：
 *   将一个复杂对象的**构建**与它的**表示**分离，使同样的构建过程
 *   可以创建**不同的表示**。
 *
 * 本例场景（游戏角色创建，延续 Learn017 的游戏语境）：
 *   一个角色是"多部件复杂对象"：身份、六项属性、装备列表、技能表。
 *   直接用构造函数组装有两个痛点：
 *
 *   痛点 1 —— 伸缩构造函数（telescoping constructor）：
 *     GameCharacter("Kael", "", 1, 120, 10, 15, 13, 5,
 *                   {"Iron Sword"}, {"Power Strike"});
 *     // 哪个 13 是防御？哪个 5 是敏捷？没人读得懂；
 *     // 想跳过某可选字段还得补一长串占位值。
 *
 *   痛点 2 —— 组装流程散落：
 *     "先定身份 → 按职业算属性 → 套默认装备 → 配技能表"这套流程
 *     若写在调用方，每个出生点都要抄一遍，且顺序无约束。
 *
 * 建造者模式的结构：
 *
 *        Director（导演：持有配方）        Builder（抽象建造者）
 *        ─────────────────────           ─────────────────
 *   ┌──────────────────────────┐   ┌──────────────────────────────┐
 *   │ CharacterDirector        │   │ CharacterBuilder             │
 *   │  construct_newbie()      │──>│  build_identity() = 0        │
 *   │  construct_arena_champ() │驱动│  build_stats()    = 0        │
 *   └──────────────────────────┘   │  build_gear()     = 0        │
 *      同一配方（构建过程）          │  build_skills()   = 0        │
 *      × 不同建造者（表示）          │  + reset()/deliver()  公共流程│
 *                                  └──────────△───────────────────┘
 *                                             │ 继承
 *                                  ┌──────────┴──────────┐
 *                                  │ WarriorBuilder      │ MageBuilder
 *                                  │  近战属性/铁剑/板甲  │  法师属性/木杖/长袍
 *                                  └──────────┬──────────┘
 *                                             │ 产出
 *                                  ┌──────────▼──────────┐
 *                                  │ GameCharacter（产品）│
 *                                  └─────────────────────┘
 *
 * 四个角色（GoF）：
 *   Product         —— GameCharacter：被组装的复杂对象（默认构造私有，
 *                      只能从建造者"出生"）
 *   Builder         —— CharacterBuilder：分步构建的抽象接口 +
 *                      公共交付/复位流程
 *   ConcreteBuilder —— WarriorBuilder / MageBuilder：两种"表示"
 *                      （属性怎么算、默认装备技能是什么）
 *   Director        —— CharacterDirector：持有"配方"（构建过程/顺序），
 *                      只依赖抽象 Builder —— 这正是定义后半句的落点：
 *                      同样的 construct_arena_champion() 配方，
 *                      套战士建造者出战士冠军，套法师建造者出法师冠军。
 *
 * 与系列前作的关系：
 *   - Builder 的 deliver()/reset() 是非虚公共流程 —— 与 Learn017
 *     模板方法同源（骨架独裁：校验/复位不允许子类绕过）；
 *   - 与工厂三兄弟（Factory Method / Abstract Factory）的分工见
 *     readme 对比表：工厂"一步到位给你整对象"，建造者"分步组装、
 *     过程可干预、成品最后交付"。
 *
 * C++ 实现注记：
 *   - 产品的默认构造私有 + friend 声明建造者 —— 除建造者外无人能
 *     造出"半成品角色"。注意 friend 不继承：抽象基类与两个具体
 *     建造者都要逐个登记（换来的是毛坯状态外界不可触碰）；
 *   - 交付时统一校验不变式（有名字、有属性、有技能），不完整的
 *     半成品在 deliver() 处被拒之门外 —— 类比 Learn017 的公共门槛。
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <iostream>
#include <stdexcept>   // std::runtime_error
#include <string>
#include <utility>     // std::move
#include <vector>

#ifdef _WIN32
#include <windows.h>   // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

// ============================================================================
// 小工具
// ============================================================================

/// 把字符串列表拼成 "a, b, c"；空列表显示 "(none)"
static std::string join(const std::vector<std::string>& items) {
    if (items.empty()) return "(none)";
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) out += ", ";
        out += items[i];
    }
    return out;
}

// ============================================================================
// Product —— 产品：多部件的复杂对象（角色）
// ============================================================================

/**
 * @brief 游戏角色：身份 + 六项属性 + 装备表 + 技能表
 *
 * 封装要点：默认构造是私有的，且把建造者声明为 friend ——
 * 除建造者外，任何代码都无法创建（更无法改动）一个"毛坯角色"。
 * 产品一经 deliver() 交出，外界只能读（print_sheet），不能改。
 */
class GameCharacter {
public:
    // 拷贝/移动保持默认（交付时把毛坯 move 出去要用）
    GameCharacter(const GameCharacter&) = default;
    GameCharacter& operator=(const GameCharacter&) = default;
    GameCharacter(GameCharacter&&) = default;
    GameCharacter& operator=(GameCharacter&&) = default;

    /// 打印属性卡（产品对外唯一的"脸面"：只读展示）
    void print_sheet(std::ostream& os) const {
        os << "  ==== " << name_ << " " << title_ << " ====" << std::endl;
        os << "  Lv." << level_
           << "  HP " << hp_ << "  MP " << mp_
           << "  ATK " << attack_ << "  DEF " << defense_
           << "  AGI " << agility_ << std::endl;
        os << "  equipment: " << join(equipment_) << std::endl;
        os << "  skills:    " << join(skills_) << std::endl;
    }

private:
    friend class CharacterBuilder;   // 基类：reset/deliver/add_equipment
    friend class WarriorBuilder;     // 具体建造者（friend 不继承，逐个登记）
    friend class MageBuilder;

    GameCharacter() = default;       // 私有：产品只能从建造者"出生"

    std::string name_;
    std::string title_;
    int level_ = 0;
    int hp_ = 0;
    int mp_ = 0;
    int attack_ = 0;
    int defense_ = 0;
    int agility_ = 0;
    std::vector<std::string> equipment_;
    std::vector<std::string> skills_;
};

// ============================================================================
// Builder —— 抽象建造者：分步接口 + 公共交付/复位流程
// ============================================================================

/**
 * @brief 角色建造者：声明"分步构建"的接口，持有毛坯产品
 *
 * 接口分工（呼应 Learn017 模板方法）：
 *   - 纯虚原语 build_identity / build_stats / build_gear / build_skills：
 *     各表示（职业）真正不同的部分，由具体建造者填写；
 *   - 非虚公共流程 reset / deliver / add_equipment：复位、交付校验、
 *     追加零件 —— 顺序与不变式由本类独裁，子类不可绕过。
 */
class CharacterBuilder {
public:
    virtual ~CharacterBuilder() = default;   // 虚析构：经基类指针删除派生对象

    // ---- 公共流程（非虚）----

    /// 复位：开始装配一个新产品（清空毛坯）
    void reset() { product_ = GameCharacter{}; }

    /**
     * @brief 交付：校验不变式后把成品交出，并自动复位迎接下一单
     *
     * 不完整的半成品在此被拒 —— 调用方永远拿不到没有属性/技能
     * 的"残次品角色"（不变式集中在交付口把关，类比 Learn017
     * 的冷却/资源门槛）。
     */
    GameCharacter deliver() {
        if (product_.name_.empty()) {
            throw std::runtime_error(
                "deliver(): character has no name (missing build_identity?)");
        }
        if (product_.hp_ <= 0) {
            throw std::runtime_error(
                "deliver(): invalid stats (missing build_stats?)");
        }
        if (product_.skills_.empty()) {
            throw std::runtime_error(
                "deliver(): no skills (missing build_skills?)");
        }
        GameCharacter out = std::move(product_);   // 成品出仓
        reset();                                    // 工位自动复位
        return out;
    }

    /// 追加单件装备（所有表示共用的"加零件"操作，Director 可用）
    void add_equipment(const std::string& item) {
        product_.equipment_.push_back(item);
    }

    // ---- 分步原语：各表示的差异所在，具体建造者必填 ----

    /// 第 1 步：身份（名字 + 头衔）
    virtual void build_identity(const std::string& name,
                                const std::string& title) = 0;

    /// 第 2 步：按等级计算属性（职业差异的核心：数值分配公式不同）
    virtual void build_stats(int level) = 0;

    /// 第 3 步：默认装备套路（职业差异的另一面）
    virtual void build_gear() = 0;

    /// 第 4 步：默认技能表
    virtual void build_skills() = 0;

protected:
    GameCharacter product_;   // 毛坯产品（protected：具体建造者直接装配）
};

// ============================================================================
// ConcreteBuilder —— 具体建造者：两种"表示"
// ============================================================================

/**
 * @brief 战士建造者：血厚甲厚、近战装备与技能
 */
class WarriorBuilder final : public CharacterBuilder {
public:
    void build_identity(const std::string& name,
                        const std::string& title) override {
        product_.name_ = name;
        product_.title_ = title;
    }

    void build_stats(int level) override {
        product_.level_ = level;
        product_.hp_ = 100 + 20 * level;      // 厚血
        product_.mp_ = 10 + 2 * level;
        product_.attack_ = 12 + 3 * level;    // 近战高攻
        product_.defense_ = 10 + 3 * level;   // 重甲高防
        product_.agility_ = 4 + 1 * level;    // 迟缓
    }

    void build_gear() override {
        product_.equipment_ = { "Iron Sword", "Plate Armor" };
    }

    void build_skills() override {
        product_.skills_ = { "Power Strike", "Shield Wall" };
    }
};

/**
 * @brief 法师建造者：脆皮高蓝、法系装备与技能
 */
class MageBuilder final : public CharacterBuilder {
public:
    void build_identity(const std::string& name,
                        const std::string& title) override {
        product_.name_ = name;
        product_.title_ = title;
    }

    void build_stats(int level) override {
        product_.level_ = level;
        product_.hp_ = 60 + 8 * level;        // 脆皮
        product_.mp_ = 80 + 15 * level;       // 巨量法力
        product_.attack_ = 6 + 1 * level;     // 杖击很弱
        product_.defense_ = 3 + 1 * level;    // 布甲低防
        product_.agility_ = 6 + 2 * level;    // 敏捷
    }

    void build_gear() override {
        product_.equipment_ = { "Oak Staff", "Cloth Robe" };
    }

    void build_skills() override {
        product_.skills_ = { "Fireball", "Frost Nova", "Mana Shield" };
    }
};

// ============================================================================
// Director —— 导演：持有"配方"（构建过程），只依赖抽象建造者
// ============================================================================

/**
 * @brief 角色导演：把"按什么顺序装哪些部件"固化成配方
 *
 * 配方只调用 CharacterBuilder 的抽象接口 —— 同一张配方
 * 换个建造者就产出另一种表示（战士冠军 / 法师冠军），
 * 这就是"同样的构建过程创建不同的表示"。
 * Director 也可以没有（调用方手动分步驱动，见演示 [4]）——
 * 它是"可复用的流程"，不是必选项。
 */
class CharacterDirector {
public:
    /// 配方 1：新手角色（1 级、默认套装）
    GameCharacter construct_newbie(CharacterBuilder& builder,
                                   const std::string& name) {
        builder.reset();
        builder.build_identity(name, "the Rookie");
        builder.build_stats(/*level=*/1);
        builder.build_gear();
        builder.build_skills();
        return builder.deliver();
    }

    /// 配方 2：竞技场冠军（20 级 + 双件冠军饰品）
    GameCharacter construct_arena_champion(CharacterBuilder& builder,
                                           const std::string& name) {
        builder.reset();
        builder.build_identity(name, "the Arena Champion");
        builder.build_stats(/*level=*/20);
        builder.build_gear();
        builder.build_skills();
        builder.add_equipment("Champion's Cape");    // 配方追加的公共件
        builder.add_equipment("Ring of Victory");
        return builder.deliver();
    }
};

// ============================================================================
// 主函数 —— 伸缩构造之痛、配方 × 表示、手动建造、交付校验
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Builder Pattern: assemble a complex" << std::endl;
    std::cout << "  character step by step" << std::endl;
    std::cout << "========================================" << std::endl;

    WarriorBuilder warrior_builder;
    MageBuilder mage_builder;
    CharacterDirector director;

    // ==== 演示 1：伸缩构造函数之痛（动机）====

    std::cout << std::endl;
    std::cout << "[1] The pain first: telescoping constructor" << std::endl;
    std::cout << "    GameCharacter(\"Kael\", \"\", 1, 120, 10, 15, 13, 5," << std::endl;
    std::cout << "                  {\"Iron Sword\"}, {\"Power Strike\"});" << std::endl;
    std::cout << "    // which number is defense? which is agility?" << std::endl;
    std::cout << "    // unreadable, order-fragile, every optional field" << std::endl;
    std::cout << "    // still needs a placeholder slot" << std::endl;
    std::cout << "    -> builder: named steps, order owned by the flow," << std::endl;
    std::cout << "       invariants checked at deliver()" << std::endl;

    // ==== 演示 2：同一配方 × 两种建造者 ====

    std::cout << std::endl;
    std::cout << "[2] One recipe (newbie) on two builders:" << std::endl;

    GameCharacter w1 = director.construct_newbie(warrior_builder, "Kael");
    GameCharacter m1 = director.construct_newbie(mage_builder, "Elyra");
    w1.print_sheet(std::cout);
    m1.print_sheet(std::cout);
    std::cout << "    (same construct_newbie() flow, different stats/gear/skills)" << std::endl;

    // ==== 演示 3：换一张配方，两种表示各自"升满" ====

    std::cout << std::endl;
    std::cout << "[3] Another recipe (arena champion), same two builders:" << std::endl;

    GameCharacter w2 = director.construct_arena_champion(warrior_builder, "Kael");
    GameCharacter m2 = director.construct_arena_champion(mage_builder, "Elyra");
    w2.print_sheet(std::cout);
    m2.print_sheet(std::cout);
    std::cout << "    (level-20 stats + champion trinkets, still one recipe)" << std::endl;

    // ==== 演示 4：不用导演 —— 调用方手动分步驱动 ====

    std::cout << std::endl;
    std::cout << "[4] Director-less: drive the builder manually" << std::endl;
    std::cout << "    (custom level, custom title, one extra trinket):" << std::endl;

    mage_builder.reset();                        // 手动开单
    mage_builder.build_identity("Sylra", "the Wanderer");
    mage_builder.build_stats(/*level=*/12);      // 自选等级
    mage_builder.build_gear();                   // 套默认装备
    mage_builder.build_skills();                 // 套默认技能
    mage_builder.add_equipment("Amulet of Wisdom");   // 再加一件私货
    GameCharacter custom = mage_builder.deliver();
    custom.print_sheet(std::cout);
    std::cout << "    (order and choices under caller's control;" << std::endl;
    std::cout << "     Director is a reusable flow, not a requirement)" << std::endl;

    // ==== 演示 5：交付校验 —— 半成品出不了厂 ====

    std::cout << std::endl;
    std::cout << "[5] deliver() rejects an incomplete build:" << std::endl;
    std::cout << "    (identity only -- no stats, no skills)" << std::endl;

    warrior_builder.reset();
    warrior_builder.build_identity("Broken", "the Half-Done");
    try {
        GameCharacter bad = warrior_builder.deliver();   // 必然抛异常
        bad.print_sheet(std::cout);                      // 永远走不到这里
    } catch (const std::runtime_error& e) {
        std::cout << "    caught: " << e.what() << std::endl;
    }
    std::cout << "    -> callers can never obtain a half-built product" << std::endl;

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. Four roles: Director(recipe) drives Builder" << std::endl;
    std::cout << "     (interface); ConcreteBuilder(representation)" << std::endl;
    std::cout << "     assembles Product(complex object)." << std::endl;
    std::cout << "  2. Same construction process, different" << std::endl;
    std::cout << "     representations: recipe x builder are decoupled." << std::endl;
    std::cout << "  3. deliver() = single exit gate with invariants" << std::endl;
    std::cout << "     (same spirit as Learn017's common gates)." << std::endl;
    std::cout << "  4. Private default ctor + friend builders: nobody" << std::endl;
    std::cout << "     else can create or mutate the bare product." << std::endl;
    std::cout << "  5. vs factories: factory returns a finished object" << std::endl;
    std::cout << "     in one shot; builder assembles step by step and" << std::endl;
    std::cout << "     the caller may intervene mid-process." << std::endl;

    return 0;
}
