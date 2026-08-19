/**
 * @file    config.cpp
 * @brief   建造者进阶二 —— 参数基类（Parameter Object）：配置作为对象整体传入
 *
 * 回答一个问题：能否设计一个"参数基类"，每种建造者调用时把基类传进去？
 * 可以，而且这是很常用的第四种解法 —— 把"配置"从建造者身上的
 * setter 状态（params.cpp 方案 1）变成**独立可传递的对象**：
 *
 *        调用方                     导演                      建造者
 *   WarriorConfig{rage=60} ──> construct_newbie(b, cfg) ──> build(cfg)
 *     派生：带特有参数            只在 BuildConfig 层面        守卫转型
 *     (基类部分先留空)            填公共参数(name/title/level)  取回特有参数
 *
 *   参数基类 BuildConfig（公共参数 + 类型守卫）
 *     ├── WarriorConfig：+ rage       （战士特有）
 *     └── MageConfig   ：+ element    （法师特有）
 *
 * 流转的关键：
 *   - 调用方 new 出**派生** config，特有参数已就位；
 *   - 导演配方拿到的是 **BuildConfig&**（基类引用）—— 它只写公共参数，
 *     特有参数随对象的派生身份"搭顺风车"原样通过；
 *   - 建造者的 build() 也只收 const BuildConfig&，具体建造者用
 *     **带守卫的向下转型**（config_of<W>）取回自己的特有参数。
 *
 * 相比 params.cpp 三方案的增益：
 *   1) 参数一次打包成型 —— 调用点自文档（不用一串 setter 调用）；
 *   2) 建造者变成**无状态装配线**：配置不在建造者身上，可复用、
 *      可并发调用同一建造者（配置冲突的可能被消灭）；
 *   3) 配置对象**可拷贝、可存档**：游戏的存档系统/网络同步/
 *      回放系统可以直接序列化 config 对象 —— setter 方案做不到；
 *   4) 特化配方只需对**派生 config** 重载（construct_furious 吃
 *      WarriorConfig&），不碰建造者接口 —— 接口膨胀问题同方案 2。
 *
 * 代价（诚实标注）：
 *   - 基类引用进来，具体建造者必须向下转型 —— 用 kind() 标签守卫
 *     （或 dynamic_cast）把错误"炸得早、信息清楚"（演示 [4]）；
 *   - builder × config 的错配要到**运行期**才暴露（方案 3 的
 *     模板检测是编译期）。C++20 下可用模板 + concepts 消除转型。
 *
 * 与系列的关系：Learn017（模板方法：build() 是非虚装配线）、
 * Learn018 main.cpp（标准四角色）、params.cpp（方案 1-3）、
 * 本文件（方案 4：参数基类整体传入）。
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
// 参数基类（Parameter Object）—— 公共参数 + 类型守卫
// ============================================================================

/**
 * @brief 构建参数基类：所有表示共用的参数 + 向下转型的守卫
 *
 * 设计要点：
 *   - 公共参数（name/title/level）直接做成数据成员 —— 导演配方
 *     在基类层面就能填它们；
 *   - kind() 是"身份证"：具体建造者转型前先对暗号，把
 *     "错误的静默转型"变成"立刻抛出的清晰错误"。
 */
class BuildConfig {
public:
    virtual ~BuildConfig() = default;

    std::string name;      // 公共参数：角色名（配方负责填）
    std::string title;     // 公共参数：头衔（配方负责填）
    int level = 1;         // 公共参数：等级（配方负责填）

    virtual const char* kind() const = 0;   // 身份证：守卫转型用
};

/// 战士配置：特有参数 rage（怒气上限）
class WarriorConfig : public BuildConfig {
public:
    int rage = 100;                              // 特有参数 + 族默认值

    const char* kind() const override { return "warrior"; }
    static const char* static_kind() { return "warrior"; }
};

/// 法师配置：特有参数 element（元素，决定技能表）
class MageConfig : public BuildConfig {
public:
    std::string element = "Arcane";              // 特有参数 + 族默认值

    const char* kind() const override { return "mage"; }
    static const char* static_kind() { return "mage"; }
};

/**
 * @brief 守卫转型助手：从参数基类安全取回具体配置
 *
 * 裸 static_cast 的风险：builder × config 错配时行为未定义（可能
 * 读到垃圾值）。先对 kind() 暗号，错配立刻抛异常且信息明确。
 * （C++ 也可用 dynamic_cast：转型失败抛 std::bad_cast，同理）
 */
template <typename C>
static const C& config_of(const BuildConfig& cfg) {
    if (std::string(cfg.kind()) != C::static_kind()) {
        throw std::runtime_error(std::string("config mismatch: builder wants '") +
                                 C::static_kind() + "', got '" + cfg.kind() + "'");
    }
    return static_cast<const C&>(cfg);
}

// ============================================================================
// Product —— 产品（与 params.cpp 同款：公共字段 + 可空的特有字段）
// ============================================================================

class GameCharacter {
public:
    GameCharacter(const GameCharacter&) = default;
    GameCharacter& operator=(const GameCharacter&) = default;
    GameCharacter(GameCharacter&&) = default;
    GameCharacter& operator=(GameCharacter&&) = default;

    void print_sheet(std::ostream& os) const {
        os << "  ==== " << name_ << " " << title_ << " ====" << std::endl;
        os << "  Lv." << level_ << "  HP " << hp_ << "  ATK " << attack_
           << std::endl;
        if (rage_max_ > 0) {
            os << "  rage: " << rage_max_ << std::endl;
        }
        if (!element_.empty()) {
            os << "  element: " << element_ << std::endl;
        }
        os << "  skills: " << join(skills_) << std::endl;
    }

private:
    friend class CharacterBuilder;   // 基类：build 装配线 / deliver
    friend class WarriorBuilder;
    friend class MageBuilder;

    GameCharacter() = default;

    std::string name_;
    std::string title_;
    int level_ = 0;
    int hp_ = 0;
    int attack_ = 0;
    int rage_max_ = 0;
    std::string element_;
    std::vector<std::string> skills_;
};

// ============================================================================
// Builder —— 建造者：无状态装配线，吃参数基类
// ============================================================================

/**
 * @brief 角色建造者：build(const BuildConfig&) 是唯一的装配入口
 *
 * 注意建造者**自身没有任何配置状态**（对比 params.cpp 的 rage_）——
 * 配置全在调用方的 config 对象里。因此：
 *   - 同一个建造者实例可以交替/并发处理不同 config，互不干扰；
 *   - 步骤签名统一：都吃 const BuildConfig&（接口最窄形态）。
 *
 * 装配线（模板方法，Learn017 同款精神 —— 非虚骨架独裁）：
 *   复位毛坯 → 身份(公共实现) → 属性(纯虚) → 技能(纯虚) → 交付
 */
class CharacterBuilder {
public:
    virtual ~CharacterBuilder() = default;

    /// 模板方法（非虚）：一次调用完成整条装配线
    GameCharacter build(const BuildConfig& cfg) {
        product_ = GameCharacter{};          // 复位毛坯（配置不在这里，无配置可清）
        build_identity(cfg);                 // 公共参数：基类直接实现
        build_stats(cfg);                    // 特有逻辑：子类守卫转型取特有参数
        build_skills(cfg);                   // 同上
        return deliver();
    }

protected:
    virtual void build_stats(const BuildConfig& cfg) = 0;
    virtual void build_skills(const BuildConfig& cfg) = 0;

    GameCharacter product_;                  // 毛坯（build 每次开头复位）

private:
    /// 公共步骤只写一遍：身份装配与"表示"无关，子类根本不用管
    void build_identity(const BuildConfig& cfg) {
        product_.name_ = cfg.name;
        product_.title_ = cfg.title;
        product_.level_ = cfg.level;
    }

    GameCharacter deliver() {
        if (product_.name_.empty()) {
            throw std::runtime_error("deliver(): no name");
        }
        if (product_.hp_ <= 0) {
            throw std::runtime_error("deliver(): invalid stats");
        }
        if (product_.skills_.empty()) {
            throw std::runtime_error("deliver(): no skills");
        }
        GameCharacter out = std::move(product_);
        product_ = GameCharacter{};
        return out;
    }
};

// ============================================================================
// ConcreteBuilder —— 具体建造者：守卫转型取回特有参数
// ============================================================================

/**
 * @brief 战士建造者：公共参数直接用（cfg.level），特有参数转型后用（rage）
 */
class WarriorBuilder final : public CharacterBuilder {
protected:
    void build_stats(const BuildConfig& cfg) override {
        const WarriorConfig& wc = config_of<WarriorConfig>(cfg);   // 对暗号
        product_.hp_ = 100 + 8 * cfg.level;        // 公共参数：不用转型
        product_.attack_ = 10 + 3 * cfg.level + wc.rage / 50;      // 特有参数
        product_.rage_max_ = wc.rage;
    }

    void build_skills(const BuildConfig& /*cfg*/) override {
        product_.skills_ = { "Power Strike", "Shield Wall", "Bloodrage" };
    }
};

/**
 * @brief 法师建造者：element 决定技能表 —— 同一技能步骤、不同表示
 */
class MageBuilder final : public CharacterBuilder {
protected:
    void build_stats(const BuildConfig& cfg) override {
        config_of<MageConfig>(cfg);               // 对暗号（本步用不到特有参数）
        product_.hp_ = 60 + 5 * cfg.level;
        product_.attack_ = 5 + 2 * cfg.level;
    }

    void build_skills(const BuildConfig& cfg) override {
        const MageConfig& mc = config_of<MageConfig>(cfg);
        product_.element_ = mc.element;
        product_.skills_ = { mc.element + " Bolt", "Mana Shield" };
    }
};

// ============================================================================
// Director —— 配方只在参数基类层面工作，特有参数搭"顺风车"
// ============================================================================

class CharacterDirector {
public:
    /// 通用配方：吃 (任意建造者, 任意配置) —— 只填 BuildConfig 的公共参数；
    /// 派生 config 里已就位的特有参数（rage/element）原样通过
    GameCharacter construct_newbie(CharacterBuilder& b, BuildConfig& cfg,
                                   const std::string& name) {
        cfg.name = name;
        cfg.title = "the Rookie";
        cfg.level = 1;
        return b.build(cfg);
    }

    /// 通用配方 2：满级冠军
    GameCharacter construct_arena_champion(CharacterBuilder& b, BuildConfig& cfg,
                                           const std::string& name) {
        cfg.name = name;
        cfg.title = "the Arena Champion";
        cfg.level = 20;
        return b.build(cfg);
    }

    /// 类型特化配方：对**派生 config** 重载（写特有参数名正言顺，
    /// 且完全不碰建造者接口 —— 特化的是数据类，不是建造者类）
    GameCharacter construct_furious(CharacterBuilder& b, WarriorConfig& cfg,
                                    const std::string& name) {
        cfg.rage = 200;
        return construct_newbie(b, cfg, name);
    }
};

// ============================================================================
// 主函数 —— 参数对象流转、特化配方、错配守卫、方案对比
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Builder with a parameter BASE class:" << std::endl;
    std::cout << "  config object travels into build()" << std::endl;
    std::cout << "========================================" << std::endl;

    WarriorBuilder warrior_builder;
    MageBuilder mage_builder;
    CharacterDirector director;

    // ==== 演示 1：参数对象的流转 ====

    std::cout << std::endl;
    std::cout << "[1] The config object travels:" << std::endl;
    std::cout << "    caller: WarriorConfig{rage=60}   (derived, specific set)" << std::endl;
    std::cout << "      |" << std::endl;
    std::cout << "    director: fills name/title/level (BuildConfig level only)" << std::endl;
    std::cout << "      |" << std::endl;
    std::cout << "    builder.build(cfg): guards, downcasts, reads rage" << std::endl;

    // ==== 演示 2：基本用法 —— 参数一次成型，无需一串 setter ====

    std::cout << std::endl;
    std::cout << "[2] Basic use: config objects through generic recipe:" << std::endl;

    WarriorConfig wc;
    wc.rage = 60;                                  // 特有参数就位（公共参数配方来填）
    GameCharacter w1 = director.construct_newbie(warrior_builder, wc, "Kael");
    w1.print_sheet(std::cout);

    MageConfig mc;                                 // element 走族默认 Arcane
    GameCharacter m1 = director.construct_newbie(mage_builder, mc, "Elyra");
    m1.print_sheet(std::cout);
    std::cout << "    (one recipe, two config types riding along)" << std::endl;

    // ==== 演示 3：特化配方与冠军配方 ====

    std::cout << std::endl;
    std::cout << "[3] Specialized recipe (derived config) + champion:" << std::endl;

    WarriorConfig fc;
    GameCharacter w2 = director.construct_furious(warrior_builder, fc, "Grom");
    std::cout << "  construct_furious (rage=200):" << std::endl;
    w2.print_sheet(std::cout);

    WarriorConfig cc;
    cc.rage = 150;                                 // 调用方自带的特有参数
    GameCharacter w3 = director.construct_arena_champion(warrior_builder, cc, "Kael");
    std::cout << "  champion recipe (level 20, caller-set rage 150):" << std::endl;
    w3.print_sheet(std::cout);

    MageConfig fc2;
    fc2.element = "Frost";                         // 元素参数影响技能表
    GameCharacter m2 = director.construct_arena_champion(mage_builder, fc2, "Jaina");
    std::cout << "  champion recipe on mage (Frost preset):" << std::endl;
    m2.print_sheet(std::cout);

    // ==== 演示 4：错配守卫 —— 炸得早、信息清楚 ====

    std::cout << std::endl;
    std::cout << "[4] Mismatch guard (warrior builder x mage config):" << std::endl;

    MageConfig wrong;
    wrong.name = "Oops";
    try {
        GameCharacter bad = warrior_builder.build(wrong);
        bad.print_sheet(std::cout);                // 永远走不到这里
    } catch (const std::runtime_error& e) {
        std::cout << "    caught: " << e.what() << std::endl;
    }
    std::cout << "    -> bare static_cast would be undefined behavior;" << std::endl;
    std::cout << "       kind() guard turns it into a loud, early error" << std::endl;

    // ==== 演示 5：与 params.cpp 三方案的对比 ====

    std::cout << std::endl;
    std::cout << "[5] Four techniques compared:" << std::endl;
    std::cout << "    1 setter config   : b.set_rage(60) before recipe" << std::endl;
    std::cout << "        -> simplest; builder carries config state" << std::endl;
    std::cout << "    2 family layer    : MeleeBuilder base + set_rage" << std::endl;
    std::cout << "        -> grouped params, family recipes, compile-time" << std::endl;
    std::cout << "    3 generic recipe  : template + if constexpr detection" << std::endl;
    std::cout << "        -> one recipe adapts at compile time" << std::endl;
    std::cout << "    4 param BASE class: config_of<W>(cfg) downcast" << std::endl;
    std::cout << "        -> THIS file: config is a portable, copyable," << std::endl;
    std::cout << "           serializable OBJECT; builder stays stateless" << std::endl;

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. Common params live in the base config;" << std::endl;
    std::cout << "     specific params ride along in the derived one." << std::endl;
    std::cout << "  2. Recipes fill the base fields only -- they never" << std::endl;
    std::cout <<     "     need to know the derived config type." << std::endl;
    std::cout << "  3. Builders become stateless assembly lines:" << std::endl;
    std::cout << "     reusable, concurrency-friendly." << std::endl;
    std::cout << "  4. Config objects are copyable & serializable --" << std::endl;
    std::cout << "     save files / network sync / replay just work." << std::endl;
    std::cout << "  5. The downcast must be guarded (kind() tag or" << std::endl;
    std::cout << "     dynamic_cast); compile-time alternative: templates." << std::endl;

    return 0;
}
