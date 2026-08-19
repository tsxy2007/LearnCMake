/**
 * @file    params.cpp
 * @brief   建造者模式进阶 —— 每种类型的参数不一致时怎么设计（Learn018 第二示例）
 *
 * 问题背景：
 *   GoF 建造者的 Director 只依赖抽象接口 —— 前提是所有表示共用同一套
 *   步骤签名。现实里各类型常有**自己特有的参数**：
 *     战士建造者需要"怒气上限 rage"（影响攻击与专属技能）；
 *     法师建造者需要"元素 element"（决定技能表：Fire Bolt / Frost Bolt）。
 *   这些参数没法放进公共接口 —— 放谁那里都别扭。
 *
 *   ✘ 反模式：参数并集接口（union interface）
 *
 *     class CharacterBuilder {
 *         virtual void set_rage(int) = 0;           // 法师：只能空实现
 *         virtual void set_element(string) = 0;    // 战士：只能空实现
 *         virtual void build_stats(int) = 0;       // 真正公共的只有这类
 *     };
 *
 *   三个坏处：
 *     1) 接口膨胀：加一种类型就加一组方法，基类无限变胖；
 *     2) 违反 LSP（里氏替换）：一半实现是"收到调用但假装没发生"；
 *     3) Director 配方里充斥无意义调用，或被迫 dynamic_cast 猜类型。
 *
 * 本程序演示三种正解：
 *
 *   ✔ 方案 1 —— 配置与毛坯分离（差异参数 = 具体建造者自有 setter）
 *     公共接口保持窄；rage/element 是**建造者的配置成员**，
 *     build_stats()/build_skills() 读配置算产品。
 *     关键细节：reset() 只清"毛坯产品"，**不清配置** ——
 *     配置一次、多次出厂；Director 的 reset 不会抹掉调用方的预设。
 *
 *   ✔ 方案 2 —— 接口分层（族参数放进中间层）
 *     CharacterBuilder（公共步骤）
 *       ├── MeleeCharacterBuilder：+ set_rage()     ← 近战族的差异参数
 *       │     └── WarriorBuilder（最终表示）
 *       └── CasterCharacterBuilder：+ set_element() ← 法系族的差异参数
 *             └── MageBuilder（最终表示）
 *     Director 的配方也可以分层：通用配方吃 CharacterBuilder&，
 *     族特化配方（construct_furious）吃 MeleeCharacterBuilder& ——
 *     重载决议自动挑配方，传错族直接编译失败（编译期类型安全）。
 *
 *   ✔ 方案 3 —— 泛型配方 + 能力检测（模板把 Director 编译期化）
 *     construct_custom<B>(B& b) 是函数模板；
 *     用 void_t 探测 B 是否有 set_rage，if constexpr 按能力注入 ——
 *     一份配方自适应任意建造者（C++20 concepts 可写得更直白：
 *     requires b.set_rage(0)）。
 *
 * 与 main.cpp（标准四角色）的关系：本文件回答它的遗留问题 ——
 * "步骤签名不一致时，'同样的构建过程'如何继续成立"。
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <iostream>
#include <stdexcept>    // std::runtime_error
#include <string>
#include <type_traits>  // void_t, false_type, true_type
#include <utility>      // std::move, std::declval
#include <vector>

#ifdef _WIN32
#include <windows.h>    // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
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
// Product —— 产品：公共字段 + 两类"特有字段"（可空）
// ============================================================================

/**
 * @brief 角色：公共字段（名字/等级/血/攻/技能）+ 特有字段（怒气或元素）
 *
 * 特有字段用"可空"表示（rage_max_ == 0 / element_ 为空 = 无此项），
 * 属性卡按需打印 —— 产品模式允许各表示携带不同的附加属性。
 */
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
        if (rage_max_ > 0) {                       // 近战特有：按需显示
            os << "  rage: " << rage_max_ << std::endl;
        }
        if (!element_.empty()) {                   // 法系特有：按需显示
            os << "  element: " << element_ << std::endl;
        }
        os << "  skills: " << join(skills_) << std::endl;
    }

private:
    friend class CharacterBuilder;   // 基类：reset / deliver
    friend class WarriorBuilder;     // 具体建造者直接装配毛坯
    friend class MageBuilder;

    GameCharacter() = default;       // 私有：产品只能从建造者"出生"

    std::string name_;
    std::string title_;
    int level_ = 0;
    int hp_ = 0;
    int attack_ = 0;
    int rage_max_ = 0;               // 近战特有（0 = 无）
    std::string element_;            // 法系特有（空 = 无）
    std::vector<std::string> skills_;
};

// ============================================================================
// Builder —— 抽象基类：只保留真正公共的步骤（窄接口）
// ============================================================================

/**
 * @brief 公共接口刻意收窄：身份/属性/技能三步 + 交付
 *
 * 注意这里**没有** set_rage / set_element —— 差异参数不属于公共接口
 * （并集接口反模式见文件头注释）。它们放在族中间层（方案 2）
 * 或具体建造者（方案 1）。
 */
class CharacterBuilder {
public:
    virtual ~CharacterBuilder() = default;

    /// 复位：只清**毛坯产品**，不清建造者的配置（方案 1 的关键）
    void reset() { product_ = GameCharacter{}; }

    /// 交付：公共不变式统一把关（与 main.cpp 同款）
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
        reset();                     // 毛坯复位；配置原封不动
        return out;
    }

    virtual void build_identity(const std::string& name,
                                const std::string& title) = 0;
    virtual void build_stats(int level) = 0;
    virtual void build_skills() = 0;

protected:
    GameCharacter product_;          // 毛坯（每次交付后清空）
};

// ============================================================================
// 方案 2 的主角：族中间层 —— 同族共享的差异参数放这里
// ============================================================================

/**
 * @brief 近战族中间层：怒气是本族每个建造者都有的差异参数
 *
 * rage_ 是**建造者配置**（不是毛坯字段）：默认 100，
 * reset() 不清它，build_stats() 读它。
 */
class MeleeCharacterBuilder : public CharacterBuilder {
public:
    void set_rage(int rage) { rage_ = rage; }      // 配置一次
    int rage() const { return rage_; }

protected:
    int rage_ = 100;                               // 族默认配置
};

/**
 * @brief 法系族中间层：元素是本族的差异参数（决定技能表内容）
 */
class CasterCharacterBuilder : public CharacterBuilder {
public:
    void set_element(const std::string& e) { element_ = e; }
    const std::string& element() const { return element_; }

protected:
    std::string element_ = "Arcane";               // 族默认配置
};

// ============================================================================
// ConcreteBuilder —— 最终表示：公共步骤读"族配置"产出差异化产品
// ============================================================================

/**
 * @brief 战士：怒气抬高攻击、写进产品；技能表带本族专属
 */
class WarriorBuilder final : public MeleeCharacterBuilder {
public:
    void build_identity(const std::string& name,
                        const std::string& title) override {
        product_.name_ = name;
        product_.title_ = title;
    }

    void build_stats(int level) override {
        product_.level_ = level;
        product_.hp_ = 100 + 8 * level;
        product_.attack_ = 10 + 3 * level + rage_ / 50;   // 怒气反哺攻击
        product_.rage_max_ = rage_;                       // 配置写进产品
    }

    void build_skills() override {
        product_.skills_ = { "Power Strike", "Shield Wall", "Bloodrage" };
    }
};

/**
 * @brief 法师：元素决定技能表 —— 同一个 build_skills 步骤，
 *        配置不同则"表示"不同
 */
class MageBuilder final : public CasterCharacterBuilder {
public:
    void build_identity(const std::string& name,
                        const std::string& title) override {
        product_.name_ = name;
        product_.title_ = title;
    }

    void build_stats(int level) override {
        product_.level_ = level;
        product_.hp_ = 60 + 5 * level;
        product_.attack_ = 5 + 2 * level;
    }

    void build_skills() override {
        product_.element_ = element_;                     // 配置写进产品
        product_.skills_ = { element_ + " Bolt", "Mana Shield" };
    }
};

// ============================================================================
// Director —— 配方也分层：通用配方 + 族特化配方（重载决议自动挑）
// ============================================================================

class CharacterDirector {
public:
    /// 通用配方：只依赖公共基类，任何建造者都能跑
    GameCharacter construct_newbie(CharacterBuilder& b,
                                   const std::string& name) {
        b.reset();                       // 清毛坯 —— 配置保留！
        b.build_identity(name, "the Rookie");
        b.build_stats(/*level=*/1);
        b.build_skills();
        return b.deliver();
    }

    /// 族特化配方：只吃近战族 —— 能名正言顺调 set_rage；
    /// 传法师进来？MeleeCharacterBuilder& 接不住，直接编译失败
    ///（差异参数的错误用法在编译期就被拦截）
    GameCharacter construct_furious(MeleeCharacterBuilder& b,
                                    const std::string& name) {
        b.set_rage(200);                 // 配方级别覆盖怒气
        return construct_newbie(b, name);
    }
};

// ============================================================================
// 方案 3 的主角：泛型配方 + 能力检测（编译期自适应差异参数）
// ============================================================================

/// 探测 B 是否提供 set_rage(int)（C++17 检测惯用法，C++20 可用 concepts）
template <typename B, typename = void>
struct has_set_rage : std::false_type {};

template <typename B>
struct has_set_rage<B, std::void_t<decltype(std::declval<B&>().set_rage(0))>>
    : std::true_type {};

/**
 * @brief 泛型配方：一份函数模板适配任意建造者
 *
 * if constexpr 在**编译期**按能力裁剪：有 set_rage 的（近战族）自动
 * 注入怒气配置；没有的（法系族）该分支根本不参与编译。
 * 相比方案 2 的"族配方重载"，这里连配方都只写一份。
 */
template <typename B>
GameCharacter construct_custom(B& b, const std::string& name, int level) {
    b.reset();
    if constexpr (has_set_rage<B>::value) {
        b.set_rage(180);                 // 编译期探测到的差异参数
    }
    b.build_identity(name, "the Custom");
    b.build_stats(level);
    b.build_skills();
    return b.deliver();
}

// ============================================================================
// 主函数 —— 反模式、配置分离、族配方、泛型配方
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Builder with heterogeneous params:" << std::endl;
    std::cout << "  warrior needs rage, mage needs element" << std::endl;
    std::cout << "========================================" << std::endl;

    WarriorBuilder warrior_builder;
    MageBuilder mage_builder;
    CharacterDirector director;

    // ==== 演示 1：反模式 —— 参数并集接口 ====

    std::cout << std::endl;
    std::cout << "[1] Antipattern: the union interface" << std::endl;
    std::cout << "    class CharacterBuilder {" << std::endl;
    std::cout << "        virtual void set_rage(int) = 0;        // mage: ignore" << std::endl;
    std::cout << "        virtual void set_element(string) = 0; // warrior: ignore" << std::endl;
    std::cout << "    };" << std::endl;
    std::cout << "    -> interface bloat + LSP violations +" << std::endl;
    std::cout << "       recipes full of meaningless calls" << std::endl;

    // ==== 演示 2：方案 1 —— 配置与毛坯分离 ====

    std::cout << std::endl;
    std::cout << "[2] Config vs blank: rage is builder CONFIG," << std::endl;
    std::cout << "    reset() clears only the blank product:" << std::endl;

    GameCharacter w0 = director.construct_newbie(warrior_builder, "Kael");
    std::cout << "  default rage (100):" << std::endl;
    w0.print_sheet(std::cout);

    warrior_builder.set_rage(60);        // 配置一次
    GameCharacter w1 = director.construct_newbie(warrior_builder, "Kael");
    GameCharacter w2 = director.construct_newbie(warrior_builder, "Kael");
    std::cout << "  after set_rage(60), built twice:" << std::endl;
    w1.print_sheet(std::cout);
    w2.print_sheet(std::cout);
    std::cout << "    (both runs see rage 60 -- recipe's reset() never" << std::endl;
    std::cout << "     touches builder config)" << std::endl;

    // ==== 演示 3：方案 2 —— 接口分层与族配方 ====

    std::cout << std::endl;
    std::cout << "[3] Layered interfaces: family recipes" << std::endl;

    GameCharacter m1 = director.construct_newbie(mage_builder, "Elyra");
    std::cout << "  generic recipe on mage (default element):" << std::endl;
    m1.print_sheet(std::cout);

    GameCharacter w3 = director.construct_furious(warrior_builder, "Grom");
    std::cout << "  melee-only recipe (construct_furious, rage 200):" << std::endl;
    w3.print_sheet(std::cout);
    std::cout << "    (passing the mage builder to construct_furious" << std::endl;
    std::cout << "     would not compile -- wrong family, caught at" << std::endl;
    std::cout << "     compile time)" << std::endl;

    // ==== 演示 4：方案 3 —— 泛型配方 + 能力检测 ====

    std::cout << std::endl;
    std::cout << "[4] Generic recipe with capability detection:" << std::endl;

    GameCharacter w4 = construct_custom(warrior_builder, "Thrall", 5);
    std::cout << "  custom recipe on warrior (auto-set rage 180):" << std::endl;
    w4.print_sheet(std::cout);

    mage_builder.set_element("Frost");   // 法系的差异参数：调用方预配置
    GameCharacter m2 = construct_custom(mage_builder, "Jaina", 5);
    std::cout << "  same recipe on mage (no rage branch; Frost preset):" << std::endl;
    m2.print_sheet(std::cout);
    std::cout << "    (if constexpr picked the rage branch for the" << std::endl;
    std::cout <<     "     warrior only, at compile time)" << std::endl;

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. Never union all params into the base" << std::endl;
    std::cout << "     interface -- it bloats and breaks LSP." << std::endl;
    std::cout << "  2. Type-specific params = builder CONFIG:" << std::endl;
    std::cout << "     reset() clears the blank, keeps the config." << std::endl;
    std::cout << "  3. Grouped params -> family layer with its own" << std::endl;
    std::cout << "     setters; family recipes take the middle" << std::endl;
    std::cout <<     "     type (compile-time safety)." << std::endl;
    std::cout << "  4. One recipe for all -> template + if constexpr" << std::endl;
    std::cout << "     capability detection (concepts in C++20)." << std::endl;

    return 0;
}
