/**
 * @file    main.cpp
 * @brief   工厂 × 建造者的组合 —— 创建型模式的"全家桶流水线"
 *
 * 动机（模式组合课）：
 *   Learn018（建造者）与 Learn019（工厂）各管一段：
 *     工厂回答"**选谁来装**"（类型名 → 建造者）；
 *     建造者回答"**怎么装**"（分步装配 + 交付校验）；
 *     抽象工厂回答"**部件从哪族来**"（装备套件配套供应）；
 *     导演回答"**按什么配方装**"（新手/冠军的步骤序列）。
 *   真实系统里它们几乎总是一起出现 —— 会单个模式只是入门，
 *   会**组合**模式才是日常。本程序把四种角色串成一条流水线：
 *
 *      调用方: forge.create("warrior", "champion", "Kael")
 *        │ 一步到位（工厂的门面）
 *        ▼
 *      CharacterForge（注册表工厂，前门）
 *        ├── 类型表: "warrior" → WarriorBuilder        ← 工厂选建造者
 *        │           "berserker" → 预配置(rage 200)      ← 工厂级参数化
 *        │           "frost-mage" → 预配置(Frost)       （Learn018 方案①
 *        │                                              的 setter 藏进注册
 *        │                                              lambda，Learn019 注册表）
 *        └── 配方表: "newbie" / "champion" → Recipe     ← 导演的配方
 *        ▼
 *      配方驱动建造者分步装配（Learn018 建造者 + Learn017 骨架）
 *        ├── build_identity  身份（公共）
 *        ├── build_stats     属性（读建造者配置 rage/element）
 *        ├── build_gear      装备 ←── WarriorKitFactory/MageKitFactory
 *        │                     （Learn019 抽象工厂：整族配套供应部件）
 *        ├── build_skills    技能
 *        └── deliver()       交付校验（半成品出不了厂）
 *        ▼
 *      GameCharacter（产品）
 *
 * 两种用法并存：
 *   [一步到位]  forge.create(type, recipe, name) —— 工厂门面，
 *               内部照样走完装配线（便利性与严谨性兼得）；
 *   [拆开用]    forge.builder_for(type) 拿到建造者，调用方自己分步
 *               驱动（灵活干预中间过程，Learn018 演示 [4] 的模式）。
 *
 * 与系列前作的关系：
 *   Learn015 Bridge / Learn016 Strategy / Learn017 Template Method /
 *   Learn018 Builder（含参数不一致四方案）/ Learn019 工厂三兄弟 ——
 *   本课是创建型两课的汇流，也是"模式组合"的第一课。
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <functional>   // std::function
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>    // std::runtime_error
#include <string>
#include <utility>      // std::move
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
// 部件层：装备族（Learn019 抽象工厂供部件，本文件只取名字）
// ============================================================================

/// 装备部件：统一的"有名之物"接口
class KitItem {
public:
    virtual ~KitItem() = default;
    virtual std::string name() const = 0;
};

class IronSword final : public KitItem {
public:
    std::string name() const override { return "Iron Sword"; }
};
class PlateArmor final : public KitItem {
public:
    std::string name() const override { return "Plate Armor"; }
};
class HealingPotion final : public KitItem {
public:
    std::string name() const override { return "Healing Potion"; }
};
class OakStaff final : public KitItem {
public:
    std::string name() const override { return "Oak Staff"; }
};
class ClothRobe final : public KitItem {
public:
    std::string name() const override { return "Cloth Robe"; }
};
class ManaPotion final : public KitItem {
public:
    std::string name() const override { return "Mana Potion"; }
};

/**
 * @brief 抽象工厂（Learn019）：一族配套装备的供应商
 *
 * 建造者的 build_gear() 步骤向它整族取货 —— 战士套件里
 * 绝不会混进法袍（族内配套由具体工厂保证）。
 */
class KitFactory {
public:
    virtual ~KitFactory() = default;
    virtual std::unique_ptr<KitItem> make_weapon() const = 0;
    virtual std::unique_ptr<KitItem> make_armor() const = 0;
    virtual std::unique_ptr<KitItem> make_potion() const = 0;
};

class WarriorKitFactory final : public KitFactory {
public:
    std::unique_ptr<KitItem> make_weapon() const override {
        return std::make_unique<IronSword>();
    }
    std::unique_ptr<KitItem> make_armor() const override {
        return std::make_unique<PlateArmor>();
    }
    std::unique_ptr<KitItem> make_potion() const override {
        return std::make_unique<HealingPotion>();
    }
};

class MageKitFactory final : public KitFactory {
public:
    std::unique_ptr<KitItem> make_weapon() const override {
        return std::make_unique<OakStaff>();
    }
    std::unique_ptr<KitItem> make_armor() const override {
        return std::make_unique<ClothRobe>();
    }
    std::unique_ptr<KitItem> make_potion() const override {
        return std::make_unique<ManaPotion>();
    }
};

// ============================================================================
// 产品（Learn018 同款：公共字段 + 可空特有字段）
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
        os << "  equipment: " << join(equipment_) << std::endl;
        os << "  skills:    " << join(skills_) << std::endl;
    }

private:
    friend class CharacterBuilder;
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
    std::vector<std::string> equipment_;
    std::vector<std::string> skills_;
};

// ============================================================================
// 建造者层（Learn018 params.cpp 结构：基类 + 族中间层 + 具体建造者）
// ============================================================================

/**
 * @brief 建造者基类：分步原语 + 非虚公共流程（reset/deliver/add_equipment）
 *
 * build_gear 的默认实现"向抽象工厂整族取货"—— 建造者与抽象工厂
 * 在这一步嵌套：装配线上的工位向配件厂下单。
 */
class CharacterBuilder {
public:
    virtual ~CharacterBuilder() = default;

    void reset() { product_ = GameCharacter{}; }   // 只清毛坯，不清配置

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
        reset();
        return out;
    }

    void add_equipment(const std::string& item) {
        product_.equipment_.push_back(item);
    }

    virtual void build_identity(const std::string& name,
                                const std::string& title) = 0;
    virtual void build_stats(int level) = 0;
    virtual void build_skills() = 0;

    /// 装备步骤：向"本族配件厂"整族取货（哪个厂由 kit() 决定）
    void build_gear() {
        const KitFactory& kit = kit_factory();
        product_.equipment_.push_back(kit.make_weapon()->name());
        product_.equipment_.push_back(kit.make_armor()->name());
        product_.equipment_.push_back(kit.make_potion()->name());
    }

protected:
    /// 本建造者配套的装备族供应商（族绑定，Learn019 抽象工厂）
    virtual const KitFactory& kit_factory() const = 0;

    GameCharacter product_;
};

/// 近战族中间层：特有参数 rage（Learn018 方案①/②）
class MeleeCharacterBuilder : public CharacterBuilder {
public:
    void set_rage(int rage) { rage_ = rage; }

protected:
    int rage_ = 100;
};

/// 法系族中间层：特有参数 element
class CasterCharacterBuilder : public CharacterBuilder {
public:
    void set_element(const std::string& e) { element_ = e; }

protected:
    std::string element_ = "Arcane";
};

/// 战士建造者：近战属性 + 战士装备族
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
        product_.attack_ = 10 + 3 * level + rage_ / 50;
        product_.rage_max_ = rage_;
    }
    void build_skills() override {
        product_.skills_ = { "Power Strike", "Shield Wall", "Bloodrage" };
    }

protected:
    const KitFactory& kit_factory() const override {
        static WarriorKitFactory kit;   // 族绑定的配件厂
        return kit;
    }
};

/// 法师建造者：法系属性 + 法师装备族；element 决定技能表
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
        product_.element_ = element_;
        product_.skills_ = { element_ + " Bolt", "Mana Shield" };
    }

protected:
    const KitFactory& kit_factory() const override {
        static MageKitFactory kit;
        return kit;
    }
};

// ============================================================================
// 导演层（Learn018 Director：配方）
// ============================================================================

class CharacterDirector {
public:
    GameCharacter construct_newbie(CharacterBuilder& b,
                                   const std::string& name) {
        b.reset();
        b.build_identity(name, "the Rookie");
        b.build_stats(/*level=*/1);
        b.build_gear();
        b.build_skills();
        return b.deliver();
    }

    GameCharacter construct_arena_champion(CharacterBuilder& b,
                                           const std::string& name) {
        b.reset();
        b.build_identity(name, "the Arena Champion");
        b.build_stats(/*level=*/20);
        b.build_gear();
        b.build_skills();
        b.add_equipment("Champion's Cape");    // 配方追加的公共件
        b.add_equipment("Ring of Victory");
        return b.deliver();
    }
};

// ============================================================================
// 工厂层（Learn019 注册表工厂）：前门 —— 选建造者 + 选配方
// ============================================================================

/// 建造者创建函数（注册 lambda 里可完成"预配置"—— 工厂级参数化）
using BuilderCreator = std::function<std::unique_ptr<CharacterBuilder>()>;

/// 配方：驱动建造者完成一次装配并交付
using Recipe = std::function<GameCharacter(CharacterBuilder&,
                                           const std::string&)>;

/**
 * @brief 角色锻造所：把"类型表 + 配方表"拼成一步到位的门面
 *
 * 这就是工厂与建造者的合体：
 *   - 对外（create）：工厂的体验 —— 一个调用拿成品；
 *   - 对内：注册表选**建造者**（而非直接选产品），由配方驱动
 *     分步装配、deliver 校验 —— 建造者的严谨一点没少。
 * 同一注册表还能输出**预配置变体**（berserker / frost-mage）：
 * Learn018 讨论的"差异参数"，在工厂层用注册 lambda 一并解决。
 */
class CharacterForge {
public:
    void register_builder(const std::string& type, BuilderCreator creator) {
        builders_[type] = std::move(creator);
    }

    void register_recipe(const std::string& name, Recipe recipe) {
        recipes_[name] = std::move(recipe);
    }

    /// 一步到位：查类型表拿建造者 -> 查配方表跑装配 -> 交付成品
    ///
    /// 注意：建造者的 unique_ptr 必须先存进局部变量保活 ——
    /// 若写成 `CharacterBuilder& b = *builder_for(type);`，
    /// unique_ptr 临时对象在语句末就被销毁，后面配方拿到的是
    /// 悬垂引用（IDE 的 -Wdangling-gsl 会正确报警）
    GameCharacter create(const std::string& type, const std::string& recipe,
                         const std::string& char_name) {
        std::unique_ptr<CharacterBuilder> builder = builder_for(type);   // 选谁装
        auto it = recipes_.find(recipe);
        if (it == recipes_.end()) {
            throw std::runtime_error("forge: unknown recipe '" + recipe + "'");
        }
        return it->second(*builder, char_name);          // 怎么装
    }

    /// 拆开用：只拿建造者，调用方自己分步驱动（灵活模式）
    std::unique_ptr<CharacterBuilder> builder_for(const std::string& type) {
        auto it = builders_.find(type);
        if (it == builders_.end()) {
            throw std::runtime_error("forge: unknown type '" + type + "'");
        }
        return it->second();
    }

private:
    std::map<std::string, BuilderCreator> builders_;   // 类型名 → 建造者
    std::map<std::string, Recipe> recipes_;            // 配方名 → 装配流程
};

// ============================================================================
// 主函数 —— 装配流水线、预配置变体、拆开用、错误处理
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Factory x Builder: one assembly line," << std::endl;
    std::cout << "  four patterns each doing its own job" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 装配流水线：注册类型表与配方表 ====

    CharacterForge forge;
    CharacterDirector director;

    // 类型表：基础类型（Learn018 方案①的差异参数走默认值）
    forge.register_builder("warrior", [] {
        return std::make_unique<WarriorBuilder>();
    });
    forge.register_builder("mage", [] {
        return std::make_unique<MageBuilder>();
    });
    // 预配置变体：差异参数（rage/element）藏在注册 lambda 里 ——
    // Learn018 params.cpp 的 set_rage/set_element 有了工厂级的家
    forge.register_builder("berserker", [] {
        auto b = std::make_unique<WarriorBuilder>();
        b->set_rage(200);                       // 工厂级参数化
        return b;
    });
    forge.register_builder("frost-mage", [] {
        auto b = std::make_unique<MageBuilder>();
        b->set_element("Frost");
        return b;
    });

    // 配方表：导演的配方注册进来
    forge.register_recipe("newbie",
        [&director](CharacterBuilder& b, const std::string& n) {
            return director.construct_newbie(b, n);
        });
    forge.register_recipe("champion",
        [&director](CharacterBuilder& b, const std::string& n) {
            return director.construct_arena_champion(b, n);
        });

    // ==== 演示 1：流水线分工图 ====

    std::cout << std::endl;
    std::cout << "[1] The assembly line (who does what):" << std::endl;
    std::cout << "    forge.create(type, recipe, name)   <- factory facade" << std::endl;
    std::cout << "      type table  : picks a BUILDER      (registry factory)" << std::endl;
    std::cout << "      recipe table: drives the steps     (director)" << std::endl;
    std::cout << "      build_gear  : orders a gear FAMILY  (abstract factory)" << std::endl;
    std::cout << "      deliver()   : invariant exit gate   (builder)" << std::endl;

    // ==== 演示 2：一步到位（工厂门面）====

    std::cout << std::endl;
    std::cout << "[2] One-stop creation (factory experience):" << std::endl;

    GameCharacter w1 = forge.create("warrior", "newbie", "Kael");
    std::cout << "  forge.create(\"warrior\", \"newbie\", \"Kael\"):" << std::endl;
    w1.print_sheet(std::cout);

    GameCharacter w2 = forge.create("warrior", "champion", "Kael");
    std::cout << "  forge.create(\"warrior\", \"champion\", \"Kael\"):" << std::endl;
    w2.print_sheet(std::cout);

    GameCharacter m1 = forge.create("mage", "newbie", "Elyra");
    std::cout << "  forge.create(\"mage\", \"newbie\", \"Elyra\"):" << std::endl;
    m1.print_sheet(std::cout);
    std::cout << "    (one call per character; inside, the full" << std::endl;
    std::cout <<     "     assembly line still runs step by step)" << std::endl;

    // ==== 演示 3：预配置变体（工厂级参数化）====

    std::cout << std::endl;
    std::cout << "[3] Preset variants registered in the factory:" << std::endl;

    GameCharacter b1 = forge.create("berserker", "champion", "Grom");
    std::cout << "  forge.create(\"berserker\", \"champion\", \"Grom\"):" << std::endl;
    b1.print_sheet(std::cout);

    GameCharacter f1 = forge.create("frost-mage", "newbie", "Jaina");
    std::cout << "  forge.create(\"frost-mage\", \"newbie\", \"Jaina\"):" << std::endl;
    f1.print_sheet(std::cout);
    std::cout << "    (rage/element presets live in the register" << std::endl;
    std::cout <<     "     lambdas -- Learn018's setters found a home)" << std::endl;

    // ==== 演示 4：拆开用（建造者的灵活体验）====

    std::cout << std::endl;
    std::cout << "[4] Take-apart mode: get the builder, drive manually:" << std::endl;

    auto builder = forge.builder_for("warrior");   // 只拿建造者
    builder->reset();
    builder->build_identity("Sylra", "the Handmade");
    builder->build_stats(/*level=*/7);
    builder->build_gear();
    builder->build_skills();
    builder->add_equipment("Lucky Coin");          // 中途插一件私货
    GameCharacter custom = builder->deliver();
    custom.print_sheet(std::cout);
    std::cout << "    (caller drives steps, intervenes mid-process;" << std::endl;
    std::cout <<     "     type-specific presets go through registered" << std::endl;
    std::cout <<     "     variants like \"berserker\")" << std::endl;

    // ==== 演示 5：错误处理（两张表各守各的门）====

    std::cout << std::endl;
    std::cout << "[5] Both tables guard their own gate:" << std::endl;

    try {
        forge.create("paladin", "newbie", "Nobody");      // 未注册类型
    } catch (const std::runtime_error& e) {
        std::cout << "  caught: " << e.what() << std::endl;
    }
    try {
        forge.create("warrior", "raid-boss", "Nobody");   // 未注册配方
    } catch (const std::runtime_error& e) {
        std::cout << "  caught: " << e.what() << std::endl;
    }

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. Factory picks WHO assembles; builder" << std::endl;
    std::cout <<     "     decides HOW to assemble." << std::endl;
    std::cout << "  2. Facade = factory experience (create);" << std::endl;
    std::cout << "     internals = builder rigor (steps + gate)." << std::endl;
    std::cout << "  3. build_gear nests an abstract factory:" << std::endl;
    std::cout <<     "     a work station ordering a gear FAMILY." << std::endl;
    std::cout << "  4. Registry variants (berserker/frost-mage)" << std::endl;
    std::cout <<     "     give type-specific params a factory home." << std::endl;
    std::cout << "  5. Both modes coexist: one-stop create() and" << std::endl;
    std::cout <<     "     take-apart builder_for() for custom work." << std::endl;

    return 0;
}
