/**
 * @file    main.cpp
 * @brief   工厂模式三兄弟 —— 简单工厂 / 工厂方法 / 抽象工厂 + 注册表工厂
 *
 * 总纲（创建型模式的分工，承接 Learn018 的对比表）：
 *   把"new 哪个具体类"这件事封装起来，让使用方只依赖抽象产品。
 *   工厂模式家族回答三种粒度的"谁来 new"：
 *
 *   1) 简单工厂（Simple Factory，非 GoF）
 *        一个函数 + switch 集中所有 new。
 *        优点：最简单。缺点：新增产品要改 switch —— 违反开闭原则
 *        （和 Learn016 策略模式里 solve() 的 switch 是同一个坏味道）。
 *
 *   2) 工厂方法（Factory Method，GoF 创建型）
 *        定义创建对象的接口，把"new 谁"延迟到**子类**。
 *        一产品一工厂：新增产品 = 新增产品类 + 工厂类，旧代码零修改。
 *        代价：类的数量翻倍。
 *
 *   3) 抽象工厂（Abstract Factory，GoF 创建型）
 *        不再生产单个产品，而是生产**一族**配套产品（本例：
 *        战士套件 = 铁剑 + 板甲 + 治疗药水；法师套件 = 木杖 + 长袍 + 法力药水）。
 *        同族配套由同一个具体工厂保证 —— 战士套件里绝不会混进法杖。
 *
 *   4) 注册表工厂（现代实用形态）
 *        map<名字, 创建函数>：把 switch 换成查表，注册可发生在
 *        **运行期**（插件生态的标准做法）。
 *
 * 本例域：游戏武器/装备（延续 Learn017 武器、Learn018 角色装备）。
 *
 * 与系列前作的关系：
 *   - Learn016 Strategy：策略是"行为随时间切换"；工厂是"对象创建一次"。
 *     注册表工厂和策略注册表长得一样 —— 因为它们共享同一机制
 *    （名字→可调用对象），意图不同。
 *   - Learn017 Template Method：工厂方法的 Creator 里通常还有
 *     使用产品的公共流程（本例 create_and_test）—— 它就是模板方法。
 *   - Learn018 Builder：工厂一步交付简单对象；建造者分步组装复杂对象
 *     （对比表见 Learn018 readme 第 5 节）。
 *   - Learn015 Bridge 的 Renderer 族、Learn016 的 Solver 族，
 *     都可以用本课的注册表工厂统一创建。
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <functional>   // std::function（注册表工厂）
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>   // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

// ============================================================================
// 产品层次 —— 抽象产品 Weapon + 四个具体产品
// ============================================================================

/**
 * @brief 抽象产品：使用方只依赖这个接口，永远不写 new Sword()
 */
class Weapon {
public:
    virtual ~Weapon() = default;             // 虚析构：经基类指针删除派生对象
    virtual std::string name() const = 0;
    virtual int damage() const = 0;
    virtual void attack() const = 0;         // 打印一次攻击动作
};

class Sword final : public Weapon {
public:
    std::string name() const override { return "Sword"; }
    int damage() const override { return 8; }
    void attack() const override {
        std::cout << "    [Sword] slashes! (damage " << damage() << ")"
                  << std::endl;
    }
};

class Bow final : public Weapon {
public:
    std::string name() const override { return "Bow"; }
    int damage() const override { return 6; }
    void attack() const override {
        std::cout << "    [Bow] shoots an arrow! (damage " << damage() << ")"
                  << std::endl;
    }
};

class MagicWand final : public Weapon {
public:
    std::string name() const override { return "MagicWand"; }
    int damage() const override { return 5; }
    void attack() const override {
        std::cout << "    [MagicWand] casts a fireball! (damage "
                  << damage() << ")" << std::endl;
    }
};

/// 斧头：本课的"后来新增者"—— 专门用来验证各种工厂的扩展成本
class Axe final : public Weapon {
public:
    std::string name() const override { return "Axe"; }
    int damage() const override { return 10; }
    void attack() const override {
        std::cout << "    [Axe] cleaves! (damage " << damage() << ")"
                  << std::endl;
    }
};

// ============================================================================
// 其一：简单工厂 —— switch 集中所有 new（OCP 的反面教材）
// ============================================================================

enum class WeaponType { Sword, Bow, MagicWand };

/**
 * @brief 简单工厂：一个静态函数 + switch
 *
 * 收益：new 的判断集中一处，调用方摆脱条件逻辑。
 * 代价：新增产品（比如 Axe）必须回来改 switch —— 修改了已测试的
 * 旧代码，违反开闭原则。产品种类少且稳定时，这个代价可以接受；
 * 种类会增长时，请升级到工厂方法或注册表（见后）。
 */
class SimpleWeaponFactory {
public:
    static std::unique_ptr<Weapon> create(WeaponType type) {
        switch (type) {   // ← 所有 new 集中于此，所有修改也集中于此
        case WeaponType::Sword:
            return std::make_unique<Sword>();
        case WeaponType::Bow:
            return std::make_unique<Bow>();
        case WeaponType::MagicWand:
            return std::make_unique<MagicWand>();
        }
        throw std::runtime_error("simple factory: unknown weapon type");
    }
};

// ============================================================================
// 其二：工厂方法（GoF）—— 把 create 本身声明为虚函数
// ============================================================================

/**
 * @brief Creator（创建者抽象）：只约束"能造武器"，不限定造哪种
 *
 * create_and_test() 是 Creator 的公共流程：它**使用**工厂方法造出的
 * 产品（顺手试一击）—— 这正是 Learn017 模板方法的结构：
 * 非虚骨架调用虚的原语（create_weapon）。GoF 原书的 Creator
 * 通常还承载这类业务流程，工厂方法只是其中的创建步骤。
 */
class WeaponFactory {
public:
    virtual ~WeaponFactory() = default;

    /// 公共流程（非虚）：造出来 -> 试一击 -> 交付
    std::unique_ptr<Weapon> create_and_test() const {
        auto weapon = create_weapon();   // ★ 工厂方法：new 谁，子类说了算
        weapon->attack();
        return weapon;
    }

    /// 工厂方法（纯虚）：创建产品的抽象声明
    virtual std::unique_ptr<Weapon> create_weapon() const = 0;

    virtual std::string name() const = 0;
};

class SwordFactory final : public WeaponFactory {
public:
    std::unique_ptr<Weapon> create_weapon() const override {
        return std::make_unique<Sword>();
    }
    std::string name() const override { return "SwordFactory"; }
};

class BowFactory final : public WeaponFactory {
public:
    std::unique_ptr<Weapon> create_weapon() const override {
        return std::make_unique<Bow>();
    }
    std::string name() const override { return "BowFactory"; }
};

class MagicWandFactory final : public WeaponFactory {
public:
    std::unique_ptr<Weapon> create_weapon() const override {
        return std::make_unique<MagicWand>();
    }
    std::string name() const override { return "MagicWandFactory"; }
};

/**
 * @brief 斧头工厂：**后来新增**的工厂 —— 对照点
 *
 * 新增 Axe 产品：加 Axe + AxeFactory 两个类，一行旧代码不碰
 * （对比简单工厂必须改 switch）。这就是工厂方法对 OCP 的兑现，
 * 代价是产品与工厂类数量 2:1 膨胀。
 */
class AxeFactory final : public WeaponFactory {
public:
    std::unique_ptr<Weapon> create_weapon() const override {
        return std::make_unique<Axe>();
    }
    std::string name() const override { return "AxeFactory"; }
};

// ============================================================================
// 其三：抽象工厂（GoF）—— 一次生产一族配套产品
// ============================================================================

/// 产品 B：护甲
class Armor {
public:
    virtual ~Armor() = default;
    virtual std::string name() const = 0;
    virtual void defend() const = 0;
};

class PlateArmor final : public Armor {
public:
    std::string name() const override { return "Plate Armor"; }
    void defend() const override {
        std::cout << "    [Plate Armor] blocks 6 damage" << std::endl;
    }
};

class ClothRobe final : public Armor {
public:
    std::string name() const override { return "Cloth Robe"; }
    void defend() const override {
        std::cout << "    [Cloth Robe] blocks 2 damage" << std::endl;
    }
};

/// 产品 C：药水
class Potion {
public:
    virtual ~Potion() = default;
    virtual std::string name() const = 0;
    virtual void drink() const = 0;
};

class HealingPotion final : public Potion {
public:
    std::string name() const override { return "Healing Potion"; }
    void drink() const override {
        std::cout << "    [Healing Potion] restores 30 hp" << std::endl;
    }
};

class ManaPotion final : public Potion {
public:
    std::string name() const override { return "Mana Potion"; }
    void drink() const override {
        std::cout << "    [Mana Potion] restores 25 mp" << std::endl;
    }
};

/**
 * @brief 抽象工厂：声明**一族**产品的创建接口
 *
 * 关键词是"族"（family）：三个 make_* 必须产出互相配套的产品。
 * 具体工厂绑定一种风格（近战 / 法系），风格切换 = 换整个工厂。
 */
class KitFactory {
public:
    virtual ~KitFactory() = default;

    virtual std::string kit_name() const = 0;
    virtual std::unique_ptr<Weapon> make_weapon() const = 0;
    virtual std::unique_ptr<Armor> make_armor() const = 0;
    virtual std::unique_ptr<Potion> make_potion() const = 0;
};

/// 具体工厂 A：战士套件（近战全家桶）
class WarriorKitFactory final : public KitFactory {
public:
    std::string kit_name() const override { return "Warrior Kit"; }
    std::unique_ptr<Weapon> make_weapon() const override {
        return std::make_unique<Sword>();
    }
    std::unique_ptr<Armor> make_armor() const override {
        return std::make_unique<PlateArmor>();
    }
    std::unique_ptr<Potion> make_potion() const override {
        return std::make_unique<HealingPotion>();
    }
};

/// 具体工厂 B：法师套件（法系全家桶）—— 同族配套由本类保证
class MageKitFactory final : public KitFactory {
public:
    std::string kit_name() const override { return "Mage Kit"; }
    std::unique_ptr<Weapon> make_weapon() const override {
        return std::make_unique<MagicWand>();
    }
    std::unique_ptr<Armor> make_armor() const override {
        return std::make_unique<ClothRobe>();
    }
    std::unique_ptr<Potion> make_potion() const override {
        return std::make_unique<ManaPotion>();
    }
};

// ============================================================================
// 其四：注册表工厂 —— switch 的查表替代品，注册开放到运行期
// ============================================================================

/// 创建函数的类型：无参可调用，返回抽象产品指针
using WeaponCreator = std::function<std::unique_ptr<Weapon>()>;

/**
 * @brief 名字 -> 创建函数 的注册表
 *
 * 相比简单工厂的 switch：
 *   - 新增产品 = 注册一个 lambda，**不修改任何既有代码**（OCP 兑现，
 *     且不需要"一产品一工厂类"的类膨胀 —— 工厂方法的两头各取一半）；
 *   - 注册可发生在运行期（动态库加载、插件初始化、脚本配置）——
 *     插件生态的标准姿势。
 * 工业实现通常再配一个全局单例注册表 + 各产品在自己 .cpp 里
 * 的静态注册对象（自注册惯用法）。
 */
static std::map<std::string, WeaponCreator> make_weapon_registry() {
    std::map<std::string, WeaponCreator> registry;
    registry["sword"] = [] { return std::make_unique<Sword>(); };
    registry["bow"] = [] { return std::make_unique<Bow>(); };
    registry["wand"] = [] { return std::make_unique<MagicWand>(); };
    registry["axe"] = [] { return std::make_unique<Axe>(); };   // 注册即生效
    return registry;
}

/// 查表创建：找不到名字就抛异常（演示 [4] 会触发一次）
static std::unique_ptr<Weapon> create_by_name(
    const std::map<std::string, WeaponCreator>& registry,
    const std::string& name) {
    auto it = registry.find(name);
    if (it == registry.end()) {
        throw std::runtime_error("registry: unknown weapon '" + name + "'");
    }
    return it->second();   // 调用注册的创建函数
}

// ============================================================================
// 主函数 —— 四种工厂依次登场
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Factory family: simple -> method ->" << std::endl;
    std::cout << "  abstract -> registry" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 演示 1：简单工厂 ====

    std::cout << std::endl;
    std::cout << "[1] Simple factory (one function + switch):" << std::endl;
    for (WeaponType t : { WeaponType::Sword, WeaponType::Bow,
                          WeaponType::MagicWand }) {
        auto w = SimpleWeaponFactory::create(t);
        w->attack();
    }
    std::cout << "    -> all 'new' concentrated in one switch..." << std::endl;
    std::cout << "       but adding Axe means EDITING it (violates OCP)" << std::endl;

    // ==== 演示 2：工厂方法 ====

    std::cout << std::endl;
    std::cout << "[2] Factory method (one factory per product):" << std::endl;

    // 注意：不能写成 { make_unique<A>(), ... } 初始化列表 ——
    // 各元素模板类型不同无法推导共同类型（Learn015 起的已知坑）
    std::vector<std::unique_ptr<WeaponFactory>> factories;
    factories.push_back(std::make_unique<SwordFactory>());
    factories.push_back(std::make_unique<BowFactory>());
    factories.push_back(std::make_unique<MagicWandFactory>());

    // 使用方只依赖抽象 WeaponFactory / Weapon —— 类型名单在这层不可见
    for (const auto& f : factories) {
        std::cout << "  via " << f->name() << ":" << std::endl;
        auto w = f->create_and_test();   // 公共流程：造 + 试一击
    }

    // 后来新增 Axe：只加了 Axe + AxeFactory 两个类，旧代码零改动
    std::cout << "  --- Axe added LATER (two new classes, zero edits) ---" << std::endl;
    AxeFactory axe_factory;
    auto axe = axe_factory.create_and_test();

    // ==== 演示 3：抽象工厂 ====

    std::cout << std::endl;
    std::cout << "[3] Abstract factory (a matched FAMILY per factory):" << std::endl;

    std::vector<std::unique_ptr<KitFactory>> kits;
    kits.push_back(std::make_unique<WarriorKitFactory>());
    kits.push_back(std::make_unique<MageKitFactory>());

    for (const auto& kit : kits) {
        std::cout << "  -- " << kit->kit_name()
                  << " (family consistency guaranteed) --" << std::endl;
        kit->make_weapon()->attack();
        kit->make_armor()->defend();
        kit->make_potion()->drink();
    }
    std::cout << "    -> switching style = swapping the whole factory;" << std::endl;
    std::cout << "       a warrior kit can never contain a wand" << std::endl;

    // ==== 演示 4：注册表工厂 ====

    std::cout << std::endl;
    std::cout << "[4] Registry factory (name -> creator, runtime open):" << std::endl;

    auto registry = make_weapon_registry();
    for (const std::string& name : { "sword", "wand", "axe" }) {
        std::cout << "  create(\"" << name << "\"):" << std::endl;
        create_by_name(registry, name)->attack();
    }

    // 运行期再注册一个（插件场景）：不碰任何既有代码
    registry["crossbow"] = [] { return std::make_unique<Bow>(); };
    std::cout << "  create(\"crossbow\") after runtime registration:" << std::endl;
    create_by_name(registry, "crossbow")->attack();

    // 未注册的名字：查表失败立即抛错（对照简单工厂 switch 漏分支的 UB）
    try {
        create_by_name(registry, "light saber");
    } catch (const std::runtime_error& e) {
        std::cout << "  caught: " << e.what() << std::endl;
    }

    // ==== 演示 5：四者对比 ====

    std::cout << std::endl;
    std::cout << "[5] Choosing among the four:" << std::endl;
    std::cout << "    simple  : one switch, cheapest; OCP violated;" << std::endl;
    std::cout << "               ok for few, stable products" << std::endl;
    std::cout << "    method  : one factory per product; OCP kept;" << std::endl;
    std::cout << "               class count doubles" << std::endl;
    std::cout << "    abstract: one factory per FAMILY; consistency;" << std::endl;
    std::cout << "               use when products come in matched sets" << std::endl;
    std::cout << "    registry: name->lambda map; runtime registration;" << std::endl;
    std::cout << "               plugin ecosystems' standard" << std::endl;
    std::cout << "    vs Builder (Learn018): factory ships a simple object" << std::endl;
    std::cout << "    in one shot; builder assembles a complex one step" << std::endl;
    std::cout << "    by step with an exit gate." << std::endl;

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. All factories hide 'new Concrete' behind an" << std::endl;
    std::cout << "     abstract product -- callers depend on interfaces." << std::endl;
    std::cout << "  2. Factory METHOD = 'which class to new' deferred" << std::endl;
    std::cout << "     to a subclass; creator also hosts common flows" << std::endl;
    std::cout << "     (create_and_test is a Learn017 template method)." << std::endl;
    std::cout << "  3. ABSTRACT factory creates a matched FAMILY;" << std::endl;
    std::cout << "     swapping families = swapping the factory object." << std::endl;
    std::cout << "  4. Registry replaces the switch with a map and" << std::endl;
    std::cout << "     opens registration to runtime (plugins)." << std::endl;
    std::cout << "  5. Series links: Strategy swaps behavior over time," << std::endl;
    std::cout << "     Factory creates once; Builder assembles stepwise." << std::endl;

    return 0;
}
