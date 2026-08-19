/**
 * @file    weapon.cpp
 * @brief   模板方法模式（游戏版）—— 武器攻击流程：骨架固定在基类 Weapon
 *
 * 与 main.cpp（数值版：迭代解法器）是**同一个模式、另一个领域**。
 * 游戏里近战剑 / 弓箭 / 法杖的攻击流程骨架完全同构：
 *
 *     门槛检查(冷却+资源) → 瞄准 → 消耗资源 → 攻击动作
 *                         → 伤害掷骰 → 结算扣血 → 命中钩子 → 进入冷却
 *
 * 把这段流程固定在 Weapon::attack()（非虚模板方法），具体武器只填
 * "会变的步骤"；伤害结算、门槛把关、冷却管理只写一遍 —— 将来加
 * "命中率高低的修正"或"换护甲减伤公式"，只改基类一处。
 *
 *        AbstractClass                  ConcreteClass
 *        ─────────────                  ─────────────
 *   ┌─────────────────────────┐   ┌──────────────────────────────┐
 *   │ Weapon                  │   │ Sword      近战：无弹药，踏步  │
 *   │  name_ base_damage_     │   │            近身，20% 暴击 ×2  │
 *   │  cooldown_ cooldown_rds │◄──┤ Bow        远程：耗箭，有冷却  │
 *   │                         │继承│ MagicWand  法术：耗蓝，溅射灼烧│
 *   │ + attack()  ← 模板方法  │   └──────────────────────────────┘
 *   │ + next_turn()  公共流程 │
 *   │ # aim()          钩子(默认实现) │
 *   │ # roll_damage()  钩子(默认实现) │
 *   │ # on_hit()       钩子(默认空)   │
 *   │ # has_resource() 原语(纯虚)     │
 *   │ # consume_resource() 原语(纯虚) │
 *   │ # fire()         原语(纯虚)     │
 *   └─────────────────────────┘
 *
 * 三种"方法"在本例中的分工（对照 main.cpp 的骨架五步）：
 *   - 模板方法 attack()：非虚。流程顺序、门槛、结算是**不变式** ——
 *     任何武器都不能绕过冷却/弹药检查私自开火，也不能跳过扣血结算。
 *     这正是游戏服防作弊外挂的第一道门：入口唯一（NVI 惯用法）。
 *   - 纯虚原语：has_resource / consume_resource / fire —— 每种武器必填。
 *     "资源"抽象统一了近战的"无弹药"、弓的"箭"、法杖的"法力"。
 *   - 带默认的钩子：aim（默认"抬手瞄准"）、roll_damage（默认
 *     基础伤害 + 0..3 浮动）、on_hit（默认空）。子类按需覆盖：
 *     Sword 覆盖 aim（踏步近身）与 roll_damage（暴击），
 *     MagicWand 覆盖 on_hit（灼烧溅射）—— 不覆盖也不会出错。
 *
 * 随机数用固定种子的 mt19937：演示输出可复现（教学程序的可贵品质）。
 *
 * 遵循本仓库约定：字符串字面量用英文，中文仅用于注释
 * （避免 GBK 控制台下的乱码问题，见 Learn012 的教训）。
 */

#include <cstdint>    // uint_fast32_t
#include <iostream>
#include <random>     // mt19937
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>  // SetConsoleOutputCP（Windows 控制台 UTF-8 支持）
#endif

/// 固定种子的随机源：伤害浮动/暴击可复现
static std::mt19937 rng(42u);

// ============================================================================
// 目标角色 —— 攻击流程的"结算端"（与模式无关的普通类）
// ============================================================================

/**
 * @brief 简化版角色：只做血量记录与扣血结算
 *
 * 注意 take_damage 是普通公有函数 —— 结算逻辑属于骨架（第 5 步），
 * 由模板方法统一调用，武器子类不直接碰它（权限集中在框架）。
 */
class Character {
public:
    Character(std::string name, int hp)
        : name_(std::move(name)), hp_(hp) {}

    void take_damage(int dmg) {
        int before = hp_;
        hp_ -= dmg;
        if (hp_ < 0) hp_ = 0;
        std::cout << "    [" << name_ << "] takes " << dmg
                  << " damage (hp " << before << " -> " << hp_ << ")"
                  << std::endl;
        if (hp_ == 0) {
            std::cout << "    [" << name_ << "] is defeated!" << std::endl;
        }
    }

    const std::string& name() const { return name_; }
    int hp() const { return hp_; }

private:
    std::string name_;
    int hp_;
};

// ============================================================================
// AbstractClass —— 抽象武器：攻击流程骨架 + 公共门槛/结算/冷却
// ============================================================================

class Weapon {
public:
    Weapon(std::string name, int base_damage, int cooldown_rounds)
        : name_(std::move(name)), base_damage_(base_damage),
          cooldown_rounds_(cooldown_rounds) {}

    virtual ~Weapon() = default;   // 虚析构：经基类指针删除派生对象

    const std::string& name() const { return name_; }

    /**
     * @brief 模板方法：攻击流程骨架 —— 刻意非虚！
     *
     * 步骤顺序是不变式：门槛永远在动作之前、结算永远在动作之后。
     * 若声明为 virtual，某种"神器"子类就能跳过冷却或扣血 ——
     * 骨架必须由基类独裁（NVI：公有接口非虚，虚步骤收进 protected）。
     */
    void attack(Character& target) {
        // 步骤 0a：公共门槛 —— 冷却（子类无权绕过）
        if (cooldown_ > 0) {
            std::cout << "    [" << name_ << "] not ready (cooldown "
                      << cooldown_ << " turn(s) left)" << std::endl;
            return;
        }
        // 步骤 0b：公共门槛 —— 资源（"弹药"的抽象统一三种武器）
        if (!has_resource()) {
            std::cout << "    [" << name_ << "] cannot attack (no resource)"
                      << std::endl;
            return;
        }

        aim(target);                 // 步骤 1：钩子（默认：抬手瞄准）
        consume_resource();          // 步骤 2：原语（剑无 / 弓箭 / 法力）
        fire(target);                // 步骤 3：原语（核心变化点：攻击动作）
        int dmg = roll_damage();     // 步骤 4：钩子（默认：基础伤害+浮动）
        target.take_damage(dmg);     // 步骤 5：公共结算（唯一扣血入口）
        on_hit(target, dmg);         // 步骤 6：钩子（默认空：命中连锁）
        cooldown_ = cooldown_rounds_;// 步骤 7：公共收尾（进入冷却）
    }

    /// 公共流程（非虚的普通方法）：回合推进、冷却递减
    void next_turn() {
        if (cooldown_ > 0) --cooldown_;
    }

protected:
    // ---- 钩子方法：带默认实现，可选覆盖（不覆盖也不出错）----

    /// 钩子 1：瞄准/起手动作（默认：抬手瞄准目标）
    virtual void aim(Character& target) {
        std::cout << "    [" << name_ << "] raises and aims at "
                  << target.name() << std::endl;
    }

    /// 钩子 2：伤害掷骰（默认：基础伤害 + 0..3 浮动）
    virtual int roll_damage() {
        int dmg = base_damage_ + static_cast<int>(rng() % 4);
        std::cout << "    [" << name_ << "] rolls damage: " << dmg
                  << std::endl;
        return dmg;
    }

    /// 钩子 3：命中后的连锁效果（默认空 —— 多数武器无连锁）
    virtual void on_hit(Character& /*target*/, int /*dmg*/) {}

    // ---- 原语操作：纯虚，具体武器必填 ----

    /// 原语 1：还打得出吗？（近战恒真；弓看箭；法杖看法力）
    virtual bool has_resource() const = 0;

    /// 原语 2：发射一发要消耗什么？
    virtual void consume_resource() = 0;

    /// 原语 3：攻击动作本身（挥砍 / 射箭 / 施法）
    virtual void fire(Character& target) = 0;

    std::string name_;          // 报表用
    int base_damage_;           // 伤害基准（钩子 roll_damage 使用）
    int cooldown_rounds_ = 0;   // 每次攻击后要冷却的回合数
    int cooldown_ = 0;          // 当前剩余冷却
};

// ============================================================================
// ConcreteClass —— 具体武器：只填原语，按需覆盖钩子
// ============================================================================

/**
 * @brief 近战剑：无弹药（has_resource 恒真）、无冷却、可暴击
 *
 * 覆盖了两个钩子：aim（踏步近身，起手动作与远程不同）、
 * roll_damage（20% 概率暴击 ×2 —— 覆盖"带默认实现"的原语）。
 */
class Sword final : public Weapon {
public:
    Sword() : Weapon("Sword", /*base_damage=*/8, /*cooldown_rounds=*/0) {}

protected:
    void aim(Character& target) override {      // 钩子覆盖：近战起手
        std::cout << "    [" << name() << "] steps into melee range of "
                  << target.name() << std::endl;
    }

    bool has_resource() const override { return true; }   // 近战无弹药

    void consume_resource() override { /* 剑不耗任何资源 */ }

    void fire(Character& target) override {
        std::cout << "    [" << name() << "] slashes at " << target.name()
                  << "!" << std::endl;
    }

    int roll_damage() override {                // 钩子覆盖：暴击掷骰
        int dmg = base_damage_ + static_cast<int>(rng() % 4);
        bool crit = rng() % 100 < 20;
        if (crit) dmg *= 2;
        std::cout << "    [" << name() << "] rolls damage: " << dmg
                  << (crit ? "  (critical hit!)" : "") << std::endl;
        return dmg;
    }
};

/**
 * @brief 弓：耗箭、攻击后冷却 1 回合 —— 演示两类公共门槛都会拦人
 */
class Bow final : public Weapon {
public:
    explicit Bow(int arrows)
        : Weapon("Bow", /*base_damage=*/6, /*cooldown_rounds=*/1),
          arrows_(arrows) {}

protected:
    bool has_resource() const override { return arrows_ > 0; }

    void consume_resource() override {
        --arrows_;
        std::cout << "    [" << name() << "] nocks an arrow ("
                  << arrows_ << " left)" << std::endl;
    }

    void fire(Character& target) override {
        std::cout << "    [" << name() << "] shoots an arrow at "
                  << target.name() << "!" << std::endl;
    }

private:
    int arrows_;
};

/**
 * @brief 法杖：耗法力、攻击后冷却 2 回合；命中附带灼烧（钩子 on_hit）
 */
class MagicWand final : public Weapon {
public:
    explicit MagicWand(int mana)
        : Weapon("MagicWand", /*base_damage=*/5, /*cooldown_rounds=*/2),
          mana_(mana), mana_cost_(10) {}

    int mana() const { return mana_; }

protected:
    bool has_resource() const override { return mana_ >= mana_cost_; }

    void consume_resource() override {
        mana_ -= mana_cost_;
        std::cout << "    [" << name() << "] spends " << mana_cost_
                  << " mana (" << mana_ << " left)" << std::endl;
    }

    void fire(Character& target) override {
        std::cout << "    [" << name() << "] casts a fireball at "
                  << target.name() << "!" << std::endl;
    }

    /// 钩子覆盖：命中连锁 —— 灼烧溅射追加 2 点（结算仍走公共扣血入口）
    void on_hit(Character& target, int /*dmg*/) override {
        std::cout << "    [" << name() << "] burn! splash 2 extra damage"
                  << std::endl;
        target.take_damage(2);
    }

private:
    int mana_;
    int mana_cost_;
};

// ============================================================================
// 主函数 —— 同一骨架跑三种武器；两类门槛拦截；钩子分工
// ============================================================================

int main() {
    // ==== Windows 控制台 UTF-8 ====
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "========================================" << std::endl;
    std::cout << "  Template Method (game flavor): weapon" << std::endl;
    std::cout << "  attack flow, skeleton in Weapon" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==== 演示 1：骨架 —— 固定在 Weapon::attack()（非虚）====

    std::cout << std::endl;
    std::cout << "[1] Skeleton fixed in Weapon::attack() (non-virtual):" << std::endl;
    std::cout << "    gate: cooldown / resource     <- common (base)" << std::endl;
    std::cout << "    aim(target)                   <- hook (default impl)" << std::endl;
    std::cout << "    consume_resource()            <- primitive (pure)" << std::endl;
    std::cout << "    fire(target)                  <- primitive (pure)" << std::endl;
    std::cout << "    roll_damage()                 <- hook (default impl)" << std::endl;
    std::cout << "    target.take_damage(dmg)       <- common settlement" << std::endl;
    std::cout << "    on_hit(target, dmg)           <- hook (default no-op)" << std::endl;
    std::cout << "    enter cooldown                <- common (base)" << std::endl;

    // ==== 演示 2：三种武器，同一骨架，不同步骤 ====

    std::cout << std::endl;
    std::cout << "[2] Same skeleton, three weapons:" << std::endl;

    Sword sword;
    Bow bow(/*arrows=*/3);
    MagicWand wand(/*mana=*/30);
    Character dummy("TrainingDummy", /*hp=*/100);

    std::cout << "  -- Sword (no ammo, no cooldown, crit) --" << std::endl;
    sword.attack(dummy);

    std::cout << "  -- Bow (arrows, 1-turn cooldown) --" << std::endl;
    bow.attack(dummy);

    std::cout << "  -- MagicWand (mana, 2-turn cooldown, burn) --" << std::endl;
    wand.attack(dummy);

    std::cout << "    dummy hp now: " << dummy.hp() << std::endl;

    // ==== 演示 3：公共门槛 —— 冷却与资源都由骨架把关 ====

    std::cout << std::endl;
    std::cout << "[3] Common gates enforced by the skeleton:" << std::endl;

    std::cout << "  -- Bow: blocked by cooldown, then next_turn --" << std::endl;
    bow.attack(dummy);        // 被冷却拦截（演示 2 那次攻击的冷却还在）
    bow.next_turn();          // 回合推进（公共流程，非模板方法）
    bow.attack(dummy);        // 恢复正常（箭 2 -> 1）

    std::cout << "  -- Bow: arrows run out (resource gate) --" << std::endl;
    bow.next_turn();          // 先清冷却
    bow.attack(dummy);        // 射出最后一支箭（箭 1 -> 0）
    bow.next_turn();          // 冷却虽清了……
    bow.attack(dummy);        // ……但被资源门槛拦下（无箭）
    bow.next_turn();
    bow.attack(dummy);        // 依然拦截：门槛 0b 与冷却无关

    std::cout << "  -- Sword: no ammo & no cooldown -> free to chain --" << std::endl;
    sword.attack(dummy);
    sword.attack(dummy);

    // ==== 演示 4：步骤角色映射（对照数值版 main.cpp）====

    std::cout << std::endl;
    std::cout << "[4] Role mapping vs the solver example (main.cpp):" << std::endl;
    std::cout << "    template method : attack()          | run()" << std::endl;
    std::cout << "    pure primitives : has_resource/     | initialize()/" << std::endl;
    std::cout << "                      consume/fire     | iterate()" << std::endl;
    std::cout << "    hooks w/ default: aim/roll_damage/  | on_iteration()" << std::endl;
    std::cout << "                      on_hit" << std::endl;
    std::cout << "    common control  : gates/settlement/ | residual check/" << std::endl;
    std::cout << "                      cooldown          | stats/stop flag" << std::endl;

    // ==== 总结 ====

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  Key takeaways" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "  1. attack() is NON-virtual: gates, order and" << std::endl;
    std::cout << "     settlement are invariants no weapon can skip." << std::endl;
    std::cout << "  2. 'Resource' abstraction unifies melee (none)," << std::endl;
    std::cout << "     arrows and mana behind one primitive." << std::endl;
    std::cout << "  3. Hooks: Sword overrides aim + roll_damage," << std::endl;
    std::cout << "     MagicWand overrides on_hit; Bow overrides none." << std::endl;
    std::cout << "  4. Fixed-seed rng -> reproducible demo output." << std::endl;
    std::cout << "  5. Same pattern as main.cpp: a game loop skeleton" << std::endl;
    std::cout << "     and a solver loop skeleton are the same shape." << std::endl;

    return 0;
}
