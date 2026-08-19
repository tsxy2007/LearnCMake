cmake_minimum_required(VERSION 3.20)

# 示例一（数值版）：迭代解法器骨架 —— Jacobi / Gauss-Seidel / CG
add_executable(Learn017 main.cpp)

# 示例二（游戏版）：武器攻击流程骨架 —— Sword / Bow / MagicWand
add_executable(Learn017Weapon weapon.cpp)

# 设置 UTF-8 编码以支持中文注释（MSVC 默认使用 GBK/CP936）
if(MSVC)
    target_compile_options(Learn017 PRIVATE /utf-8)
    target_compile_options(Learn017Weapon PRIVATE /utf-8)
endif()
