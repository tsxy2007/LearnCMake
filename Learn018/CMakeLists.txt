cmake_minimum_required(VERSION 3.20)

# 示例一（标准四角色）：游戏角色的分步构建
add_executable(Learn018 main.cpp)

# 示例二（进阶）：各类型参数不一致时的三种设计
add_executable(Learn018Params params.cpp)

# 示例三（进阶二）：参数基类（Parameter Object）整体传入
add_executable(Learn018Config config.cpp)

# 设置 UTF-8 编码以支持中文注释（MSVC 默认使用 GBK/CP936）
if(MSVC)
    target_compile_options(Learn018 PRIVATE /utf-8)
    target_compile_options(Learn018Params PRIVATE /utf-8)
    target_compile_options(Learn018Config PRIVATE /utf-8)
endif()
