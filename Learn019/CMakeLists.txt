cmake_minimum_required(VERSION 3.20)

add_executable(Learn019 main.cpp)

# 设置 UTF-8 编码以支持中文注释（MSVC 默认使用 GBK/CP936）
if(MSVC)
    target_compile_options(Learn019 PRIVATE /utf-8)
endif()
