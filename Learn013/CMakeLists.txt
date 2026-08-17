cmake_minimum_required(VERSION 3.20)

add_executable(Learn013 main.cu)

# 设置 UTF-8 编码以支持中文注释（MSVC 默认使用 GBK/CP936）
# NVCC + MSVC 需要通过 -Xcompiler 将 /utf-8 传递给 cl.exe
if(MSVC)
    target_compile_options(Learn013 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
