cmake_minimum_required(VERSION 3.20)

project(LearnCMake C CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(Learn001)
add_subdirectory(Learn002)
add_subdirectory(Learn003)
add_subdirectory(Learn004)
add_subdirectory(Learn005)
add_subdirectory(Learn006)
