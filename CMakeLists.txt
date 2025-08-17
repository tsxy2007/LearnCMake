cmake_minimum_required(VERSION 3.20)

project(LearnCMake C CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(Learn001)
add_subdirectory(Learn002)
add_subdirectory(Learn003)
add_subdirectory(Learn004)
add_subdirectory(Learn005)
add_subdirectory(Learn006) 
add_subdirectory(Learn007)
add_subdirectory(Learn008)
add_subdirectory(Learn009)
add_subdirectory(Learn010)