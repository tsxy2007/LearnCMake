cmake_minimum_required(VERSION 3.20)

add_library(myLib STATIC src/Library.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(Learn002 main.cpp)
target_link_libraries(Learn002 PUBLIC myLib)