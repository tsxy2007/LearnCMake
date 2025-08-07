cmake_minimum_required(VERSION 3.20)

add_library(common src/common.cpp)

target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)