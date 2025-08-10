cmake_minimum_required(VERSION 3.20)

add_library(common_object OBJECT src/common.cpp)
target_include_directories(common_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(common_new STATIC $<TARGET_OBJECTS:common_object>)
target_include_directories(common_new PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)