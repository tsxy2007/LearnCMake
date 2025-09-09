cmake_minimum_required(VERSION 3.20)
add_executable(LearnECS main.cpp)
target_include_directories(LearnECS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)