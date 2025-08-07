cmake_minimum_required(VERSION 3.20)

add_subdirectory(3rdparty)
add_library(myLib STATIC src/Library.cpp)
# include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(Learn002 main.cpp)
target_link_libraries(Learn002 PUBLIC myLib) 
target_link_libraries(Learn002 PRIVATE Calc)

