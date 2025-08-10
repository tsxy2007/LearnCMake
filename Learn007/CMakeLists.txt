cmake_minimum_required(VERSION 3.20)

add_subdirectory(3rdparty)
add_library(myLib_new STATIC src/Library.cpp)
# include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(Learn007 main.cpp)
target_link_libraries(Learn007 PUBLIC myLib_new) 
target_link_libraries(Learn007 PRIVATE Calc_new)

