cmake_minimum_required(VERSION 3.20)

file(GLOB mysrc ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(Add STATIC ${mysrc})
target_link_libraries(Add PRIVATE common)
target_include_directories(Add PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)