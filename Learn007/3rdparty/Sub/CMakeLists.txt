cmake_minimum_required(VERSION 3.20)
file(GLOB SubSrc ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(Sub_object OBJECT ${SubSrc})
target_link_libraries(Sub_object PRIVATE common)
target_include_directories(Sub_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_library(Sub_new STATIC $<TARGET_OBJECTS:Sub_object> $<TARGET_OBJECTS:common_object>)
target_include_directories(Sub_new PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
