cmake_minimum_required(VERSION 3.20)

file(GLOB mysrc ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(Add_object STATIC ${mysrc})
target_include_directories(Add_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Add_object PRIVATE common_new)
add_library(Add_new STATIC $<TARGET_OBJECTS:Add_object> $<TARGET_OBJECTS:common_object>)
target_include_directories(Add_new PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)