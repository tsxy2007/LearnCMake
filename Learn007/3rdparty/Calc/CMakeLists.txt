file(GLOB mysrc ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_library(Calc_object OBJECT ${mysrc})
target_link_libraries(Calc_object PRIVATE Sub_new)
target_link_libraries(Calc_object PRIVATE Add_new)
target_include_directories(Calc_object PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


add_library(Calc_new STATIC $<TARGET_OBJECTS:Calc_object> $<TARGET_OBJECTS:Add_object> $<TARGET_OBJECTS:Sub_object> $<TARGET_OBJECTS:common_object>)
target_include_directories(Calc_new PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)