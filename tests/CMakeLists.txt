add_executable(test_easyatom
    test_main.cpp
    test_clifford.cpp
)
target_link_libraries(test_easyatom PRIVATE easyatom_core)
target_include_directories(test_easyatom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME easyatom_all COMMAND test_easyatom)
