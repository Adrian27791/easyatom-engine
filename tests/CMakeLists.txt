add_executable(test_easyatom
    test_main.cpp
    test_clifford.cpp
    test_hilbert.cpp
    test_ops.cpp
    test_infogeo.cpp
    test_topology.cpp
    test_dynamics.cpp
    test_laws.cpp
    test_qkernel.cpp
    test_c_api.cpp
    test_introspect.cpp
    ../src/c_api.cpp
)
target_link_libraries(test_easyatom PRIVATE easyatom_core)
target_include_directories(test_easyatom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_test(NAME easyatom_all COMMAND test_easyatom)
