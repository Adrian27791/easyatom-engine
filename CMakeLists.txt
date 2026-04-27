cmake_minimum_required(VERSION 3.20)
project(easyatom_engine LANGUAGES CXX VERSION 0.1.0)

# C++20 estricto, sin extensiones de compilador.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings agresivos: el motor no se permite el lujo de avisos sueltos.
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
endif()

# Header-only library del núcleo (de momento todo en .hpp).
add_library(easyatom_core INTERFACE)
target_include_directories(easyatom_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(easyatom_core INTERFACE cxx_std_20)

enable_testing()
add_subdirectory(tests)
