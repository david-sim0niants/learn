cmake_minimum_required(VERSION 3.13)
project(simd C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_executable(permute permute.cpp)
target_compile_options(permute PRIVATE -mavx)

add_executable(dot_product dot_product.cpp)
target_compile_options(dot_product PRIVATE -mavx -O3)
