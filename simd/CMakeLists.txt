cmake_minimum_required(VERSION 3.13)
project(simd C CXX)

add_executable(permute permute.cpp)
target_compile_options(permute PRIVATE -mavx)
