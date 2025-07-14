cmake_minimum_required(VERSION 3.13)
project(LearnASMJIT C CXX)

add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE -lasmjit)
