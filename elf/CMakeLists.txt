cmake_minimum_required(VERSION 3.13)
project(LearnELF C CXX)

set(CMAKE_CXX_STANDARD 20)

set(TARGET_NAME loadelf)
add_library(${TARGET_NAME} SHARED loadelf.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE -lelf)

set(TARGET_NAME runelf)
add_executable(${TARGET_NAME} runelf.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE loadelf)
