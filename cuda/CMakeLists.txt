cmake_minimum_required(VERSION 3.13)
project(LearnCuda C CXX CUDA)

set(TARGET_NAME run_samples)
set(SRCS
    main.cpp
    vector_add.cu
)

set(ROOT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

add_executable(${TARGET_NAME} ${SRCS})
target_include_directories(${TARGET_NAME} PRIVATE ${ROOT_INCLUDE_DIR})
