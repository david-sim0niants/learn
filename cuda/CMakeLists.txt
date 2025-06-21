cmake_minimum_required(VERSION 3.13)
project(LearnCuda CXX CUDA)

if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif ()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(ROOT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

enable_testing()
add_subdirectory(test)
add_subdirectory(src)

option(HAVE_BENCHMARK "Enable benchmarks?" OFF)
if (HAVE_BENCHMARK)
    add_subdirectory(benchmark)
endif ()
