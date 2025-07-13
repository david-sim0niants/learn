cmake_minimum_required(VERSION 3.13)
project(learn_io_uring C)

find_library(URING_LIB uring REQUIRED)

add_executable(basic_fread basic_fread.c)
target_link_libraries(basic_fread PRIVATE ${URING_LIB})
