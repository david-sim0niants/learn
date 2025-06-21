find_package(benchmark REQUIRED)

set(TARGET_NAME run_bms)
set(SRCS
    vector_add.cpp
)

add_executable(${TARGET_NAME} ${SRCS})
target_link_libraries(${TARGET_NAME} PRIVATE benchmark::benchmark_main learncuda_src)
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
