set(INSTALL_GTEST OFF)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(SRCS
    vector_add.cpp
)

set(TARGET_NAME run_tests)
add_executable(${TARGET_NAME} ${SRCS})
set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE gtest_main learncuda_src)

add_test(NAME Test COMMAND run_tests)
