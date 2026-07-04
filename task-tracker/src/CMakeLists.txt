set(TARGET_NAME task-tracker)
add_executable(${TARGET_NAME} main.cpp)
set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE CLI11::CLI11)
