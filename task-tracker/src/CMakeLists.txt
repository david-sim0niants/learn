set(TARGET_NAME task-tracker)
add_executable(${TARGET_NAME} main.cpp task_tracker.cpp platform.cpp sqlite_helpers.cpp)

target_include_directories(${TARGET_NAME}
    PRIVATE ${ROOT_INCLUDE_DIR})

set_target_properties(${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(SQLite3 REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE CLI11::CLI11 SQLite3::SQLite3)

target_compile_definitions(${TARGET_NAME} PRIVATE TASK_TRACKER_SHARED_DATA_DIR="${SHARED_DATA_DIR}")
