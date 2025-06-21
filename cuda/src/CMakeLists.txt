set(TARGET_NAME learncuda_src)

set(SRCS
    vector_add.cu
)

add_library(${TARGET_NAME} INTERFACE)
target_sources(${TARGET_NAME} INTERFACE ${SRCS})
target_include_directories(${TARGET_NAME} INTERFACE ${ROOT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
