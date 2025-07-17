cmake_minimum_required(VERSION 3.13)
project(SecBootSim C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_library(SSL_LIB NAME ssl REQUIRED)
find_library(CRYPTO_LIB NAME crypto REQUIRED)

add_library(common STATIC pkey.cpp checksum.cpp)
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_link_libraries(common PUBLIC ${SSL_LIB} ${CRYPTO_LIB})

add_executable(sign sign.cpp)
target_link_libraries(sign PRIVATE common)

add_executable(verify verify.cpp)
target_link_libraries(verify PRIVATE common)

add_custom_target(keygen ALL
    COMMAND cmake -E copy ${CMAKE_CURRENT_SOURCE_DIR}/keygen.sh ${CMAKE_CURRENT_BINARY_DIR}/keygen)
