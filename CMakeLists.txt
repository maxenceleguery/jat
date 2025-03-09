cmake_minimum_required(VERSION 3.10)
project(jat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(include)

set(SOURCES
    #src/main.cpp
    src/cuda/kernels.cu
)

add_library(jat ${SOURCES})

#add_executable(jat_main src/main.cpp)
#target_link_libraries(jat_main jat)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB SRC_TESTS_FILES tests/*.cc)

add_executable(test_main tests/test_main.cpp ${SRC_TESTS_FILES})
target_link_libraries(test_main gtest gtest_main jat)
target_compile_options(test_main PRIVATE -w)
add_test(NAME test_main COMMAND test_main)