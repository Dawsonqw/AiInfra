cmake_minimum_required(VERSION 3.20)
project(cuda_learning LANGUAGES CXX CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(01-vector-add)
add_subdirectory(02-elementwise)
add_subdirectory(03-reduce)
add_subdirectory(04-softmax)
add_subdirectory(05-mat-transpose)
add_subdirectory(06-sgemm)