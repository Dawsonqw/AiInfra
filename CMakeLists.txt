cmake_minimum_required(VERSION 3.16)
project(aiinfra_onnx_parser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AIINFRA_USE_BUNDLED_THIRDPARTY "Build third-party dependencies from submodules" ON)
option(AIINFRA_BUILD_BENCHMARKS "Build parser benchmarks" ON)

if(AIINFRA_USE_BUNDLED_THIRDPARTY)
    foreach(dependency googletest benchmark spdlog)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/${dependency}/CMakeLists.txt")
            message(FATAL_ERROR "Missing third-party submodule: thirdParty/${dependency}. Run git submodule update --init --recursive")
        endif()
    endforeach()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    add_subdirectory(thirdParty/googletest EXCLUDE_FROM_ALL)
    add_subdirectory(thirdParty/benchmark EXCLUDE_FROM_ALL)
    add_subdirectory(thirdParty/spdlog EXCLUDE_FROM_ALL)
endif()

find_package(Protobuf REQUIRED)
if(NOT AIINFRA_USE_BUNDLED_THIRDPARTY)
    find_package(spdlog CONFIG REQUIRED)
endif()

add_library(onnx_parser
    src/onnx_parser/onnx_parser.cc
    src/onnx_parser/operator.cc
    src/onnx_parser/operator_registry.cc
    src/onnx_parser/op_kind.cc
    src/onnx_parser/onnx_op_mapping.cc
    src/onnx_parser/shape_inference.cc
    src/onnx_parser/operators/common.cc
    src/onnx_parser/operators/unary.cc
    src/onnx_parser/operators/add.cc
    src/onnx_parser/operators/batch_normalization.cc
    src/onnx_parser/operators/conv.cc
    src/onnx_parser/operators/max_pool.cc
    src/onnx_parser/operators/global_average_pool.cc
    src/onnx_parser/operators/flatten.cc
    src/onnx_parser/operators/gemm.cc
    src/onnx_parser/operators/register_builtin.cc
    include/onnx/onnx.pb.cc
)
target_include_directories(onnx_parser
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(onnx_parser PUBLIC protobuf::libprotobuf)
target_link_libraries(onnx_parser PUBLIC spdlog::spdlog)
target_compile_options(onnx_parser PRIVATE -Wall -Wextra -Wpedantic)

add_executable(onnx_parser_cli src/onnx_parser/main.cc)
target_link_libraries(onnx_parser_cli PRIVATE onnx_parser)

include(CTest)
if(BUILD_TESTING AND AIINFRA_USE_BUNDLED_THIRDPARTY)
    add_executable(onnx_parser_test tests/onnx_parser_test.cc)
    target_link_libraries(onnx_parser_test PRIVATE onnx_parser)
    target_link_libraries(onnx_parser_test PRIVATE GTest::gtest_main)
    add_test(NAME onnx_parser_test COMMAND onnx_parser_test)
endif()

if(AIINFRA_BUILD_BENCHMARKS AND AIINFRA_USE_BUNDLED_THIRDPARTY)
    add_executable(onnx_parser_benchmark benchmarks/onnx_parser_benchmark.cc)
    target_compile_definitions(onnx_parser_benchmark PRIVATE
        AIINFRA_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/asserts")
    target_link_libraries(onnx_parser_benchmark PRIVATE onnx_parser benchmark::benchmark)
endif()
