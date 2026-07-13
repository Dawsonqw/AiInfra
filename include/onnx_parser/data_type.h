#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace aiinfra::onnx {

enum class DataType {
    Unknown = 0,
    Float32,
    Float16,
    Int32,
    Int64,
};

std::size_t data_type_size(DataType type);
std::string data_type_name(DataType type);

}  // namespace aiinfra::onnx
