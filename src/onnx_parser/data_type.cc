#include "onnx_parser/data_type.h"

#include <stdexcept>

namespace aiinfra::onnx {

std::size_t data_type_size(DataType type) {
    switch (type) {
    case DataType::Float32: return sizeof(float);
    case DataType::Float16: return sizeof(std::uint16_t);
    case DataType::Int32: return sizeof(std::int32_t);
    case DataType::Int64: return sizeof(std::int64_t);
    case DataType::Unknown: break;
    }
    throw std::invalid_argument("unknown data type has no storage size");
}

std::string data_type_name(DataType type) {
    switch (type) {
    case DataType::Float32: return "float32";
    case DataType::Float16: return "float16";
    case DataType::Int32: return "int32";
    case DataType::Int64: return "int64";
    case DataType::Unknown: return "unknown";
    }
    return "unknown";
}

}  // namespace aiinfra::onnx
