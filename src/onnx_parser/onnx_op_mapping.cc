#include "onnx_parser/onnx_op_mapping.h"

#include <unordered_map>

namespace aiinfra::onnx {
namespace {

const std::unordered_map<std::string_view, OpKind> kOnnxOpMap = {
    {"Identity", OpKind::Identity},
    {"Conv", OpKind::Conv},
    {"Relu", OpKind::Relu},
    {"MaxPool", OpKind::MaxPool},
    {"Add", OpKind::Add},
    {"BatchNormalization", OpKind::BatchNormalization},
    {"GlobalAveragePool", OpKind::GlobalAveragePool},
    {"Flatten", OpKind::Flatten},
    {"Gemm", OpKind::Gemm},
};

}  // namespace

OpKind onnx_op_kind(std::string_view op_type, std::string_view domain) {
    if (!domain.empty() && domain != "ai.onnx") return OpKind::Unknown;
    const auto found = kOnnxOpMap.find(op_type);
    return found == kOnnxOpMap.end() ? OpKind::Unknown : found->second;
}

}  // namespace aiinfra::onnx
