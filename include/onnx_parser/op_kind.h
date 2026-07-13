#pragma once

#include <string>

namespace aiinfra::onnx {

// 模型格式适配器将外部算子名称转换为这个稳定的内部标识。
enum class OpKind {
    Unknown = 0,
    Identity,
    Conv,
    Relu,
    MaxPool,
    Add,
    BatchNormalization,
    GlobalAveragePool,
    Flatten,
    Gemm,
};

std::string op_kind_name(OpKind kind);

}  // namespace aiinfra::onnx
