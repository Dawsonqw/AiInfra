#include "onnx_parser/op_kind.h"

namespace aiinfra::onnx {

std::string op_kind_name(OpKind kind) {
    switch (kind) {
    case OpKind::Identity: return "Identity";
    case OpKind::Conv: return "Conv";
    case OpKind::Relu: return "Relu";
    case OpKind::MaxPool: return "MaxPool";
    case OpKind::Add: return "Add";
    case OpKind::BatchNormalization: return "BatchNormalization";
    case OpKind::GlobalAveragePool: return "GlobalAveragePool";
    case OpKind::Flatten: return "Flatten";
    case OpKind::Gemm: return "Gemm";
    case OpKind::Unknown: return "Unknown";
    }
    return "Unknown";
}

}  // namespace aiinfra::onnx
