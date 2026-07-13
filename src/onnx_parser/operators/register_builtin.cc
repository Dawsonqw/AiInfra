#include "onnx_parser/operator_registry.h"

#include "onnx_parser/operators/add.h"
#include "onnx_parser/operators/batch_normalization.h"
#include "onnx_parser/operators/conv.h"
#include "onnx_parser/operators/flatten.h"
#include "onnx_parser/operators/gemm.h"
#include "onnx_parser/operators/global_average_pool.h"
#include "onnx_parser/operators/max_pool.h"
#include "onnx_parser/operators/unary.h"

namespace aiinfra::onnx {

void register_builtin_operators(OperatorRegistry& registry) {
    using operators::AddOperator;
    using operators::BatchNormalizationOperator;
    using operators::ConvOperator;
    using operators::FlattenOperator;
    using operators::GemmOperator;
    using operators::GlobalAveragePoolOperator;
    using operators::MaxPoolOperator;
    using operators::UnaryOperator;

    registry.register_operator(OpKind::Identity, [] { return std::make_unique<UnaryOperator>(OpKind::Identity); });
    registry.register_operator(OpKind::Relu, [] { return std::make_unique<UnaryOperator>(OpKind::Relu); });
    registry.register_operator(OpKind::BatchNormalization, [] { return std::make_unique<BatchNormalizationOperator>(); });
    registry.register_operator(OpKind::Add, [] { return std::make_unique<AddOperator>(); });
    registry.register_operator(OpKind::Conv, [] { return std::make_unique<ConvOperator>(); });
    registry.register_operator(OpKind::MaxPool, [] { return std::make_unique<MaxPoolOperator>(); });
    registry.register_operator(OpKind::GlobalAveragePool, [] { return std::make_unique<GlobalAveragePoolOperator>(); });
    registry.register_operator(OpKind::Flatten, [] { return std::make_unique<FlattenOperator>(); });
    registry.register_operator(OpKind::Gemm, [] { return std::make_unique<GemmOperator>(); });
}

}  // namespace aiinfra::onnx
