#include "onnx_parser/operators/batch_normalization.h"

namespace aiinfra::onnx::operators {

BatchNormalizationOperator::BatchNormalizationOperator()
    : UnaryOperator(OpKind::BatchNormalization) {}

}  // namespace aiinfra::onnx::operators
