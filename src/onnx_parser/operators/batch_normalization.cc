#include "onnx_parser/operators/batch_normalization.h"

namespace aiinfra::onnx::operators {

BatchNormalizationOperator::BatchNormalizationOperator()
    : UnaryOperator("BatchNormalization") {}

}  // namespace aiinfra::onnx::operators
