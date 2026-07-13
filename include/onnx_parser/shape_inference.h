#pragma once

#include "onnx_parser/onnx_parser.h"

namespace aiinfra::onnx {

class ShapeInference {
public:
    explicit ShapeInference(bool allow_unknown_operators = false)
        : allow_unknown_operators_(allow_unknown_operators) {}

    void infer(GraphInfo& graph) const;

private:
    bool allow_unknown_operators_ = false;
};

}  // namespace aiinfra::onnx
