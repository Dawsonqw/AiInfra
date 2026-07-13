#pragma once

#include <string_view>

#include "onnx_parser/op_kind.h"

namespace aiinfra::onnx {

// ONNX 专属的名称映射，核心 Operator/Registry 不依赖 ONNX 字符串。
OpKind onnx_op_kind(std::string_view op_type, std::string_view domain = {});

}  // namespace aiinfra::onnx
