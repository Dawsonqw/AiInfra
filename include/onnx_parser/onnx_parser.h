#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace aiinfra::onnx {

struct Dimension {
    bool has_value = false;
    int64_t value = 0;
    std::string parameter;
};

struct TensorInfo {
    std::string name;
    int32_t elem_type = 0;
    std::vector<Dimension> shape;
    bool is_initializer = false;
    int64_t initializer_bytes = 0;
    bool shape_inferred = false;
};

using AttributeValue = std::variant<int64_t, float, std::string,
                                    std::vector<int64_t>, std::vector<float>,
                                    std::vector<std::string>>;

struct AttributeInfo {
    std::string name;
    int32_t type = 0;
    int32_t value_count = 0;
    AttributeValue value = int64_t{0};
};

struct NodeInfo {
    int32_t index = 0;
    std::string name;
    std::string op_type;
    std::string domain;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<AttributeInfo> attributes;
};

struct GraphInfo {
    std::string name;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    std::vector<TensorInfo> value_infos;
    std::vector<TensorInfo> initializers;
    std::vector<TensorInfo> tensors;
    std::vector<NodeInfo> nodes;
    std::vector<int32_t> topological_order;
};

struct ModelInfo {
    int64_t ir_version = 0;
    std::string producer_name;
    std::string producer_version;
    std::vector<std::string> opsets;
    GraphInfo graph;
};

class OnnxParser {
public:
    ModelInfo parse_file(const std::string& path, bool infer_shapes = true) const;
};

}  // namespace aiinfra::onnx
