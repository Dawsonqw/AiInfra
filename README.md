# ONNX Protobuf 依赖

项目使用 ONNX 的 Protocol Buffers 定义生成 C++ 代码。Ubuntu 环境下可按以下步骤安装依赖并生成头文件：

```bash
# 安装 protoc 编译器和 Protocol Buffers C++ 开发库
sudo apt update
sudo apt install -y protobuf-compiler libprotobuf-dev

# 下载与项目匹配的 ONNX 协议定义
mkdir -p thirdParty/onnx
wget -O thirdParty/onnx/onnx.proto \
  https://raw.githubusercontent.com/onnx/onnx/v1.19.1/onnx/onnx.proto

# 生成 C++ 源文件和头文件
mkdir -p include/onnx
protoc \
  --proto_path=thirdParty/onnx \
  --cpp_out=include/onnx \
  thirdParty/onnx/onnx.proto

# 检查生成结果
ls include/onnx/onnx.pb.h include/onnx/onnx.pb.cc
```

其中，`onnx.proto` 是 ONNX v1.19.1 的协议定义，`onnx.pb.h` 和 `onnx.pb.cc` 是 `protoc` 生成的 C++ 文件。若更换 ONNX 版本，请同步更新下载地址并重新生成代码。

## ResNet-18 ONNX Parser

项目提供一个不依赖具体算子实现的 ONNX 计算图解析器。它将 protobuf 中的
`ModelProto/GraphProto` 转换为稳定的 C++ IR，包含模型元信息、图输入输出、
initializer、节点属性摘要以及拓扑序，后续 CUDA executor 可以直接基于这些
结构实现算子调度。

使用当前 uv 虚拟环境导出可复用的 torchvision ResNet-18：

```bash
.venv/bin/python scripts/export_resnet18.py
```

模型保存到 `asserts/resnet18.onnx`，输入固定为 `[1, 3, 224, 224]`。

构建、测试并解析模型：

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
build/onnx_parser_cli asserts/resnet18.onnx
```

解析器会检查重复 tensor producer、未知输入、输入与 initializer 命名冲突以及
计算图环路；节点原始顺序保存在 `GraphInfo::nodes`，可执行拓扑顺序保存在
`GraphInfo::topological_order`。

## CUDA Backend

项目已启用 CUDA，并提供独立的 backend 层：

```text
src/backend/cuda/
├── cuda_tensor.cc              # RAII device tensor 与 host/device copy
├── cuda_operator_registry.cc   # 基于 OpKind 的 CUDA 算子工厂
├── cuda_executor.cc            # 拓扑序编译与 stream 执行
└── operators/
    ├── identity.cu
    └── relu.cu
```

构建 CUDA 版本：

```bash
cmake -S . -B build-cuda \
  -DAIINFRA_BUILD_BENCHMARKS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
```

`Operator` 仍只负责 shape inference；CUDA 算子实现 `cuda::Operator` 负责执行。
当前 CUDA smoke test 覆盖 `Identity -> Relu` 图和 float32 Tensor。后续新增算子时，
分别在 `src/backend/cuda/operators/` 增加 kernel、实现类，并在 CUDA registry 中注册。
当前 executor 使用单 stream 和每个 Tensor 独立分配的简单策略，暂未引入 workspace
复用、memory planner、initializer 上传策略和 Conv kernel。

CUDA executor 使用依赖注入的 registry，不会在 `compile()` 内隐式修改全局注册表：

```cpp
cuda::OperatorRegistry registry;
registry.register_operator(
    OpKind::Relu,
    [](const NodeInfo& node) {
        return std::make_unique<MyReluOperator>(node);
    });

cuda::Executor executor(registry);
executor.compile(graph);
executor.copy_tensor("input", input.data(), input_bytes);
executor.copy_tensor("weight", weight.data(), weight_bytes);  // initializer 也用同一接口
executor.run();
executor.copy_output("output", output.data(), output_bytes);
executor.synchronize();
```

单算子测试应构造只有一个节点的 `GraphInfo`，使用独立 registry 注册被测算子，
再用 CPU/PyTorch reference 结果与 `copy_output()` 取回的结果进行比较。这样测试
不会依赖全局状态，也不会把 CUDA 实现本身当作正确性依据。

算子代码按文件拆分在 `include/onnx_parser/operators/` 与
`src/onnx_parser/operators/` 下：每个算子拥有独立实现文件，`common.*` 放置
公共 shape/属性辅助函数，`register_builtin.cc` 只负责内置算子注册。

核心节点使用 `OpKind` enum，不直接依赖 ONNX 的字符串名称。ONNX 的
`op_type/domain` 仅在 `onnx_op_mapping.*` 中转换为内部类型；以后接入其他模型
格式时，只需新增对应 adapter 映射到同一组 `OpKind`，算子实现和 CUDA executor
无需感知模型来源。未知扩展算子会保留原始名称，并在 shape inference 阶段给出清晰错误。

### 算子扩展模型

算子由 `Operator` 多态基类定义，核心接口是：

```cpp
class MyOperator final : public Operator {
public:
    std::vector<TensorInfo> infer_shape(const OperatorContext& context) const override;
    OpKind type() const noexcept override { return OpKind::Conv; }
};

OperatorRegistry::global().register_operator(
    OpKind::Conv, [] { return std::make_unique<MyOperator>(); });
```

`OperatorRegistry` 负责按 `domain:op_type` 创建实例，`ShapeInference` 按计算图拓扑
顺序调用 `infer_shape`，并把结果回写到 `GraphInfo::tensors` 和 graph outputs。
内置算子目前覆盖 ResNet-18 所需的 `Identity`、`Conv`、`Relu`、`MaxPool`、
`BatchNormalization`、`Add`、`GlobalAveragePool`、`Flatten`、`Gemm`。

### 第三方依赖

GoogleTest、Google Benchmark 和 spdlog 通过 Git submodule 管理：

```bash
git submodule update --init --recursive
cmake -S . -B build -DAIINFRA_USE_BUNDLED_THIRDPARTY=ON
```

关闭 `AIINFRA_USE_BUNDLED_THIRDPARTY` 可在已有系统依赖环境中复用 protobuf；
但测试和 benchmark target 需要 bundled dependencies。
