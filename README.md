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
