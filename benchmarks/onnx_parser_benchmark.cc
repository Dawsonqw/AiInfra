#include <benchmark/benchmark.h>

#include "onnx_parser/onnx_parser.h"

void benchmark_parse_model(benchmark::State& state) {
    const std::string path = std::string(AIINFRA_ASSET_DIR) + "/resnet18.onnx";
    for (auto _ : state) {
        const auto model = aiinfra::onnx::OnnxParser().parse_file(path);
        benchmark::DoNotOptimize(model.graph.nodes.size());
    }
}

BENCHMARK(benchmark_parse_model);
BENCHMARK_MAIN();
