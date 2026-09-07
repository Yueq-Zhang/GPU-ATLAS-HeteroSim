#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status)); \
      return 2;                                                                 \
    }                                                                           \
  } while (0)

namespace {

constexpr int kHiddenSize = 2048;
constexpr int kVocabSize = 32000;
constexpr const char* kRevision =
    "fe8a4ea1ffedaf415f4da2f062534de366a451e6";

__global__ void token_embedding_gather(const std::int64_t* token_ids,
                                       const __half* weight, __half* output,
                                       int token_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = token_count * kHiddenSize;
  if (index < count) {
    const int token = index / kHiddenSize;
    const int hidden = index % kHiddenSize;
    output[index] = weight[token_ids[token] * kHiddenSize + hidden];
  }
}

__global__ void residual_add_fp16(const __half* input, const __half* residual,
                                  __half* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __hadd(input[index], residual[index]);
  }
}

struct Summary {
  double minimum_fs;
  double p10_fs;
  double median_fs;
  double p90_fs;
  double maximum_fs;
  double mean_fs;
};

Summary summarize(std::vector<double> values_fs) {
  std::sort(values_fs.begin(), values_fs.end());
  auto percentile = [&](double fraction) {
    const std::size_t index = static_cast<std::size_t>(
        std::llround(fraction * static_cast<double>(values_fs.size() - 1)));
    return values_fs[index];
  };
  const double mean =
      std::accumulate(values_fs.begin(), values_fs.end(), 0.0) /
      static_cast<double>(values_fs.size());
  return {values_fs.front(), percentile(0.10), percentile(0.50),
          percentile(0.90), values_fs.back(), mean};
}

int measure_cuda_event(const std::function<void()>& operation, int warmup,
                       int iterations, Summary* result) {
  for (int index = 0; index < warmup; ++index) {
    operation();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> values_fs;
  values_fs.reserve(iterations);
  for (int index = 0; index < iterations; ++index) {
    CUDA_CHECK(cudaEventRecord(start));
    operation();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    values_fs.push_back(static_cast<double>(elapsed_ms) * 1.0e12);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  *result = summarize(values_fs);
  return 0;
}

void write_summary(std::ofstream& stream, const Summary& summary) {
  stream << "{\"min_fs\":" << std::llround(summary.minimum_fs)
         << ",\"p10_fs\":" << std::llround(summary.p10_fs)
         << ",\"median_fs\":" << std::llround(summary.median_fs)
         << ",\"p90_fs\":" << std::llround(summary.p90_fs)
         << ",\"max_fs\":" << std::llround(summary.maximum_fs)
         << ",\"mean_fs\":" << std::llround(summary.mean_fs) << "}";
}

void write_tensor(std::ofstream& stream, const std::string& tensor_id,
                  const std::string& role, const void* address,
                  std::size_t size_bytes, const std::string& shape,
                  const std::string& strides, const std::string& dtype,
                  bool comma) {
  stream << "    {\n"
         << "      \"tensor_id\": \"" << tensor_id << "\",\n"
         << "      \"role\": \"" << role << "\",\n"
         << "      \"address\": "
         << static_cast<unsigned long long>(
                reinterpret_cast<std::uintptr_t>(address))
         << ",\n"
         << "      \"size_bytes\": " << size_bytes << ",\n"
         << "      \"shape\": " << shape << ",\n"
         << "      \"strides\": " << strides << ",\n"
         << "      \"dtype\": \"" << dtype << "\",\n"
         << "      \"layout\": \"strided\",\n"
         << "      \"alignment_bytes\": 256\n"
         << "    }" << (comma ? "," : "") << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string operator_name;
  std::string metadata_output;
  std::string native_measurement_output;
  int context = 16;
  int warmup = 50;
  int iterations = 500;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--operator" && index + 1 < argc) {
      operator_name = argv[++index];
    } else if (argument == "--context" && index + 1 < argc) {
      context = std::atoi(argv[++index]);
    } else if (argument == "--metadata-output" && index + 1 < argc) {
      metadata_output = argv[++index];
    } else if (argument == "--native-measurement-output" &&
               index + 1 < argc) {
      native_measurement_output = argv[++index];
    } else if (argument == "--warmup" && index + 1 < argc) {
      warmup = std::atoi(argv[++index]);
    } else if (argument == "--iterations" && index + 1 < argc) {
      iterations = std::atoi(argv[++index]);
    } else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[index]);
      return 2;
    }
  }
  if ((operator_name != "token_embedding" &&
       operator_name != "residual_add") ||
      context <= 0 || warmup < 0 || iterations <= 0 ||
      (metadata_output.empty() == native_measurement_output.empty())) {
    std::fprintf(stderr,
                 "usage: %s --operator token_embedding|residual_add "
                 "--context N (--metadata-output PATH | "
                 "--native-measurement-output PATH "
                 "[--warmup N] [--iterations N])\n",
                 argv[0]);
    return 2;
  }

  const int elements = context * kHiddenSize;
  const std::size_t tensor_bytes = elements * sizeof(__half);
  void* first = nullptr;
  void* second = nullptr;
  __half* output = nullptr;
  std::int64_t* token_ids = nullptr;

  if (operator_name == "token_embedding") {
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kVocabSize) * kHiddenSize * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&token_ids, context * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&first, weight_bytes));
    CUDA_CHECK(cudaMalloc(&output, tensor_bytes));
    CUDA_CHECK(cudaMemset(token_ids, 0, context * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMemset(first, 0, weight_bytes));
    CUDA_CHECK(cudaMemset(output, 0, tensor_bytes));
  } else {
    CUDA_CHECK(cudaMalloc(&first, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&second, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&output, tensor_bytes));
    CUDA_CHECK(cudaMemset(first, 0, tensor_bytes));
    CUDA_CHECK(cudaMemset(second, 0, tensor_bytes));
    CUDA_CHECK(cudaMemset(output, 0, tensor_bytes));
  }

  const int grid_blocks = (elements + 255) / 256;
  const auto launch = [&] {
    if (operator_name == "token_embedding") {
      token_embedding_gather<<<grid_blocks, 256>>>(
          token_ids, static_cast<const __half*>(first), output, context);
    } else {
      residual_add_fp16<<<grid_blocks, 256>>>(
          static_cast<const __half*>(first), static_cast<const __half*>(second),
          output, elements);
    }
  };

  if (!metadata_output.empty()) {
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::ofstream metadata(metadata_output);
    if (!metadata) {
      std::fprintf(stderr, "cannot open metadata output: %s\n",
                   metadata_output.c_str());
      return 2;
    }
    metadata << "{\n"
           << "  \"schema_version\": \"heterosim-exact-llm-operator/v2\",\n"
           << "  \"model\": \"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",\n"
           << "  \"model_spec_name\": \"TinyLlama-1.1B\",\n"
           << "  \"revision\": \"" << kRevision << "\",\n"
           << "  \"operator\": \"" << operator_name << "\",\n"
           << "  \"phase\": \"prefill\",\n"
           << "  \"layer_id\": 0,\n"
           << "  \"batch_size\": 1,\n"
           << "  \"context_length\": " << context << ",\n"
           << "  \"q_len\": " << context << ",\n"
           << "  \"kv_length\": " << context << ",\n"
           << "  \"dtype\": \"fp16\",\n"
           << "  \"implementation\": \"heterosim_cuda_reference_"
           << operator_name << "_v1\",\n"
           << "  \"compilation\": {\"framework\": \"standalone_cuda\", "
              "\"compiler\": \"nvcc\", \"cuda_toolkit\": \"11.8\", "
              "\"target_sm\": 86},\n"
           << "  \"warmup_iterations\": 0,\n"
           << "  \"capture_selector\": \"single_kernel_process\",\n"
           << "  \"capture_allocator\": null,\n"
           << "  \"scope\": \"shape_and_implementation_locked_operator_not_framework_overhead\",\n"
           << "  \"performance_eligible\": false,\n"
           << "  \"tensors\": [\n";
    if (operator_name == "token_embedding") {
      write_tensor(metadata, "tinyllama.layer0.token_embedding.token_ids", "input",
                   token_ids, context * sizeof(std::int64_t),
                   "[1, " + std::to_string(context) + "]",
                   "[" + std::to_string(context) + ", 1]", "int64", true);
      write_tensor(metadata, "tinyllama.layer0.token_embedding.weight", "parameter",
                   first,
                   static_cast<std::size_t>(kVocabSize) * kHiddenSize *
                       sizeof(__half),
                   "[32000, 2048]", "[2048, 1]", "float16", true);
      write_tensor(metadata, "tinyllama.layer0.token_embedding.output", "output",
                   output, tensor_bytes,
                   "[1, " + std::to_string(context) + ", 2048]",
                   "[" + std::to_string(context * kHiddenSize) + ", 2048, 1]",
                   "float16", false);
    } else {
      write_tensor(metadata, "tinyllama.layer0.residual_add.input", "input", first,
                   tensor_bytes,
                   "[1, " + std::to_string(context) + ", 2048]",
                   "[" + std::to_string(context * kHiddenSize) + ", 2048, 1]",
                   "float16", true);
      write_tensor(metadata, "tinyllama.layer0.residual_add.residual", "input",
                   second, tensor_bytes,
                   "[1, " + std::to_string(context) + ", 2048]",
                   "[" + std::to_string(context * kHiddenSize) + ", 2048, 1]",
                   "float16", true);
      write_tensor(metadata, "tinyllama.layer0.residual_add.output", "output", output,
                   tensor_bytes,
                   "[1, " + std::to_string(context) + ", 2048]",
                   "[" + std::to_string(context * kHiddenSize) + ", 2048, 1]",
                   "float16", false);
    }
    metadata << "  ]\n}\n";
    metadata.close();
  } else {
    Summary summary{};
    if (measure_cuda_event(launch, warmup, iterations, &summary) != 0) {
      return 2;
    }
    CUDA_CHECK(cudaGetLastError());
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int driver_version = 0;
    int runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::ofstream measurement(native_measurement_output);
    if (!measurement) {
      std::fprintf(stderr, "cannot open native measurement output: %s\n",
                   native_measurement_output.c_str());
      return 2;
    }
    const char* kernel_name = operator_name == "token_embedding"
                                  ? "token_embedding_gather"
                                  : "residual_add_fp16";
    measurement
        << "{\n"
        << "  \"schema_version\": \"hetero-p17-sealed-native-operator/v1\",\n"
        << "  \"model_spec_name\": \"TinyLlama-1.1B\",\n"
        << "  \"checkpoint_revision\": \"" << kRevision << "\",\n"
        << "  \"operator_type\": \"" << operator_name << "\",\n"
        << "  \"phase\": \"prefill\",\n"
        << "  \"batch_size\": 1,\n"
        << "  \"context_length\": " << context << ",\n"
        << "  \"dtype\": \"fp16\",\n"
        << "  \"device\": {\"name\": \"" << properties.name
        << "\", \"compute_capability\": \"" << properties.major << "."
        << properties.minor << "\", \"multiprocessors\": "
        << properties.multiProcessorCount << ", \"global_memory_bytes\": "
        << static_cast<unsigned long long>(properties.totalGlobalMem) << "},\n"
        << "  \"software\": {\"cuda_driver_version\": " << driver_version
        << ", \"cuda_runtime_version\": " << runtime_version
        << ", \"cudart_compile_version\": " << CUDART_VERSION << "},\n"
        << "  \"protocol\": {\"warmup_iterations\": " << warmup
        << ", \"measured_iterations\": " << iterations
        << ", \"timer\": \"cuda_event_per_iteration\", "
           "\"synchronization\": \"stop_event_synchronize_each_iteration\", "
           "\"statistic\": \"median\"},\n"
        << "  \"launch\": {\"kernel_name\": \"" << kernel_name
        << "\", \"grid\": [" << grid_blocks
        << ", 1, 1], \"block\": [256, 1, 1], "
           "\"dynamic_shared_memory_bytes\": 0, \"target_sm\": 86},\n"
        << "  \"measurement\": ";
    write_summary(measurement, summary);
    measurement
        << ",\n  \"measurement_scope\": \"native_rtx3070_local_vram\",\n"
        << "  \"performance_eligible\": false\n"
        << "}\n";
    measurement.close();
  }

  if (token_ids != nullptr) {
    CUDA_CHECK(cudaFree(token_ids));
  }
  CUDA_CHECK(cudaFree(first));
  if (second != nullptr) {
    CUDA_CHECK(cudaFree(second));
  }
  CUDA_CHECK(cudaFree(output));
  std::printf("%s %s complete: context=%d hidden=%d\n", operator_name.c_str(),
              metadata_output.empty() ? "native measurement" : "trace workload",
              context, kHiddenSize);
  return 0;
}
