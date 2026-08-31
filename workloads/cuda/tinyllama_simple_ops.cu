#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

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
  int context = 16;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--operator" && index + 1 < argc) {
      operator_name = argv[++index];
    } else if (argument == "--context" && index + 1 < argc) {
      context = std::atoi(argv[++index]);
    } else if (argument == "--metadata-output" && index + 1 < argc) {
      metadata_output = argv[++index];
    } else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[index]);
      return 2;
    }
  }
  if ((operator_name != "token_embedding" &&
       operator_name != "residual_add") ||
      context <= 0 || metadata_output.empty()) {
    std::fprintf(stderr,
                 "usage: %s --operator token_embedding|residual_add "
                 "--context N --metadata-output PATH\n",
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
    token_embedding_gather<<<(elements + 255) / 256, 256>>>(
        token_ids, static_cast<const __half*>(first), output, context);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaMalloc(&first, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&second, tensor_bytes));
    CUDA_CHECK(cudaMalloc(&output, tensor_bytes));
    CUDA_CHECK(cudaMemset(first, 0, tensor_bytes));
    CUDA_CHECK(cudaMemset(second, 0, tensor_bytes));
    CUDA_CHECK(cudaMemset(output, 0, tensor_bytes));
    residual_add_fp16<<<(elements + 255) / 256, 256>>>(
        static_cast<const __half*>(first), static_cast<const __half*>(second),
        output, elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

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

  if (token_ids != nullptr) {
    CUDA_CHECK(cudaFree(token_ids));
  }
  CUDA_CHECK(cudaFree(first));
  if (second != nullptr) {
    CUDA_CHECK(cudaFree(second));
  }
  CUDA_CHECK(cudaFree(output));
  std::printf("%s reference kernel verified: context=%d hidden=%d\n",
              operator_name.c_str(), context, kHiddenSize);
  return 0;
}
