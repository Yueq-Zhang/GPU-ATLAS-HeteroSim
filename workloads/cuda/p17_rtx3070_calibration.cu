#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
      return 2;                                                                  \
    }                                                                            \
  } while (0)

namespace {

constexpr int kHiddenSize = 2048;
constexpr int kVocabSize = 32000;
constexpr int kContext = 16;

__global__ void empty_kernel() {}

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
int measure_synchronized_launch(int warmup, int iterations, Summary* result) {
  for (int index = 0; index < warmup; ++index) {
    empty_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  std::vector<double> values_fs;
  values_fs.reserve(iterations);
  for (int index = 0; index < iterations; ++index) {
    const auto start = std::chrono::steady_clock::now();
    empty_kernel<<<1, 1>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
    values_fs.push_back(static_cast<double>(elapsed_ns) * 1.0e6);
  }
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

}  // namespace

int main(int argc, char** argv) {
  std::string output;
  int warmup = 100;
  int iterations = 1000;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--output" && index + 1 < argc) {
      output = argv[++index];
    } else if (argument == "--warmup" && index + 1 < argc) {
      warmup = std::atoi(argv[++index]);
    } else if (argument == "--iterations" && index + 1 < argc) {
      iterations = std::atoi(argv[++index]);
    } else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", argv[index]);
      return 2;
    }
  }
  if (output.empty() || warmup < 0 || iterations <= 0) {
    std::fprintf(stderr,
                 "usage: %s --output FILE [--warmup N] [--iterations N]\n",
                 argv[0]);
    return 2;
  }

  int device = 0;
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  int driver_version = 0;
  int runtime_version = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

  const int elements = kContext * kHiddenSize;
  const std::size_t tensor_bytes =
      static_cast<std::size_t>(elements) * sizeof(__half);
  const std::size_t weight_bytes =
      static_cast<std::size_t>(kVocabSize) * kHiddenSize * sizeof(__half);
  std::int64_t* token_ids = nullptr;
  __half* weight = nullptr;
  __half* first = nullptr;
  __half* second = nullptr;
  __half* output_tensor = nullptr;
  CUDA_CHECK(cudaMalloc(&token_ids, kContext * sizeof(std::int64_t)));
  CUDA_CHECK(cudaMalloc(&weight, weight_bytes));
  CUDA_CHECK(cudaMalloc(&first, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&second, tensor_bytes));
  CUDA_CHECK(cudaMalloc(&output_tensor, tensor_bytes));
  CUDA_CHECK(cudaMemset(token_ids, 0, kContext * sizeof(std::int64_t)));
  CUDA_CHECK(cudaMemset(weight, 0, weight_bytes));
  CUDA_CHECK(cudaMemset(first, 0, tensor_bytes));
  CUDA_CHECK(cudaMemset(second, 0, tensor_bytes));
  CUDA_CHECK(cudaMemset(output_tensor, 0, tensor_bytes));

  Summary empty_event{};
  Summary synchronized_launch{};
  Summary embedding{};
  Summary residual{};
  Summary copy_32k{};
  if (measure_cuda_event([] { empty_kernel<<<1, 1>>>(); }, warmup, iterations,
                         &empty_event) != 0 ||
      measure_synchronized_launch(warmup, iterations, &synchronized_launch) != 0 ||
      measure_cuda_event(
          [&] {
            token_embedding_gather<<<(elements + 255) / 256, 256>>>(
                token_ids, weight, output_tensor, kContext);
          },
          warmup, iterations, &embedding) != 0 ||
      measure_cuda_event(
          [&] {
            residual_add_fp16<<<(elements + 255) / 256, 256>>>(
                first, second, output_tensor, elements);
          },
          warmup, iterations, &residual) != 0 ||
      measure_cuda_event(
          [&] {
            cudaMemcpyAsync(output_tensor, first, tensor_bytes,
                            cudaMemcpyDeviceToDevice);
          },
          warmup, iterations, &copy_32k) != 0) {
    return 2;
  }
  CUDA_CHECK(cudaGetLastError());

  std::ofstream stream(output);
  if (!stream) {
    std::fprintf(stderr, "cannot open output: %s\n", output.c_str());
    return 2;
  }
  stream << "{\n"
         << "  \"schema_version\": \"hetero-p17-native-calibration/v1\",\n"
         << "  \"measurement_scope\": "
            "\"native_rtx3070_local_vram_not_external_3ddram\",\n"
         << "  \"gpu\": {\"name\": \"" << properties.name
         << "\", \"compute_capability\": \"" << properties.major << "."
         << properties.minor << "\", \"multiprocessors\": "
         << properties.multiProcessorCount << ", \"clock_rate_khz\": "
         << properties.clockRate << ", \"memory_clock_rate_khz\": "
         << properties.memoryClockRate << ", \"memory_bus_width_bits\": "
         << properties.memoryBusWidth << ", \"global_memory_bytes\": "
         << static_cast<unsigned long long>(properties.totalGlobalMem) << "},\n"
         << "  \"software\": {\"cuda_driver_version\": " << driver_version
         << ", \"cuda_runtime_version\": " << runtime_version
         << ", \"cudart_compile_version\": " << CUDART_VERSION << "},\n"
         << "  \"protocol\": {\"warmup_iterations\": " << warmup
         << ", \"measured_iterations\": " << iterations
         << ", \"kernel_statistic\": \"cuda_event_per_iteration\", "
            "\"host_statistic\": \"steady_clock_kernel_plus_synchronize\"},\n"
         << "  \"measurements\": {\n"
         << "    \"empty_kernel_event\": ";
  write_summary(stream, empty_event);
  stream << ",\n    \"synchronized_launch\": ";
  write_summary(stream, synchronized_launch);
  stream << ",\n    \"token_embedding_ctx16_hidden2048\": ";
  write_summary(stream, embedding);
  stream << ",\n    \"residual_add_ctx16_hidden2048\": ";
  write_summary(stream, residual);
  stream << ",\n    \"d2d_copy_32768B\": ";
  write_summary(stream, copy_32k);
  const double seconds = copy_32k.median_fs / 1.0e15;
  stream << ",\n    \"d2d_copy_32768B_median_bandwidth_Bps\": "
         << std::llround(static_cast<double>(tensor_bytes) / seconds) << "\n"
         << "  },\n"
         << "  \"performance_qualification\": {\"eligible\": false, "
            "\"reason\": \"native local-VRAM measurements require matched "
            "native-memory simulation and do not calibrate the external link, "
            "logic-die gateway or 3D-DRAM\"}\n"
         << "}\n";
  stream.close();

  CUDA_CHECK(cudaFree(token_ids));
  CUDA_CHECK(cudaFree(weight));
  CUDA_CHECK(cudaFree(first));
  CUDA_CHECK(cudaFree(second));
  CUDA_CHECK(cudaFree(output_tensor));
  std::printf("P17 RTX 3070 native calibration written: %s\n", output.c_str());
  return 0;
}
