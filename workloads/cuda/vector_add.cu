#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t status = (call);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status)); \
      return 2;                                                                 \
    }                                                                           \
  } while (0)

__global__ void vector_add(const float* left, const float* right, float* output,
                           int count) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = left[index] + right[index];
  }
}

int main() {
  constexpr int count = 4096;
  constexpr std::size_t bytes = count * sizeof(float);
  std::vector<float> left(count, 1.25f);
  std::vector<float> right(count, 2.5f);
  std::vector<float> output(count, 0.0f);
  float *device_left = nullptr, *device_right = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_left, bytes));
  CUDA_CHECK(cudaMalloc(&device_right, bytes));
  CUDA_CHECK(cudaMalloc(&device_output, bytes));
  CUDA_CHECK(cudaMemcpy(device_left, left.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_right, right.data(), bytes, cudaMemcpyHostToDevice));
  vector_add<<<16, 256>>>(device_left, device_right, device_output, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_left));
  CUDA_CHECK(cudaFree(device_right));
  CUDA_CHECK(cudaFree(device_output));
  for (float value : output) {
    if (std::fabs(value - 3.75f) > 1e-6f) {
      std::fprintf(stderr, "verification failed\n");
      return 1;
    }
  }
  std::printf("vector_add verified: %d elements\n", count);
  return 0;
}
