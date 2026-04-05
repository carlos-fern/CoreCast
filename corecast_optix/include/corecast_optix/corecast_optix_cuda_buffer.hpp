#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_types.h>

#include <concepts>
#include <cstdint>
#include <cuda/std/variant>
#include <stdexcept>
#include <type_traits>

namespace corecast::optix {

template <typename T>
concept ValidCUDAType = std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>;

template <ValidCUDAType UnprocessedType, ValidCUDAType ProcessedType>
struct CUDABuffer {
  static_assert(ValidCUDAType<UnprocessedType>,
                "UnprocessedType must be a valid CUDA type that is trivially copyable and not a pointer");
  static_assert(ValidCUDAType<ProcessedType>,
                "ProcessedType must be a valid CUDA type that is trivially copyable and not a pointer");
  static_assert(sizeof(UnprocessedType) == sizeof(ProcessedType), "Type sizes must match for direct byte-copying.");

  CUDABuffer(UnprocessedType *unprocessed_data, int num_elements, bool upload_to_device = true)
      : device_ptr_(0),
        unprocessed_data_(unprocessed_data),
        num_elements_(num_elements),
        processed_data_(new ProcessedType[num_elements]),
        size_in_bytes_(sizeof(UnprocessedType) * num_elements) {
    if (device_ptr_ == 0) {
      cudaError_t result = cudaMalloc(reinterpret_cast<void **>(&device_ptr_), size_in_bytes_);
      if (result != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
      }
      if (upload_to_device) {
        upload_to_device_sync();
      }
    }
  }

  ~CUDABuffer() {
    delete[] processed_data_;
    if (device_ptr_ != 0) {
      cudaFree(reinterpret_cast<void *>(device_ptr_));
    }
  }

  CUDABuffer(const CUDABuffer &) = delete;
  CUDABuffer &operator=(const CUDABuffer &) = delete;

  CUdeviceptr device_ptr_;
  size_t size_in_bytes_;
  UnprocessedType *unprocessed_data_;
  unsigned int num_elements_;
  ProcessedType *processed_data_;

  inline ProcessedType *get_device_ptr() const { return reinterpret_cast<ProcessedType *>(device_ptr_); }
  inline ProcessedType *get_processed_data_ptr() const { return processed_data_; }
  inline UnprocessedType *get_unprocessed_data_ptr() const { return unprocessed_data_; }
  inline size_t get_size_in_bytes() const { return size_in_bytes_; }
  inline unsigned int get_num_elements() const { return num_elements_; }

  void upload_to_device_sync() {
    cudaError_t result =
        cudaMemcpy(reinterpret_cast<void *>(device_ptr_), unprocessed_data_, size_in_bytes_, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to upload to device");
    }
  }

  void download_from_device_sync() {
    cudaError_t result =
        cudaMemcpy(processed_data_, reinterpret_cast<void *>(device_ptr_), size_in_bytes_, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to download from device");
    }
  }

  void upload_to_device_async(CUstream stream) {
    cudaError_t result = cudaMemcpyAsync(reinterpret_cast<void *>(device_ptr_), unprocessed_data_, size_in_bytes_,
                                         cudaMemcpyHostToDevice, stream);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to upload to device");
    }
  }

  void download_from_device_async(CUstream stream) {
    cudaError_t result = cudaMemcpyAsync(processed_data_, reinterpret_cast<void *>(device_ptr_), size_in_bytes_,
                                         cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to download from device");
    }
  }
};




}  // namespace corecast::optix
