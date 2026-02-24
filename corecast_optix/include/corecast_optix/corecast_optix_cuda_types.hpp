#pragma once

#include <cstdint>
#include <cuda/std/variant>
#include <cuda_runtime.h>
#include <type_traits>
#include <concepts>

namespace corecast_optix {

template <typename T>
concept ValidCUDAType = std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>;

template <typename UnprocessedType, typename ProcessedType> struct CUDABuffer {

  static_assert(ValidCUDAType<UnprocessedType>, "UnprocessedType must be a valid CUDA type that is trivially copyable and not a pointer");
  static_assert(ValidCUDAType<ProcessedType>, "ProcessedType must be a valid CUDA type that is trivially copyable and not a pointer");
  static_assert(sizeof(UnprocessedType) == sizeof(ProcessedType), 
            "Type sizes must match for direct byte-copying.");

  CUDABuffer(UnprocessedType *unprocessed_data, int num_elements, ProcessedType *processed_data)
      : device_ptr_(0), unprocessed_data_(unprocessed_data), num_elements_(num_elements), processed_data_(processed_data),
        size_in_bytes_(sizeof(UnprocessedType) * num_elements) {
            if (device_ptr_ == 0) {
                cudaError_t result = cudaMalloc(reinterpret_cast<void **>(&device_ptr_), size_in_bytes_);
                if (result != cudaSuccess) {
                  throw std::runtime_error("Failed to allocate device memory");
                }
              }
        }

  ~CUDABuffer() {
    if (device_ptr_ != 0) {
      cudaFree(reinterpret_cast<void *>(device_ptr_));
    }
  }

  CUdeviceptr device_ptr_;
  const size_t size_in_bytes_;
  UnprocessedType *unprocessed_data_;
  unsigned int num_elements_;
  ProcessedType *processed_data_;

  inline ProcessedType* get_device_ptr() const { return reinterpret_cast<ProcessedType*>(device_ptr_); }
  inline ProcessedType* get_processed_data_ptr() const { return processed_data_; }
  inline UnprocessedType* get_unprocessed_data_ptr() const { return unprocessed_data_; }
  inline size_t get_size_in_bytes() const { return size_in_bytes_; }
  inline unsigned int get_num_elements() const { return num_elements_; }

  void upload_to_device() {
    cudaError_t result = cudaMemcpy(reinterpret_cast<void *>(device_ptr_), unprocessed_data_, size_in_bytes_,
               cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to upload to device");
    }
  }

  void download_from_device() {
    cudaError_t result = cudaMemcpy(processed_data_, reinterpret_cast<void *>(device_ptr_), size_in_bytes_,
               cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to download from device");
    }   
  }
};

// Helloworld image types TODO: Remove this in favor of pointcloud types
struct Params {
  uchar4 *data;
  unsigned int image_width;
  unsigned int image_height;
};

struct RayGenData {
  float r, g, b;
  float circle_center_x;
  float circle_center_y;
  float circle_radius;
};

// Core pointcloud type
struct __attribute__((packed)) PointXYZI {
  float x;
  float y;
  float z;
  float intensity;
  uint16_t ring;
  double timestampOffset;
};

struct PointCloudParams {
  PointXYZI *data;
  uint32_t num_points;
  OptixTraversableHandle traversable;

  uint8_t *inlier_mask;     // represents the inlier mask for the plane
  float plane_z;            // represents the z-coordinate of the plane
  float distance_threshold; // represents the distance thrshold for the plane
};

// Global raygen data (The map)
struct PointCloudRayGenData {
  PointXYZI *data;
  unsigned int num_points;
};

struct PointCloudHitData {
  float intensity;
  uint16_t ring;
  double timestampOffset;
};

struct PointCloudMissData {
  float empty_color;
};

struct PointCloudRayPayload {
  PointXYZI point;
};

} // namespace corecast_optix
