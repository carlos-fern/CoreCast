#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <sutil/Exception.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace corecast_optix
{


class CoreCastOptixAccel
{
public:
  /**
  * @brief Constructor for the CoreCastOptixAccel class.
  */
  CoreCastOptixAccel(OptixDeviceContext optixContext,
                     const OptixAccelBuildOptions& build_options,
                     const std::vector<OptixBuildInput>& build_inputs,
                     CUstream stream)
      : optixContext_(optixContext),
        stream_(stream),
        build_options_(build_options),
        build_inputs_(build_inputs) {

    if (build_inputs_.empty()) {
      throw std::runtime_error("CoreCastOptixAccel requires at least one OptixBuildInput");
    }

    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optixContext_, &build_options_, build_inputs_.data(),
        static_cast<unsigned int>(build_inputs_.size()), &buffer_sizes_));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output_buffer_),
                          buffer_sizes_.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_buffer_),
                          buffer_sizes_.tempSizeInBytes));
  }

  ~CoreCastOptixAccel() {
    if (output_buffer_ != 0) {
      CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(output_buffer_)));
    }
    if (temp_buffer_ != 0) {
      CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(temp_buffer_)));
    }
  }

  OptixTraversableHandle build_acceleration_structure(){


    OptixResult result = optixAccelBuild(
      optixContext_,
      stream_,
      &build_options_,
      build_inputs_.data(),
      static_cast<unsigned int>(build_inputs_.size()),
      temp_buffer_,
      buffer_sizes_.tempSizeInBytes,
      output_buffer_,
      buffer_sizes_.outputSizeInBytes,
      &output_handle_,
      nullptr,
      0
    );

    if (result != OPTIX_SUCCESS) {
      throw std::runtime_error("Failed to build acceleration structure");
    }

    return output_handle_;
  }

  OptixTraversableHandle get_output_handle() const { return output_handle_; }


private:
  OptixDeviceContext optixContext_;
  CUstream stream_;
  OptixAccelBuildOptions build_options_;
  std::vector<OptixBuildInput> build_inputs_;
  OptixAccelBufferSizes buffer_sizes_ = {};
  CUdeviceptr output_buffer_ = 0;
  CUdeviceptr temp_buffer_ = 0;
  OptixTraversableHandle output_handle_ = 0;

};



}  // namespace corecast_optix