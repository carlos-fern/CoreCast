#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <string>

namespace corecast_optix
{


class CoreCastOptixAccel
{
public:
  /**
  * @brief Constructor for the CoreCastOptixAccel class.
  */
  CoreCastOptixAccel(OptixDeviceContext& optixContext, OptixAccleBuildOptions& build_options, const std::vector<OptixBuildInput>& build_inputs, CUstream stream){

    optixContext_ = optixContext;
    stream_ = stream;
    build_options_ = build_options;
    build_inputs_ = build_inputs;
    OptixAccelBufferSizes bufferSizes = {};
    optixAccelComputeMemoryUsage( optixContext_, &build_options_, build_inputs_.data(), build_inputs_.size(), &bufferSizes );

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output_buffer_), bufferSizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_buffer_), bufferSizes.tempSizeInBytes));
  }

  OptixTraversableHandle build_acceleration_structure(){


    OptixResult result = optixAccelBuild(
      optixContext_,
      stream_,
      &build_options_,
      build_inputs_.data(),
      build_inputs_.size(),
      temp_buffer_,
      outputbufferSizes.tempSizeInBytes,
      output_buffer_,
      bufferSizes.outputSizeInBytes,
      &output_handle_,
      nullptr,
      0
    );

    if (result != OPTIX_SUCCESS) {
      throw std::runtime_error("Failed to build acceleration structure");
    }

    return output_handle_;
  }

 
  }



private:
  OptixDeviceContext optixContext_;
  CUstream stream_;
  OptixAccleBuildOptions build_options_;
  std::vector<OptixBuildInput> build_inputs_;
  CUdeviceptr output_buffer_ = 0;
  CUdeviceptr temp_buffer_ = 0;
  OptixTraversableHandle output_handle_ = 0;

};



}  // namespace corecast_optix