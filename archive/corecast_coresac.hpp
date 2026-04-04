#pragma once

#include <optix_types.h>

#include <corecast_optix/corecast_optix.hpp>

namespace corecast::processing {

class CoreCastCoreSac {
 public:
  CoreCastCoreSac(std::shared_ptr<corecast::optix::CoreCastOptix> optix) {
    aabb_buffer_ = std::make_unique<corecast::optix::CUDABuffer<OptixAabb, OptixAabb>>(1);
  }

  ~CoreCastCoreSac();

  void set_build_input() {
    build_input_ = {};
    build_input_.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_buffers_[0] = aabb_buffer_->get_device_ptr();
    primitive_flags_[0] = OPTIX_GEOMETRY_FLAG_NONE;
    build_input_.customPrimitiveArray.aabbBuffers = aabb_buffers_;
    build_input_.customPrimitiveArray.numPrimitives = 1;
    build_input_.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    build_input_.customPrimitiveArray.flags = primitive_flags_;
    build_input_.customPrimitiveArray.numSbtRecords = 1;
    build_input_.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    build_input_.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    build_input_.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;
  }

 private:
  OptixBuildInput build_input_;
  std::unique_ptr<corecast::optix::CUDABuffer<OptixAabb, OptixAabb>> aabb_buffer_;
  CUdeviceptr aabb_buffers_[1]{};
  uint32_t primitive_flags_[1]{};
};

}  // namespace corecast::processing