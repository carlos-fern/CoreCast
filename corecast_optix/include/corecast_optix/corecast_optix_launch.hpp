#pragma once

#include <optix.h>

#include <cuda_runtime.h>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_module.hpp"

namespace corecast_optix {

class CoreCastOptixLaunch {

public:
  CoreCastOptixLaunch(std::shared_ptr<CoreCastOptixContext> context,
                      Params &params, OptixPipeline pipeline,
                      OptixShaderBindingTable &sbt);

  ~CoreCastOptixLaunch() = default;

  void wait_for_completion();

private:
  std::shared_ptr<CoreCastOptixContext> context_;
  CUstream stream_ = 0;
  Params params_;
  CUdeviceptr d_param_ = 0;
};

} // namespace corecast_optix
