#pragma once

#include <optix.h>

#include <cuda_runtime.h>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_module.hpp"

namespace corecast_optix {

struct ICoreCastOptixLaunch {
  virtual ~ICoreCastOptixLaunch() = default;
  virtual void wait_for_completion() = 0;
  virtual CUstream get_stream() const = 0;
};

template <typename ParamsType>
class CoreCastOptixLaunch final : public ICoreCastOptixLaunch {

public:
  CoreCastOptixLaunch(std::shared_ptr<CoreCastOptixContext> context,
                      const ParamsType &params, OptixPipeline pipeline,
                      OptixShaderBindingTable &sbt)
      : context_(context), params_(params) {

    const void *launch_param_ptr = static_cast<const void *>(&params_);
    const size_t launch_param_size = sizeof(ParamsType);

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_param_), launch_param_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_param_), launch_param_ptr,
                          launch_param_size, cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<ParamsType, Params>) {
      OPTIX_CHECK(optixLaunch(pipeline, stream_, d_param_, launch_param_size,
                              &sbt, params_.image_width, params_.image_height,
                              /*depth=*/1));
    } else if constexpr (std::is_same_v<ParamsType, PointCloudParams>) {
      OPTIX_CHECK(optixLaunch(pipeline, stream_, d_param_, launch_param_size,
                              &sbt, params_.num_points, 1,/*depth=*/1));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param_)));
  }

  ~CoreCastOptixLaunch() = default;

  void wait_for_completion() {
    std::cout << "Waiting for completion" << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  CUstream get_stream() const { return stream_; }

private:
  std::shared_ptr<CoreCastOptixContext> context_;
  CUstream stream_ = 0;
  ParamsType params_;
  CUdeviceptr d_param_ = 0;
};

} // namespace corecast_optix
