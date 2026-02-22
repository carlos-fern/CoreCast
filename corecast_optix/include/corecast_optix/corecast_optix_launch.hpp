#pragma once

#include <optix.h>

#include <cuda_runtime.h>
#include <sutil/Exception.h>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_module.hpp"

namespace corecast_optix
{

class CoreCastOptixLaunch
{

    public:
    CoreCastOptixLaunch(std::shared_ptr<CoreCastOptixContext> context, Params& params, OptixPipeline pipeline, OptixShaderBindingTable& sbt): context_(context), params_(params){

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param_), sizeof(params_)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param_), &params_, sizeof(params), cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(pipeline, stream_, d_param_, sizeof(params), &sbt, params_.image_width, params_.image_height, /*depth=*/1));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param_)));
    }
    
    ~CoreCastOptixLaunch() = default;

    private:
    std::shared_ptr<CoreCastOptixContext> context_;
    CUstream stream_ =0;
    Params params_;
    CUdeviceptr d_param_=0;
};

}  // namespace corecast_optix
