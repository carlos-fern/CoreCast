#pragma once

#include <optix.h>

#include <cuda_runtime.h>
#include <sutil/CUDAOutputBuffer.h>

namespace corecast_optix
{

class CoreCastOptixLaunch
{

    public:
    CoreCastOptixLaunch(std::shared_ptr<CoreCastOptixContext> context, Params& params, OptixPipeline pipeline, OptixShaderBindingTable& sbt): context_(context), params_(params), output_buffer_(sutil::CUDAOutputBufferType::CUDA_DEVICE, params_.image_width, params_.image_height){

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param_), sizeof(params_)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param_), &params_, sizeof(params), cudaMemcpyHostToDevice));


        OPTIX_CHECK(optixLaunch(pipeline, stream_, d_param_, sizeof(params), &sbt, params_.image_width, params_.image_height, /*depth=*/1));
        output_buffer_.unmap();
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param_)));
    }
    
    ~CoreCastOptixLaunch() = default;

    private:
    std::shared_ptr<CoreCastOptixContext> context_;
    CUstream stream_;
    Params params_;
    CUdeviceptr d_param_;
    sutil::CUDAOutputBuffer<uchar4> output_buffer_;
};

}  // namespace corecast_optix
