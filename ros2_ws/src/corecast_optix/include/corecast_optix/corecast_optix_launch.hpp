#pragma once

#include <optix.h>

#include <cuda_runtime.h>

namespace corecast_optix
{

class CoreCastOptixLaunch
{

    public:
    CoreCastOptixLaunch(std::shared_ptr<CoreCastOptixContext> context, Params& params){

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param_), sizeof(params_)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param_), &params_, sizeof(params), cudaMemcpyHostToDevice));
        output_buffer_ = sutil::CUDAOutputBuffer<uchar4>(sutil::CUDAOutputBufferType::CUDA_DEVICE, params_.image_width, params_.image_height);

        output_buffer_.unmap();
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param_)));
    }
    
    ~CoreCastOptixLaunch() = default;

    private:
    CUstream stream_;
    Params params_;
    CUdeviceptr d_param_;
    sutil::CUDAOutputBuffer<uchar4> output_buffer_;
};  // namespace corecast_optix