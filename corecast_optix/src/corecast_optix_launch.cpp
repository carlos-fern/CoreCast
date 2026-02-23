#include "corecast_optix/corecast_optix_launch.hpp"

#include <optix_stubs.h>

namespace corecast_optix
{

CoreCastOptixLaunch::CoreCastOptixLaunch(std::shared_ptr<CoreCastOptixContext> context, Params& params, OptixPipeline pipeline, OptixShaderBindingTable& sbt): context_(context), params_(params){

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param_), sizeof(params_)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param_), &params_, sizeof(params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, stream_, d_param_, sizeof(params), &sbt, params_.image_width, params_.image_height, /*depth=*/1));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param_)));
}

void CoreCastOptixLaunch::wait_for_completion(){
    std::cout << "Waiting for completion" << std::endl;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace corecast_optix