#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_utils.hpp"
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <iomanip>
#include <iostream>

namespace corecast::optix
{

CoreCastOptixContext::CoreCastOptixContext(CUcontext context_id, OptixDeviceContextOptions& options):
  options_(options)
{
  // Force CUDA runtime initialization so current context can be queried.
  check_cuda(cudaFree(0), "cudaFree(0)");
  cuCtx_ = context_id;

  std::cout << "Context ID: " << context_id << std::endl;
  std::cout << "Options: " << options_.logCallbackFunction << std::endl;

  if (options_.logCallbackFunction == nullptr){
    std::cout << "Setting log callback function to default" << std::endl;
    options_.logCallbackFunction = &context_log_cb;
  }

  std::cout << "Initializing OptiX" << std::endl;
  check_optix(optixInit(), "optixInit()");
  
  std::cout << "Creating OptiX device context" << std::endl;
  check_optix(optixDeviceContextCreate(cuCtx_, &options_, &this->context_), "optixDeviceContextCreate()");
}

CoreCastOptixContext::~CoreCastOptixContext()
{
  if (context_ != nullptr) {
    optixDeviceContextDestroy(context_);
  }
}

void CoreCastOptixContext::context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/)
{
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

}  // namespace corecast::optix
