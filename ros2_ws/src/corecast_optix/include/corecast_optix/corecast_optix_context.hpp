#pragma once

#include <optix.h>

#include <cuda_runtime.h>

namespace corecast_optix
{

class CoreCastOptixContext
{
  OptixDeviceContext context_;
public:
  CoreCastOptixContext();
  ~CoreCastOptixContext();

  OptixDeviceContext get_context() const { return context_; }
  static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */);
  
};

}  // namespace corecast_optix
