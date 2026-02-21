#pragma once

#include <optix.h>

#include <cuda_runtime.h>

#include "corecast_optix/corecast_optix_context.hpp"

namespace corecast_optix
{

class CoreCastOptixModule
{
public:
  CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context);
  ~CoreCastOptixModule();

  OptixModule get_module() const { return module_; }
  OptixPipelineCompileOptions get_pipeline_compile_options() const { return pipeline_compile_options_; }
  OptixModuleCompileOptions get_module_compile_options() const { return module_compile_options_; }
  
 private:
  std::shared_ptr<CoreCastOptixContext> context_;
  OptixModule module_;
  OptixPipelineCompileOptions pipeline_compile_options_;
  OptixModuleCompileOptions module_compile_options_;
};

}  // namespace corecast_optix