#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_module.hpp"
#include "corecast_optix/corecast_optix_program_registry.hpp"

namespace corecast_optix
{

class CoreCastOptixPipeline
{
public:
  CoreCastOptixPipeline(std::shared_ptr<CoreCastOptixContext> context, 
                        std::shared_ptr<CoreCastOptixProgramRegistry> program_registry, 
                        std::shared_ptr<CoreCastOptixModule> module, 
                        const OptixPipelineLinkOptions& link_options, 
                        const std::vector<std::string>& program_names);
  ~CoreCastOptixPipeline();
  OptixPipeline get_pipeline() const { return pipeline_; }

private:
  std::shared_ptr<CoreCastOptixContext> context_;
  std::shared_ptr<CoreCastOptixProgramRegistry> program_registry_;
  std::shared_ptr<CoreCastOptixModule> module_;
  OptixPipelineLinkOptions link_options_;
  std::vector<std::string> program_names_;
  OptixPipeline pipeline_;
};

}  // namespace corecast_optix
