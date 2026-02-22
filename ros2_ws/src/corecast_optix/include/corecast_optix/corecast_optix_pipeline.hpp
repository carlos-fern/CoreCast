#pragma once

#include <optix.h>

#include <cuda_runtime.h>

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

private:
  std::shared_ptr<CoreCastOptixContext> context_;
  std::shared_ptr<CoreCastOptixProgramRegistry> program_registry_;
  std::shared_ptr<CoreCastOptixModule> module_;
  OptixPipelineLinkOptions link_options_;
  OptixPipeline pipeline_;
};

}  // namespace corecast_optix