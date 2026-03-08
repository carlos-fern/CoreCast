#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_module.hpp"
#include "corecast_optix/corecast_optix_program_registry.hpp"

namespace corecast::optix
{

class CoreCastOptixPipeline
{
public:
  /**
  * @brief Constructor for the CoreCastOptixPipeline class.
  * @param context The context for the pipeline.
  * @param program_registry The program registry for the pipeline.
  * @param module The module for the pipeline.
  * @param link_options The options for the pipeline link.
  * @param program_names The names of the programs for the pipeline.
  */
  CoreCastOptixPipeline(std::shared_ptr<CoreCastOptixContext> context, 
                        std::shared_ptr<CoreCastOptixProgramRegistry> program_registry, 
                        std::shared_ptr<CoreCastOptixModule> module, 
                        const OptixPipelineLinkOptions& link_options, 
                        const std::vector<std::string>& program_names);

  /**
  * @brief Destructor for the CoreCastOptixPipeline class.
  */
  ~CoreCastOptixPipeline();

  /**
  * @brief Get the pipeline.
  */
  OptixPipeline get_pipeline() const { return pipeline_; }

private:
  std::shared_ptr<CoreCastOptixContext> context_;
  std::shared_ptr<CoreCastOptixProgramRegistry> program_registry_;
  std::shared_ptr<CoreCastOptixModule> module_;
  OptixPipelineLinkOptions link_options_;
  std::vector<std::string> program_names_;
  OptixPipeline pipeline_;
};

}  // namespace corecast::optix
