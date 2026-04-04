#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include <memory>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_cuda_types.hpp"

namespace corecast::optix {

class CoreCastOptixModule {
 public:
  /**
   * @brief Constructor for the CoreCastOptixModule class.
   * @param context The context for the module.
   * @param pipeline_compile_options The options for the pipeline compile.
   * @param module_compile_options The options for the module compile.
   */
  CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context,
                      OptixPipelineCompileOptions& pipeline_compile_options,
                      OptixModuleCompileOptions& module_compile_options, std::string& ptx_path);

  CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context,
                      OptixPipelineCompileOptions& pipeline_compile_options,
                      OptixModuleCompileOptions& module_compile_options, OptixBuiltinISOptions& builtin_is_options);

  /**
   * @brief Destructor for the CoreCastOptixModule class.
   */
  ~CoreCastOptixModule();

  /**
   * @brief Get the module.
   */
  OptixModule get_module() const { return module_; }

  /**
   * @brief Get the pipeline compile options.
   */
  const OptixPipelineCompileOptions& get_pipeline_compile_options() const { return pipeline_compile_options_; }

  /**
   * @brief Get the module compile options.
   */
  const OptixModuleCompileOptions& get_module_compile_options() const { return module_compile_options_; }

 private:
  std::shared_ptr<CoreCastOptixContext> context_;
  OptixModule module_;
  OptixPipelineCompileOptions pipeline_compile_options_;
  OptixModuleCompileOptions module_compile_options_;
  OptixBuiltinISOptions builtin_is_options_;
};

}  // namespace corecast::optix
