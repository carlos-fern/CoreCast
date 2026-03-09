#include "corecast_optix/corecast_optix_module.hpp"

#include <optix_stubs.h>

#include "corecast_optix/corecast_optix_utils.hpp"

namespace corecast::optix {

CoreCastOptixModule::CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context,
                                         OptixPipelineCompileOptions& pipeline_compile_options,
                                         OptixModuleCompileOptions& module_compile_options, std::string& ptx_path)
    : context_(context),
      pipeline_compile_options_(pipeline_compile_options),
      module_compile_options_(module_compile_options) {
  std::vector<char> ptx = read_file_bytes(ptx_path);

  OPTIX_CHECK_LOG(optixModuleCreate(context_->get_context(), &module_compile_options_, &pipeline_compile_options_,
                                    ptx.data(), ptx.size(), LOG, &LOG_SIZE, &module_));
}

CoreCastOptixModule::CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context,
                                         OptixPipelineCompileOptions& pipeline_compile_options,
                                         OptixModuleCompileOptions& module_compile_options,
                                         OptixBuiltinISOptions& builtin_is_options)
    : context_(context),
      pipeline_compile_options_(pipeline_compile_options),
      module_compile_options_(module_compile_options),
      builtin_is_options_(builtin_is_options) {
  OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context_->get_context(), &module_compile_options_, &pipeline_compile_options_,
                                          &builtin_is_options_, &module_));
}

CoreCastOptixModule::~CoreCastOptixModule() = default;

}  // namespace corecast::optix
