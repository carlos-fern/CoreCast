#include "corecast_optix/corecast_optix.hpp"

namespace corecast_optix
{

CoreCastOptix::CoreCastOptix(CUcontext context_id, OptixDeviceContextOptions& options)
{
    context_ = std::make_shared<CoreCastOptixContext>(context_id, options);
    program_registry_ = std::make_shared<CoreCastOptixProgramRegistry>(context_);
}


void CoreCastOptix::create_module(std::string &module_name, OptixPipelineCompileOptions& pipeline_compile_options, OptixModuleCompileOptions& module_compile_options, std::string& ptx_path){
    modules_[module_name] = std::make_shared<CoreCastOptixModule>(context_, pipeline_compile_options, module_compile_options, ptx_path);
}

void CoreCastOptix::add_program_to_module(std::string &module_name, CoreCastProgram& program){
    program_registry_->register_program(program, *modules_[module_name]);
}

void CoreCastOptix::build_pipeline(std::string &pipeline_name, std::vector<std::string> program_names, OptixPipelineLinkOptions& link_options){
    pipelines_[pipeline_name] = std::make_shared<CoreCastOptixPipeline>(context_, program_registry_, modules_[program_names[0]], link_options, program_names);
}




}  // namespace corecast_optix