#include "corecast_optix/corecast_optix.hpp"

namespace corecast_optix
{

CoreCastOptix::CoreCastOptix(CUcontext context_id, OptixDeviceContextOptions& options)
{
    context_ = std::make_shared<CoreCastOptixContext>(context_id, options);
    program_registry_ = std::make_shared<CoreCastOptixProgramRegistry>(context_);
}


void CoreCastOptix::create_module(std::string &module_name, OptixPipelineCompileOptions& pipeline_compile_options, OptixModuleCompileOptions& module_compile_options){
    modules_[module_name] = std::make_shared<CoreCastOptixModule>(context_, pipeline_compile_options, module_compile_options);
}

void CoreCastOptix::add_program_to_module(std::string &module_name, CoreCastProgram& program){
    program_registry_->register_program(program, *modules_[module_name]);
}

void CoreCastOptix::build_pipeline(std::string &pipeline_name, std::vector<std::string> program_names, OptixPipelineLinkOptions& link_options){
    pipelines_[pipeline_name] = std::make_shared<CoreCastOptixPipeline>(context_, program_registry_, modules_[program_names[0]], link_options, program_names);
}

void CoreCastOptix::launch_pipeline(std::string &pipeline_name, Params& params, std::string &sbt_name){
    auto pipeline_it = pipelines_.find(pipeline_name);
    if (pipeline_it == pipelines_.end()) {
        throw std::runtime_error("Pipeline not found: " + pipeline_name);
    }
    auto sbt_it = sbt_tables_.find(sbt_name);
    if (sbt_it == sbt_tables_.end()) {
        throw std::runtime_error("SBT not found: " + sbt_name);
    }

    launchers_[pipeline_name] = std::make_shared<CoreCastOptixLaunch>(context_, params, pipeline_it->second->get_pipeline(), sbt_it->second);    
}

}  // namespace corecast_optix