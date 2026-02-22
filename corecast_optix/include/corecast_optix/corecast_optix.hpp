#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_module.hpp"
#include "corecast_optix/corecast_optix_program_registry.hpp"
#include "corecast_optix/corecast_optix_pipeline.hpp"
#include "corecast_optix/corecast_optix_launch.hpp"
#include "corecast_optix/corecast_optix_sbt.hpp"


namespace corecast_optix
{

    
class CoreCastOptix
{
    public:
    CoreCastOptix(){
        context_ = std::make_shared<CoreCastOptixContext>();
        program_registry_ = std::make_shared<CoreCastOptixProgramRegistry>(context_);
    }
    ~CoreCastOptix() = default;


    void create_module(std::string &module_name, OptixPipelineCompileOptions& pipeline_compile_options, OptixModuleCompileOptions& module_compile_options){

        modules_[module_name] = std::make_shared<CoreCastOptixModule>(context_, pipeline_compile_options, module_compile_options);
    }
    
    void add_program_to_module(std::string &module_name, CoreCastProgram& program){
        
        program_registry_->register_program(program, *modules_[module_name]);
    }

    void build_pipeline(std::string &pipeline_name, std::vector<std::string> &program_names, OptixPipelineLinkOptions& link_options){

        pipelines_[pipeline_name] = std::make_shared<CoreCastOptixPipeline>(context_, program_registry_, modules_[program_names[0]], link_options, program_names);
    }

    template <typename RecordType, typename DataType>
    void create_sbt(std::string &sbt_name, std::string &program_name, DataType& data){

        sbts_[sbt_name] = std::make_shared<CoreCastOptixSBT<RecordType, DataType>>(program_name, *program_registry_, data);
    }

    void launch_pipeline(std::string &pipeline_name, Params& params, std::string &sbt_name){

        launchers_[pipeline_name] = std::make_shared<CoreCastOptixLaunch>(context_, params, pipelines_[pipeline_name]->get_pipeline(), sbts_[sbt_name]);
    }

    private:
    std::shared_ptr<CoreCastOptixContext> context_;
    std::unordered_map<std::string, std::shared_ptr<CoreCastOptixModule>> modules_;
    std::shared_ptr<CoreCastOptixProgramRegistry> program_registry_;
    std::unordered_map<std::string, std::shared_ptr<CoreCastOptixPipeline>> pipelines_;
    std::unordered_map<std::string, std::shared_ptr<CoreCastOptixLaunch>> launchers_;
    std::unordered_map<std::string, std::shared_ptr<void>> sbts_;
};    

}  // namespace corecast_optix