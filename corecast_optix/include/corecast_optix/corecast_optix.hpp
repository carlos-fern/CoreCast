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

    void build_pipeline(std::string &pipeline_name, std::vector<std::string> program_names, OptixPipelineLinkOptions& link_options){

        pipelines_[pipeline_name] = std::make_shared<CoreCastOptixPipeline>(context_, program_registry_, modules_[program_names[0]], link_options, program_names);
    }

    template <typename RecordType, typename DataType>
    void create_sbt(std::string &sbt_name, std::string &program_name, DataType& data){

        auto sbt = std::make_shared<CoreCastOptixSBT<RecordType, DataType>>(program_name, *program_registry_, data);
        sbt_tables_[sbt_name] = sbt->get_sbt();
        sbts_[sbt_name] = sbt;
    }

    void launch_pipeline(std::string &pipeline_name, Params& params, std::string &sbt_name){
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

    private:
    std::shared_ptr<CoreCastOptixContext> context_;
    std::unordered_map<std::string, std::shared_ptr<CoreCastOptixModule>> modules_;
    std::shared_ptr<CoreCastOptixProgramRegistry> program_registry_;
    std::unordered_map<std::string, std::shared_ptr<CoreCastOptixPipeline>> pipelines_;
    std::unordered_map<std::string, std::shared_ptr<CoreCastOptixLaunch>> launchers_;
    std::unordered_map<std::string, std::shared_ptr<void>> sbts_;
    std::unordered_map<std::string, OptixShaderBindingTable> sbt_tables_;
};    

}  // namespace corecast_optix