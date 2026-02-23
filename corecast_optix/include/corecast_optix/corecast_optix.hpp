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
    CoreCastOptix(CUcontext context_id, OptixDeviceContextOptions& options);
    ~CoreCastOptix() = default;

    void create_module(std::string &module_name, OptixPipelineCompileOptions& pipeline_compile_options, OptixModuleCompileOptions& module_compile_options);
    
    void add_program_to_module(std::string &module_name, CoreCastProgram& program);
    void build_pipeline(std::string &pipeline_name, std::vector<std::string> program_names, OptixPipelineLinkOptions& link_options);
    void launch_pipeline(std::string &pipeline_name, Params& params, std::string &sbt_name);
    
    template <typename RecordType, typename DataType>
    void create_sbt(std::string &sbt_name, std::string &program_name, DataType& data){

        auto sbt = std::make_shared<CoreCastOptixSBT<RecordType, DataType>>(program_name, *program_registry_, data);
        sbt_tables_[sbt_name] = sbt->get_sbt();
        sbts_[sbt_name] = sbt;
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