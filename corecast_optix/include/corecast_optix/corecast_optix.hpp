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
    /**
    * @brief Constructor for the CoreCastOptix class.
    * @param context_id The CUDA context ID to use.
    * @param options The options for the OptixDeviceContext.
    */
    CoreCastOptix(CUcontext context_id, OptixDeviceContextOptions& options);

    /**
    * @brief Destructor for the CoreCastOptix class.
    */
    ~CoreCastOptix() = default;

    /**
    * @brief Create a module.
    * @param module_name The name of the module to create.
    * @param pipeline_compile_options The options for the pipeline compile.
    * @param module_compile_options The options for the module compile.
    */
    void create_module(std::string &module_name, OptixPipelineCompileOptions& pipeline_compile_options, OptixModuleCompileOptions& module_compile_options);
    
    /**
    * @brief Add a program to a module.
    * @param module_name The name of the module to add the program to.
    * @param program The program to add to the module.
    */
    void add_program_to_module(std::string &module_name, CoreCastProgram& program);

    /**
    * @brief Build a pipeline.
    * @param pipeline_name The name of the pipeline to build.
    * @param program_names The names of the programs to build the pipeline with.
    * @param link_options The options for the pipeline link.
    */
    void build_pipeline(std::string &pipeline_name, std::vector<std::string> program_names, OptixPipelineLinkOptions& link_options);

    /**
    * @brief Launch a pipeline.
    * @param pipeline_name The name of the pipeline to launch.
    * @param params The parameters for the pipeline.
    * @param sbt_name The name of the SBT to use for the pipeline.
    */
    void launch_pipeline(std::string &pipeline_name, Params& params, std::string &sbt_name);
    
    /**
    * @brief Get the result from a pipeline.
    * @param pipeline_name The name of the pipeline to get the result from.
    * @param params The parameters for the pipeline.
    * @param host_pixels The host pixels to store the result in.
    */
    void get_result(std::string &pipeline_name, corecast_optix::Params& params, std::vector<uchar4>& host_pixels);

    /**
    * @brief Create a SBT.
    * @param sbt_name The name of the SBT to create.
    * @param program_name The name of the program to create the SBT for.
    * @param data The data to store in the SBT.
    */
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