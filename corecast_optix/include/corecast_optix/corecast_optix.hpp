#pragma once

#include <memory>
#include <string>
#include <type_traits>
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
    template <typename ParamsType>
    void launch_pipeline(std::string &pipeline_name, const ParamsType& params, std::string &sbt_name){
            auto pipeline_it = pipelines_.find(pipeline_name);
            if (pipeline_it == pipelines_.end()) {
                throw std::runtime_error("Pipeline not found: " + pipeline_name);
            }
            auto sbt_it = sbt_tables_.find(sbt_name);
            if (sbt_it == sbt_tables_.end()) {
                throw std::runtime_error("SBT not found: " + sbt_name);
            }
            launchers_[pipeline_name] = std::make_shared<CoreCastOptixLaunch<ParamsType>>(context_, params, pipeline_it->second->get_pipeline(), sbt_it->second);    
    }
    
    /**
    * @brief Get the result from a pipeline.
    * @param pipeline_name The name of the pipeline to get the result from.
    * @param params The parameters for the pipeline.
    * @param host_pixels The host pixels to store the result in.
    */

    template <typename ParamsType, typename ResultType>
    void get_result(std::string &pipeline_name, ParamsType& params, ResultType& host_data){
        using HostValueType = typename std::remove_reference_t<ResultType>::value_type;

        auto& launcher = launchers_[pipeline_name];

        CUDA_CHECK(cudaMemcpyAsync(
            host_data.data(),
            params.data,
            host_data.size() * sizeof(HostValueType),
            cudaMemcpyDeviceToHost,
            launcher->get_stream()
        ));
    
        launcher->wait_for_completion();
    
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.data)));
    }

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
    std::unordered_map<std::string, std::shared_ptr<ICoreCastOptixLaunch>> launchers_;
    std::unordered_map<std::string, OptixShaderBindingTable> sbt_tables_;
    std::unordered_map<std::string, std::shared_ptr<void>> sbts_;

};    

}  // namespace corecast_optix