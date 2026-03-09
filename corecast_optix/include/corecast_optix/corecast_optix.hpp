#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "corecast_optix/corecast_optix_accel.hpp"
#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_launch.hpp"
#include "corecast_optix/corecast_optix_module.hpp"
#include "corecast_optix/corecast_optix_pipeline.hpp"
#include "corecast_optix/corecast_optix_program_registry.hpp"
#include "corecast_optix/corecast_optix_sbt.hpp"

namespace corecast::optix {

class CoreCastOptix {
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
   * @brief Expose underlying OptiX device context for advanced setup (e.g. GAS build).
   */
  OptixDeviceContext get_device_context() const { return context_->get_context(); }

  /**
   * @brief Create a module.
   * @param module_name The name of the module to create.
   * @param pipeline_compile_options The options for the pipeline compile.
   * @param module_compile_options The options for the module compile.
   * @param ptx_path The path to the PTX file to use for the module.
   */
  void create_module(std::string& module_name, OptixPipelineCompileOptions& pipeline_compile_options,
                     OptixModuleCompileOptions& module_compile_options, std::string& ptx_path);

  /**
   * @brief Create a builtin IS module.
   * @param module_name The name of the module to create.
   * @param pipeline_compile_options The options for the pipeline compile.
   * @param module_compile_options The options for the module compile.
   * @param builtin_is_options The options for the builtin IS module.
   * @param ptx_path The path to the PTX file to use for the module.
   */
  void create_module(std::string& module_name, OptixPipelineCompileOptions& pipeline_compile_options,
                     OptixModuleCompileOptions& module_compile_options, OptixBuiltinISOptions& builtin_is_options);

  /**
   * @brief Get a module.
   * @param module_name The name of the module to get.
   * @return The module.
   */
  OptixModule get_module(std::string& module_name) const { return modules_.at(module_name)->get_module(); }

  /**
   * @brief Add a program to a module.
   * @param module_name The name of the module to add the program to.
   * @param program The program to add to the module.
   */
  void add_program_to_module(std::string& module_name, CoreCastProgram& program);

  /**
   * @brief Build a pipeline.
   * @param pipeline_name The name of the pipeline to build.
   * @param program_names The names of the programs to build the pipeline with.
   * @param link_options The options for the pipeline link.
   */
  void build_pipeline(std::string& pipeline_name, std::vector<std::string> program_names,
                      OptixPipelineLinkOptions& link_options);

  /**
   * @brief Launch a pipeline.
   * @param pipeline_name The name of the pipeline to launch.
   * @param params The parameters for the pipeline.
   * @param sbt_name The name of the SBT to use for the pipeline.
   */
  template <typename ParamsType>
  void launch_pipeline(std::string& pipeline_name, const ParamsType& params, std::string& sbt_name) {
    auto pipeline_it = pipelines_.find(pipeline_name);
    if (pipeline_it == pipelines_.end()) {
      throw std::runtime_error("Pipeline not found: " + pipeline_name);
    }
    auto sbt_it = sbt_tables_.find(sbt_name);
    if (sbt_it == sbt_tables_.end()) {
      throw std::runtime_error("SBT not found: " + sbt_name);
    }
    launchers_[pipeline_name] = std::make_shared<CoreCastOptixLaunch<ParamsType>>(
        context_, params, pipeline_it->second->get_pipeline(), sbt_it->second);
  }

  /**
   * @brief Get the result from a pipeline.
   * @param pipeline_name The name of the pipeline to get the result from.
   * @param params The parameters for the pipeline.
   * @param host_pixels The host pixels to store the result in.
   */

  template <ValidCUDAType UnprocessedType, ValidCUDAType ProcessedType>
  ProcessedType* get_result(std::string& pipeline_name, CUDABuffer<UnprocessedType, ProcessedType>& host_buffer) {
    auto& launcher = launchers_[pipeline_name];

    host_buffer.download_from_device_async(launcher->get_stream());

    launcher->wait_for_completion();

    return host_buffer.get_processed_data_ptr();
  }

  /**
   * @brief Create a SBT.
   * @param sbt_name The name of the SBT to create.
   * @param program_name The name of the program to create the SBT for.
   * @param data The data to store in the SBT.
   */
  template <typename RecordType, typename DataType>
  void create_sbt(std::string& sbt_name, std::string& program_name, DataType& data) {
    auto sbt = std::make_shared<CoreCastOptixSBT<RecordType, DataType>>(program_name, *program_registry_, data);
    sbt_tables_[sbt_name] = sbt->get_sbt();
    sbts_[sbt_name] = sbt;
  }

  template <typename RaygenRecordType, typename RaygenDataType, typename MissRecordType, typename MissDataType,
            typename HitgroupRecordType, typename HitgroupDataType>
  void create_trace_sbt(std::string& sbt_name, std::string& raygen_program_name, std::string& miss_program_name,
                        std::string& hitgroup_program_name, const RaygenDataType& raygen_data,
                        const MissDataType& miss_data, const HitgroupDataType& hitgroup_data) {
    auto sbt = std::make_shared<CoreCastOptixTraceSBT<RaygenRecordType, RaygenDataType, MissRecordType, MissDataType,
                                                      HitgroupRecordType, HitgroupDataType>>(
        raygen_program_name, miss_program_name, hitgroup_program_name, *program_registry_, raygen_data, miss_data,
        hitgroup_data);
    sbt_tables_[sbt_name] = sbt->get_sbt();
    sbts_[sbt_name] = sbt;
  }

  void create_acceleration_structure(std::string& acceleration_structure_name, OptixAccelBuildOptions& build_options,
                                     const std::vector<OptixBuildInput>& build_inputs) {
    auto acceleration_structure =
        std::make_shared<CoreCastOptixAccel>(context_->get_context(), build_options, build_inputs, stream_);
    acceleration_structures_[acceleration_structure_name] = acceleration_structure;
  }

 private:
  std::shared_ptr<CoreCastOptixContext> context_;
  std::unordered_map<std::string, std::shared_ptr<CoreCastOptixModule>> modules_;
  std::shared_ptr<CoreCastOptixProgramRegistry> program_registry_;
  std::unordered_map<std::string, std::shared_ptr<CoreCastOptixPipeline>> pipelines_;
  std::unordered_map<std::string, std::shared_ptr<ICoreCastOptixLaunch>> launchers_;
  std::unordered_map<std::string, OptixShaderBindingTable> sbt_tables_;
  std::unordered_map<std::string, std::shared_ptr<void>> sbts_;
  std::unordered_map<std::string, std::shared_ptr<CoreCastOptixAccel>> acceleration_structures_;
  CUstream stream_ = 0;
};

}  // namespace corecast::optix