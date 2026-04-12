module;

#include <optix.h>
#include <optix_stubs.h>

#include <concepts>
#include <corecast_optix/corecast_optix_types.hpp>
#include <corecast_optix/corecast_optix_utils.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

#include "corecast_optix/corecast_optix_cuda_buffer.hpp"

export module workflow;

// Forward declaration
namespace corecast::optix {
class CoreCastOptix;
}

export namespace corecast::optix {

template <typename T>
concept ContextLogCallbackType = std::invocable<T, unsigned int, const char*, const char*, void*>;

template <typename SpecificWorkflow>
class BaseWorkflow {
 public:
  BaseWorkflow(std::shared_ptr<CoreCastOptix> optix, std::optional<WorkflowOptions> workflow_options);
  ~BaseWorkflow();

  auto load_data(auto data) { return static_cast<SpecificWorkflow&>(*this).load_data(data); }

  void process() { static_cast<SpecificWorkflow&>(*this).process(); }

  auto get_result() { return static_cast<SpecificWorkflow&>(*this).get_result(); }

 protected:
  std::shared_ptr<CoreCastOptix> optix_;
  CoreCastOptixContext context_;
  CoreCastOptixProgramRegistry program_registry_;
  std::vector<std::string> program_names_;
  std::unordered_map<std::string, corecast::optix::CoreCastOptixTraceSBT> sbt_map_;

  // Step 1: Initialize the context

  void init_context(CoreCastOptixContext& context) {
    check_cuda(cudaFree(0), "cudaFree(0)");
    context_ = context;

    if (context_.options_.logCallbackFunction == nullptr) {
      std::puts("Setting log callback function to default");
      // Note: context_log_cb signature has changed, alignment required here
    }

    std::puts("Initializing OptiX");
    check_optix(optixInit(), "optixInit()");

    std::puts("Creating OptiX device context");
    check_optix(optixDeviceContextCreate(context_.cuCtx_, &context_.options_, &context_.device_context_),
                "optixDeviceContextCreate()");
  }

  // Step 2:Setup module
  void setup_module(CoreCastOptixModule& module) {
    if (module.ptx_path_.has_value()) {  // Custom module
      std::puts("Loading custom module");
      std::vector<char> ptx = read_file_bytes(module.ptx_path_.value());

      OPTIX_CHECK_LOG(optixModuleCreate(context_.device_context_, &module.module_compile_options_,
                                        &module.pipeline_compile_options_, ptx.data(), ptx.size(), LOG, &LOG_SIZE,
                                        &module.module_));

    } else {  // Builtin module
      std::puts("Loading builtin module");
      OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context_.device_context_, &module.module_compile_options_,
                                              &module.pipeline_compile_options_, &module.builtin_is_options_,
                                              &module.module_));
    }
  }

  // Step 4: Setup pipelines
  void setup_pipelines(CoreCastOptixPipeline& pipeline) {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    // Assuming program_names and pipeline_ are members or accessible via ActualWorkflow
    std::vector<OptixProgramGroup> groups = program_registry_.get_program_groups(derived.program_names_);

    OPTIX_CHECK_LOG(optixPipelineCreate(
        context_.device_context_, &derived.module_.pipeline_compile_options_, &derived.pipeline_link_options_,
        groups.data(), static_cast<uint32_t>(groups.size()), LOG, &LOG_SIZE, &derived.pipeline_.pipeline_));
  }

  // Step 5 setup GAS
  OptixTraversableHandle setup_gas(CoreCastOptixGAS& gas) {
    assert(gas.build_inputs_.empty());

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_.device_context_, &gas.build_options_, gas.build_inputs_.data(),
                                             static_cast<unsigned int>(gas.build_inputs_.size()), &gas.buffer_sizes_));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.output_buffer_), gas.buffer_sizes_.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.temp_buffer_), gas.buffer_sizes_.tempSizeInBytes));
    OptixResult result = optixAccelBuild(context_.device_context_, gas.stream_, &gas.build_options_,
                                         gas.build_inputs_.data(), static_cast<unsigned int>(gas.build_inputs_.size()),
                                         gas.temp_buffer_, gas.buffer_sizes_.tempSizeInBytes, gas.output_buffer_,
                                         gas.buffer_sizes_.outputSizeInBytes, &gas.output_handle_, nullptr, 0);

    if (result != OPTIX_SUCCESS) {
      throw std::runtime_error("Failed to build acceleration structure");
    }

    return gas.output_handle_;
  }

  // Step 6 : Setup SBT

  void create_sbt(std::string program_name, SbtRecordType auto& raygen_record, SbtRecordType auto& miss_record,
                  SbtRecordType auto& hitgroup_record) {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    sbt_map_[program_name] =
        CoreCastOptixTraceSBT(program_name, program_registry_, raygen_record, miss_record, hitgroup_record);
  }

  // Step 7 : setup launch
  void setup_launch(CoreCastOptixLaunch& launch) {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    /*
    const void* launch_param_ptr = static_cast<const void*>(&derived.launch_params_.params_);
    const size_t launch_param_size = sizeof(ParamsType);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&launch.d_param_), launch_param_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(launch.d_param_), launch_param_ptr, launch_param_size,
                          cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<ParamsType, Params>) {
      OPTIX_CHECK(optixLaunch(pipeline, launch.stream_, launch.d_param_, launch_param_size, &sbt, params_.image_width,
                              params_.image_height,
                              depth=1));
  }
  else if constexpr (std::is_same_v<ParamsType, PointCloudLaunchParams>) {
    OPTIX_CHECK(optixLaunch(pipeline, launch.stream_, launch.d_param_, launch_param_size, &sbt, params_.image_width,
                            params_.image_height, depth=1));
  }

  CUDA_CHECK(cudaStreamSynchronize(launch.stream_));
  CUDA_CHECK(cudaFree(reinterpret_cast<void*>(launch.d_param_)));
  */
  }

  // Default log callback for the context
  void context_log_cb(ContextLogCallbackType auto log_cb) {}
};

}  // namespace corecast::optix