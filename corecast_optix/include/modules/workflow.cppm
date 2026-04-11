module;

#include <optix.h>
#include <optix_stubs.h>

#include <concepts>
#include <corecast_optix/corecast_optix_types.hpp>
#include <corecast_optix/corecast_optix_utils.hpp>
#include <memory>
#include <optional>

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
    char LOG[2048];
    size_t LOG_SIZE = sizeof(LOG);

    if (module.ptx_path_.has_value()) {  // Custom module
      std::vector<char> ptx = read_file_bytes(module.ptx_path_.value());

      OPTIX_CHECK_LOG(optixModuleCreate(context_.device_context_, &module.module_compile_options_,
                                        &module.pipeline_compile_options_, ptx.data(), ptx.size(), LOG, &LOG_SIZE,
                                        &module.module_));
    } else {  // Builtin module
      OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context_.device_context_, &module.module_compile_options_,
                                              &module.pipeline_compile_options_, &module.builtin_is_options_,
                                              &module.module_));
    }
  }

  // Step 4: Setup pipelines
  void setup_pipelines(CoreCastOptixPipeline& pipeline) {
    char LOG[2048];
    size_t LOG_SIZE = sizeof(LOG);

    auto derived = static_cast<ActualWorkflow&>(*this);
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
  template <typename RecordType, typename DataType>
  void setup_sbt(CoreCastOptixSBT<RecordType, DataType>& sbt) {
    sbt.raygen_host_record_ptr_ = std::make_unique<RecordType>();
    sbt.miss_host_record_ptr_ = std::make_unique<RecordType>();
    sbt.hitgroup_host_record_ptr_ = std::make_unique<RecordType>();

    // sbt.raygen_host_record_ptr_->data = raygen_data;
    // sbt.miss_host_record_ptr_->data = miss_data;
    // sbt.hitgroup_host_record_ptr_->data = hitgroup_data;
    /*
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry_.get_program_group(raygen_program_name),
                                         sbt.raygen_host_record_ptr_.get()));
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry_.get_program_group(miss_program_name),
                                         sbt.miss_host_record_ptr_.get()));
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry_.get_program_group(hitgroup_program_name),
                                         sbt.hitgroup_host_record_ptr_.get()));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.raygen_device_ptr_), sizeof(RecordType)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.miss_device_ptr_), sizeof(RecordType)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.hitgroup_device_ptr_), sizeof(RecordType)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.raygen_device_ptr_), sbt.raygen_host_record_ptr_.get(),
                          sizeof(RecordType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.miss_device_ptr_), sbt.miss_host_record_ptr_.get(),
                          sizeof(RecordType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.hitgroup_device_ptr_), sbt.hitgroup_host_record_ptr_.get(),
                          sizeof(RecordType), cudaMemcpyHostToDevice));

    sbt.sbt_.raygenRecord = sbt.raygen_device_ptr_;
    sbt.sbt_.missRecordBase = sbt.miss_device_ptr_;
    sbt.sbt_.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(RecordType));
    sbt.sbt_.missRecordCount = 1;
    sbt.sbt_.hitgroupRecordBase = sbt.hitgroup_device_ptr_;
    sbt.sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(RecordType));
    sbt.sbt_.hitgroupRecordCount = 1;

    */
  }

  // Step 7 : setup launch
  void setup_launch(CoreCastOptixLaunch& launch) {
    auto derived = static_cast<ActualWorkflow&>(*this);
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