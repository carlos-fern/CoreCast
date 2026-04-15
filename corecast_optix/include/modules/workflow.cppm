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

export namespace corecast::optix {

template <typename T>
concept ContextLogCallbackType = std::invocable<T, unsigned int, const char*, const char*, void*>;

// Ensures that the CRTP derived class implements all required configuration hooks.
template <typename T>
concept ImplementsWorkflowHooks = requires(T t) {
  { t.define_modules() } -> std::same_as<void>;
  { t.define_programs() } -> std::same_as<void>;
  { t.define_pipelines() } -> std::same_as<void>;
  { t.define_gases() } -> std::same_as<void>;
  { t.define_sbts() } -> std::same_as<void>;
  { t.define_launches() } -> std::same_as<void>;
};

template <typename SpecificWorkflow>
class BaseWorkflow {
 public:
  BaseWorkflow(std::shared_ptr<corecast::optix::CoreCastOptix> optix, std::optional<WorkflowOptions> workflow_options)
      : program_registry_(std::make_shared<CoreCastOptixProgramRegistry>(context_)) {
    build_workflow();
  };
  ~BaseWorkflow() {};

  auto load_data(auto data) { return static_cast<SpecificWorkflow&>(*this).load_data(data); }

  void process() { static_cast<SpecificWorkflow&>(*this).process(); }

  auto get_result() { return static_cast<SpecificWorkflow&>(*this).get_result(); }

  void register_result_cb();

  void build_workflow() {
    // We static_assert inside a method so the derived class is fully defined
    // by the time the compiler evaluates the concept constraint.
    static_assert(ImplementsWorkflowHooks<SpecificWorkflow>, 
        "\nCRTP Error: Your Workflow class is missing one or more required hooks. "
        "You must implement: define_modules(), define_programs(), define_pipelines(), "
        "define_gases(), define_sbts(), and define_launches(). (They can be empty lists).");

    init_context();
    setup_modules();
    setup_programs();
    setup_pipelines();
    setup_gases();
    setup_sbts();
    setup_launches();
  }

 protected:
  std::shared_ptr<corecast::optix::CoreCastOptix> optix_;
  std::shared_ptr<CoreCastOptixContext> context_;
  std::shared_ptr<CoreCastOptixProgramRegistry> program_registry_;
  std::unordered_map<std::string, CoreCastOptixPipeline> pipelines_;
  std::unordered_map<std::string, CoreCastOptixModule> modules_;
  std::unordered_map<std::string, CoreCastOptixGAS> gases_;
  std::unordered_map<std::string, OptixTraversableHandle> gas_handles_;
  std::unordered_map<std::string, CoreCastOptixTraceSBT> sbts_;

  void init_context() {
    check_cuda(cudaFree(0), "cudaFree(0)");
    context_ = std::make_shared<CoreCastOptixContext>();

    if (context_->options_.logCallbackFunction == nullptr) {
      std::puts("Setting log callback function to default");
      // Note: context_log_cb signature has changed, alignment required here
    }

    std::puts("Initializing OptiX");
    check_optix(optixInit(), "optixInit()");
    std::puts("Creating OptiX device context");
    check_optix(optixDeviceContextCreate(context_->cuCtx_, &context_->options_, &context_->device_context_),
                "optixDeviceContextCreate()");
  }

  void add_module(const std::string& name, CoreCastOptixModule& module) { modules_.emplace(name, module); }

  void add_pipeline(const std::string& name, CoreCastOptixPipeline& pipeline) { pipelines_.emplace(name, pipeline); }

  void add_sbt(const std::string& name, const CoreCastOptixTraceSBT& sbt) { sbts_.emplace(name, sbt); }

  // Step 2:Setup module
  void setup_modules() {
    auto derived = static_cast<SpecificWorkflow&>(*this);

    derived.define_modules();

    for (auto& [name, module] : modules_) {
      if (module.ptx_path_.has_value()) {  // Custom module
        std::puts("Loading custom module");
        std::vector<char> ptx = read_file_bytes(module.ptx_path_.value());

        OPTIX_CHECK_LOG(optixModuleCreate(context_->device_context_, &module.module_compile_options_,
                                          &module.pipeline_compile_options_, ptx.data(), ptx.size(), LOG, &LOG_SIZE,
                                          &module.module_));

      } else {  // Builtin module
        std::puts("Loading builtin module");
        OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context_->device_context_, &module.module_compile_options_,
                                                &module.pipeline_compile_options_, &module.builtin_is_options_,
                                                &module.module_));
      }
    }
  }

  void setup_programs() {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    derived.define_programs();
  }

  // Step 4: Setup pipelines
  void setup_pipelines() {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    derived.define_pipelines();

    for (auto& [name, pipeline] : pipelines_) {
      std::vector<OptixProgramGroup> groups = program_registry_->get_program_groups(pipeline.program_names_);

      OPTIX_CHECK_LOG(optixPipelineCreate(context_->device_context_, &pipeline.compile_options_,
                                          &pipeline.link_options_, groups.data(), static_cast<uint32_t>(groups.size()),
                                          LOG, &LOG_SIZE, &pipeline.pipeline_));
    }
  }

  // Step 5 setup GAS
  void setup_gases() {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    derived.define_gases();

    for (auto& [name, gas] : gases_) {
      assert(gas.build_inputs_.empty());

      OPTIX_CHECK(optixAccelComputeMemoryUsage(context_->device_context_, &gas.build_options_, gas.build_inputs_.data(),
                                               static_cast<unsigned int>(gas.build_inputs_.size()),
                                               &gas.buffer_sizes_));

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.output_buffer_), gas.buffer_sizes_.outputSizeInBytes));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.temp_buffer_), gas.buffer_sizes_.tempSizeInBytes));
      OptixResult result = optixAccelBuild(
          context_->device_context_, gas.stream_, &gas.build_options_, gas.build_inputs_.data(),
          static_cast<unsigned int>(gas.build_inputs_.size()), gas.temp_buffer_, gas.buffer_sizes_.tempSizeInBytes,
          gas.output_buffer_, gas.buffer_sizes_.outputSizeInBytes, &gas.output_handle_, nullptr, 0);

      if (result != OPTIX_SUCCESS) {
        throw std::runtime_error("Failed to build acceleration structure");
      }

      gas_handles_.emplace(name, gas.output_handle_);
    }
  }

  // Step 6 : Setup SBT

  void setup_sbts() {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    derived.define_sbts();
  }

  std::unordered_map<std::string, CoreCastOptixLaunch> launches_;

  void add_launch(const std::string& name, const CoreCastOptixLaunch& launch) {
    launches_.emplace(name, launch);
  }

  // Step 7: Allocate launch buffers (called by build_workflow)
  void setup_launches() {
    auto derived = static_cast<SpecificWorkflow&>(*this);
    derived.define_launches();
  }

  // Execution Step: Actually launch a specific pipeline (called by derived process())
  template <typename ParamsType>
  void execute_launch(const std::string& launch_name, const std::string& pipeline_name, const std::string& sbt_name, const ParamsType& host_params, unsigned int width, unsigned int height, unsigned int depth = 1) {
    
    // Default to stream 0 if launch object wasn't specifically defined
    CUstream stream = 0; 
    
    const void* launch_param_ptr = static_cast<const void*>(&host_params);
    const size_t launch_param_size = sizeof(ParamsType);
    
    CUdeviceptr d_param = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), launch_param_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), launch_param_ptr, launch_param_size, cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipelines_.at(pipeline_name).pipeline_, stream, d_param, launch_param_size, &sbts_.at(sbt_name).get_sbt(), width, height, depth));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
  }

  // Default log callback for the context
  void context_log_cb(ContextLogCallbackType auto log_cb) {}
};

}  // namespace corecast::optix