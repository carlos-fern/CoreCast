#include <corecast_optix/corecast_workflow.hpp>
#include <corecast_optix/corecast_optix.hpp>

namespace corecast::optix {

CoreCastWorkflow::CoreCastWorkflow(std::shared_ptr<CoreCastOptix> optix) : optix_(optix) {


init_context(context_);

}

CoreCastWorkflow::~CoreCastWorkflow() {

    if (context_.context_ != nullptr) {
        optixDeviceContextDestroy(context_.context_);
      }

      if (module_ != nullptr) {
        optixModuleDestroy(module_);
      }

      if (pipeline_ != nullptr) {
        optixPipelineDestroy(pipeline_);
      } 

      if (raygen_device_ptr_ != 0) {
        CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(raygen_device_ptr_)));
      }
      if (miss_device_ptr_ != 0) {
        CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(miss_device_ptr_)));
      }
      if (hitgroup_device_ptr_ != 0) {
        CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(hitgroup_device_ptr_)));
      }
}

void CoreCastWorkflow::init_context(CoreCastOptixContext& context) {
    
    check_cuda(cudaFree(0), "cudaFree(0)");
    context.device_context_id_ = context_id;
  
    std::puts("Context ID: %d", context_id);
    std::puts("Options: %p", context.options_.logCallbackFunction);

    if (context.options_.logCallbackFunction == nullptr) {
      std::puts("Setting log callback function to default");
      context.options_.logCallbackFunction = &CoreCastWorkflow::context_log_cb;
    }
  
    std::puts("Initializing OptiX");
    check_optix(optixInit(), "optixInit()");
  
    std::puts("Creating OptiX device context");
    check_optix(optixDeviceContextCreate(context.cuCtx_, &context.options_, &context.context_id), "optixDeviceContextCreate()");
}

void CoreCastWorkfFlow::setup_module(CoreCastOptixModule& module) {

  if (module.ptx_path_.has_value()) { // Custom module
    std::vector<char> ptx = read_file_bytes(module.ptx_path_.value());

    OPTIX_CHECK_LOG(optixModuleCreate(context.device_context_, &module.module_compile_options_, &module.pipeline_compile_options_,
    ptx.data(), ptx.size(), LOG, &LOG_SIZE, &module.module_));
  }
  else { // Builtin module
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context.device_context_, &module.module_compile_options_, &module.pipeline_compile_options_,
    &module.builtin_is_options_, &module.module_));
  }
}

void CoreCastWorkflow::setup_programs_and_program_groups() {

}

void CoreCastWorkflow::setup_pipelines() {
    std::vector<OptixProgramGroup> groups = program_registry_->get_program_groups(program_names);

    OPTIX_CHECK_LOG(optixPipelineCreate(context_.device_context_, &module_.pipeline_compile_options_, &pipeline_link_options_,
                                        groups.data(), static_cast<uint32_t>(groups.size()), LOG, &LOG_SIZE, &pipeline_));
}

OptixTraversableHandle CoreCastWorkflow::setup_gas(CoreCastOptixGAS& gas) {

    assert(gas.build_inputs_.empty());

    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_.device_context_, &gas.build_options_, gas.build_inputs_.data(),
                                              static_cast<unsigned int>(gas.build_inputs_.size()), &gas.buffer_sizes_));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.output_buffer_), gas.buffer_sizes_.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.temp_buffer_), gas.buffer_sizes_.tempSizeInBytes));
    OptixResult result =
    optixAccelBuild(context_.device_context_, gas.stream_, &gas.build_options_, gas.build_inputs_.data(),
                    static_cast<unsigned int>(gas.build_inputs_.size()), gas.temp_buffer_, gas.buffer_sizes_.tempSizeInBytes,
                    gas.output_buffer_, gas.buffer_sizes_.outputSizeInBytes, &gas.output_handle_, nullptr, 0);

  if (result != OPTIX_SUCCESS) {
    throw std::runtime_error("Failed to build acceleration structure");
  }

  return gas.output_handle_;
    }

void CoreCastWorkflow::setup_sbt(CoreCastOptixSBT& sbt) {
    sbt.raygen_host_record_ptr_ = std::make_unique<RaygenRecordType>();
    sbt.miss_host_record_ptr_ = std::make_unique<MissRecordType>();
    sbt.hitgroup_host_record_ptr_ = std::make_unique<HitgroupRecordType>();

    sbt.raygen_host_record_ptr_->data = raygen_data;
    sbt.miss_host_record_ptr_->data = miss_data;
    sbt.hitgroup_host_record_ptr_->data = hitgroup_data;

    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(raygen_program_name),
                                         sbt.raygen_host_record_ptr_.get()));
    OPTIX_CHECK(
        optixSbtRecordPackHeader(program_registry.get_program_group(miss_program_name), sbt.miss_host_record_ptr_.get()));
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(hitgroup_program_name),
                                         sbt.hitgroup_host_record_ptr_.get()));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.raygen_device_ptr_), sizeof(RaygenRecordType)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.miss_device_ptr_), sizeof(MissRecordType)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt.hitgroup_device_ptr_), sizeof(HitgroupRecordType)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.raygen_device_ptr_), sbt.raygen_host_record_ptr_.get(),
                          sizeof(RaygenRecordType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.miss_device_ptr_), sbt.miss_host_record_ptr_.get(),
                          sizeof(MissRecordType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt.hitgroup_device_ptr_), sbt. hitgroup_host_record_ptr_.get(),
                          sizeof(HitgroupRecordType), cudaMemcpyHostToDevice));

    sbt.sbt_.raygenRecord = raygen_device_ptr_;
    sbt.sbt_.missRecordBase = miss_device_ptr_;
    sbt.sbt_.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(MissRecordType));
    sbt.sbt_.missRecordCount = 1;
    sbt.sbt_.hitgroupRecordBase = hitgroup_device_ptr_;
    sbt.sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(HitgroupRecordType));
    sbt.sbt_.hitgroupRecordCount = 1;
}

void CoreCastWorkflow::setup_launch(CoreCastOptixLaunch& launch) {

    const void *launch_param_ptr = static_cast<const void *>(&launch.params_);
    const size_t launch_param_size = sizeof(ParamsType);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&launch.d_param_), launch_param_size));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void *>(launch.d_param_), launch_param_ptr, launch_param_size, cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<ParamsType, Params>) {
      OPTIX_CHECK(optixLaunch(pipeline, launch.stream_, launch.d_param_, launch_param_size, &sbt, params_.image_width,
                              params_.image_height,
                              /*depth=*/1));
    } else if constexpr (std::is_same_v<ParamsType, PointCloudLaunchParams>) {
      OPTIX_CHECK(optixLaunch(pipeline, launch.stream_, launch.d_param_, launch_param_size, &sbt, params_.image_width,
                              params_.image_height, /*depth=*/1));
    }

    CUDA_CHECK(cudaStreamSynchronize(launch.stream_));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(launch.d_param_)));


}

void CoreCastWorkflow::context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
  }

} // namespace corecast::optix