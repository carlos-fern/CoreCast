#include <corecast_optix_v2/corecast_workflow.hpp>


namespace corecast::optix {

CoreCastWorkflow::CoreCastWorkflow(std::shared_ptr<CoreCastOptix> optix) : optix_(optix) {

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

void CoreCastWorkflow::init_context(CUcontext context_id,OptixDeviceContextOptions& options) {
    check_cuda(cudaFree(0), "cudaFree(0)");
    context.cuCtx_ = context_id;
  
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

void CoreCastWorkflow::setup_module( OptixPipelineCompileOptions& pipeline_compile_options,
        OptixModuleCompileOptions& module_compile_options, std::string& ptx_path) {
    std::vector<char> ptx = read_file_bytes(ptx_path);
    OPTIX_CHECK_LOG(optixModuleCreate(context_->get_context(), &module_compile_options_, &pipeline_compile_options_,
    ptx.data(), ptx.size(), LOG, &LOG_SIZE, &module_));
    }

void CoreCastWorkflow::setup_builtin_is_module(OptixPipelineCompileOptions& pipeline_compile_options,
        OptixModuleCompileOptions& module_compile_options, OptixBuiltinISOptions& builtin_is_options) {
    OPTIX_CHECK_LOG(optixBuiltinISModuleGet(context_->get_context(), &module_compile_options_, &pipeline_compile_options_,
    &builtin_is_options_, &module_));
}

void CoreCastWorkflow::setup_programs_and_program_groups() {

}

void CoreCastWorkflow::setup_pipelines() {
    std::vector<OptixProgramGroup> groups = program_registry_->get_program_groups(program_names);

    OPTIX_CHECK_LOG(optixPipelineCreate(context_->get_context(), &module_->get_pipeline_compile_options(), &link_options_,
                                        groups.data(), static_cast<uint32_t>(groups.size()), LOG, &LOG_SIZE, &pipeline_));
}

void CoreCastWorkflow::setup_gas() {

    if (build_inputs_.empty()) {
        throw std::runtime_error("CoreCastOptixAccel requires at least one OptixBuildInput");
      }
  
      OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext_, &build_options_, build_inputs_.data(),
                                               static_cast<unsigned int>(build_inputs_.size()), &buffer_sizes_));
  
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output_buffer_), buffer_sizes_.outputSizeInBytes));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp_buffer_), buffer_sizes_.tempSizeInBytes));
      OptixResult result =
      optixAccelBuild(optixContext_, stream_, &build_options_, build_inputs_.data(),
                      static_cast<unsigned int>(build_inputs_.size()), temp_buffer_, buffer_sizes_.tempSizeInBytes,
                      output_buffer_, buffer_sizes_.outputSizeInBytes, &output_handle_, nullptr, 0);

  if (result != OPTIX_SUCCESS) {
    throw std::runtime_error("Failed to build acceleration structure");
  }

  return output_handle_;


    }

void CoreCastWorkflow::setup_sbt() {
    raygen_host_record_ptr_ = std::make_unique<RaygenRecordType>();
    miss_host_record_ptr_ = std::make_unique<MissRecordType>();
    hitgroup_host_record_ptr_ = std::make_unique<HitgroupRecordType>();

    raygen_host_record_ptr_->data = raygen_data;
    miss_host_record_ptr_->data = miss_data;
    hitgroup_host_record_ptr_->data = hitgroup_data;

    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(raygen_program_name),
                                         raygen_host_record_ptr_.get()));
    OPTIX_CHECK(
        optixSbtRecordPackHeader(program_registry.get_program_group(miss_program_name), miss_host_record_ptr_.get()));
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(hitgroup_program_name),
                                         hitgroup_host_record_ptr_.get()));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_device_ptr_), sizeof(RaygenRecordType)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_device_ptr_), sizeof(MissRecordType)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_device_ptr_), sizeof(HitgroupRecordType)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_device_ptr_), raygen_host_record_ptr_.get(),
                          sizeof(RaygenRecordType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_device_ptr_), miss_host_record_ptr_.get(),
                          sizeof(MissRecordType), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_device_ptr_), hitgroup_host_record_ptr_.get(),
                          sizeof(HitgroupRecordType), cudaMemcpyHostToDevice));

    sbt_.raygenRecord = raygen_device_ptr_;
    sbt_.missRecordBase = miss_device_ptr_;
    sbt_.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(MissRecordType));
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = hitgroup_device_ptr_;
    sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(HitgroupRecordType));
    sbt_.hitgroupRecordCount = 1;
}

void CoreCastWorkflow::setup_launch() {

    const void *launch_param_ptr = static_cast<const void *>(&params_);
    const size_t launch_param_size = sizeof(ParamsType);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param_), launch_param_size));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void *>(d_param_), launch_param_ptr, launch_param_size, cudaMemcpyHostToDevice));

    if constexpr (std::is_same_v<ParamsType, Params>) {
      OPTIX_CHECK(optixLaunch(pipeline, stream_, d_param_, launch_param_size, &sbt, params_.image_width,
                              params_.image_height,
                              /*depth=*/1));
    } else if constexpr (std::is_same_v<ParamsType, PointCloudLaunchParams>) {
      OPTIX_CHECK(optixLaunch(pipeline, stream_, d_param_, launch_param_size, &sbt, params_.image_width,
                              params_.image_height, /*depth=*/1));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param_)));


}

void CoreCastWorkflow::context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
  }