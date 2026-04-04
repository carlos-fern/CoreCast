
struct CoreCastOptixContext {
    OptixDeviceContext context_ = nullptr;
    CUcontext cuCtx_ = 0;
    OptixDeviceContextOptions options_ = {};
}

struct CoreCastOptixModule {
    OptixModule module_;
    OptixPipelineCompileOptions pipeline_compile_options_;
    OptixModuleCompileOptions module_compile_options_;
    OptixBuiltinISOptions builtin_is_options_;
}

struct CoreCastOptixPipeline {
    OptixPipelineLinkOptions link_options_;
    std::vector<std::string> program_names_;
    OptixPipeline pipeline_;
}

struct CoreCastOptixGAS {
    OptixDeviceContext optixContext_;
    CUstream stream_;
    OptixAccelBuildOptions build_options_;
    std::vector<OptixBuildInput> build_inputs_;
    OptixAccelBufferSizes buffer_sizes_ = {};
    CUdeviceptr output_buffer_ = 0;
    CUdeviceptr temp_buffer_ = 0;
    OptixTraversableHandle output_handle_ = 0;
}

struct CoreCastSBT {
    OptixShaderBindingTable sbt_;

    CUdeviceptr raygen_device_ptr_ = 0;
    CUdeviceptr miss_device_ptr_ = 0;
    CUdeviceptr hitgroup_device_ptr_ = 0;
  
    std::unique_ptr<RaygenRecordType> raygen_host_record_ptr_;
    std::unique_ptr<MissRecordType> miss_host_record_ptr_;
    std::unique_ptr<HitgroupRecordType> hitgroup_host_record_ptr_;
}