#pragma once

#include <optix.h>
#include <optix_types.h>
#include <sutil/Exception.h>

#include <any>
#include <corecast_optix/corecast_optix_cuda_buffer.hpp>
#include <corecast_optix/corecast_optix_program_registry.hpp>
#include <memory>
#include <optional>
#include <vector>

namespace corecast::optix {

class CoreCastOptix;

template <typename T>
concept SbtRecordType =
    // 1. Check that the members actually exist
    requires(T r) {
      r.header;
      r.data;
    } &&
    // 2. Check the alignment of the struct matches the OptiX requirement
    alignof(T) == OPTIX_SBT_RECORD_ALIGNMENT &&
    // 3. Check the header is exactly the right size
    sizeof(T::header) == OPTIX_SBT_RECORD_HEADER_SIZE &&
    // 4. Check your original constraints on the 'data' member
    std::is_trivially_copyable_v<decltype(T::data)> && !std::is_pointer_v<decltype(T::data)>;

class CoreCastOptixSBT {
 public:
  /**
   * @brief Constructor for the CoreCastOptixSBT class.
   * @param program_name The name of the program to create the SBT for.
   * @param program_registry The program registry to use to get the program group.
   * @param data The data to store in the SBT.
   */
  CoreCastOptixSBT(std::string program_name, CoreCastOptixProgramRegistry& program_registry, SbtRecordType auto& record)
      : sbt_({}) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), sizeof(record)));

    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(program_name), &record));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(device_ptr_), &record, sizeof(record), cudaMemcpyHostToDevice));

    sbt_.raygenRecord = device_ptr_;
    sbt_.missRecordBase = device_ptr_;
    sbt_.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(record));
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = device_ptr_;
    sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(record));
    sbt_.hitgroupRecordCount = 1;
  }

  /**
   * @brief Destructor for the CoreCastOptixSBT class.
   */
  ~CoreCastOptixSBT() {
    if (device_ptr_ != 0) {
      CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(device_ptr_)));
    }
  }

  /**
   * @brief Get the SBT.
   * @return The SBT.
   */
  const OptixShaderBindingTable& get_sbt() const { return sbt_; }

 private:
  OptixShaderBindingTable sbt_;

  // Device pointer to the SBT record
  CUdeviceptr device_ptr_;
  size_t device_size_{};
};

class CoreCastOptixTraceSBT {
 public:
  CoreCastOptixTraceSBT(const std::string& raygen_program_name, const std::string& miss_program_name,
                        const std::string& hitgroup_program_name, CoreCastOptixProgramRegistry& program_registry,
                        SbtRecordType auto& raygen_record, SbtRecordType auto& miss_record,
                        SbtRecordType auto& hitgroup_record)
      : sbt_({}) {
    // 3. Pack headers
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(raygen_program_name), &raygen_record));
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(miss_program_name), &miss_record));
    OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(hitgroup_program_name), &hitgroup_record));

    // 4. Allocate Device Memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_device_ptr_), sizeof(decltype(raygen_record))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_device_ptr_), sizeof(decltype(miss_record))));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_device_ptr_), sizeof(decltype(hitgroup_record))));

    // 5. Copy to Device
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_device_ptr_), &raygen_record, sizeof(decltype(raygen_record)),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_device_ptr_), &miss_record, sizeof(decltype(miss_record)),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_device_ptr_), &hitgroup_record,
                          sizeof(decltype(hitgroup_record)), cudaMemcpyHostToDevice));

    // 6. Setup SBT
    sbt_.raygenRecord = raygen_device_ptr_;
    sbt_.missRecordBase = miss_device_ptr_;
    sbt_.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(decltype(miss_record)));
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = hitgroup_device_ptr_;
    sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(decltype(hitgroup_record)));
    sbt_.hitgroupRecordCount = 1;
  }

  ~CoreCastOptixTraceSBT() {
    if (raygen_device_ptr_ != 0) CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(raygen_device_ptr_)));
    if (miss_device_ptr_ != 0) CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(miss_device_ptr_)));
    if (hitgroup_device_ptr_ != 0) CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(hitgroup_device_ptr_)));
  }

  const OptixShaderBindingTable& get_sbt() const { return sbt_; }

 private:
  OptixShaderBindingTable sbt_;

  CUdeviceptr raygen_device_ptr_ = 0;
  CUdeviceptr miss_device_ptr_ = 0;
  CUdeviceptr hitgroup_device_ptr_ = 0;
};

struct CoreCastOptixContext {
  OptixDeviceContext device_context_ = nullptr;
  CUcontext cuCtx_ = 0;
  OptixDeviceContextOptions options_ = {};
};

struct CoreCastOptixModule {
  OptixModule module_;
  OptixPipelineCompileOptions pipeline_compile_options_;
  OptixModuleCompileOptions module_compile_options_;
  OptixBuiltinISOptions builtin_is_options_;
  std::optional<std::string> ptx_path_;
};

struct CoreCastOptixPipeline {
  OptixPipelineLinkOptions link_options_;
  OptixPipelineCompileOptions compile_options_;
  std::vector<std::string> program_names_;
  OptixPipeline pipeline_;
};

struct CoreCastOptixGAS {
  OptixDeviceContext optixContext_;
  CUstream stream_;
  OptixAccelBuildOptions build_options_;
  std::vector<OptixBuildInput> build_inputs_;
  OptixAccelBufferSizes buffer_sizes_ = {};
  CUdeviceptr output_buffer_ = 0;
  CUdeviceptr temp_buffer_ = 0;
  OptixTraversableHandle output_handle_ = 0;
};

struct CoreCastSBT {
  OptixShaderBindingTable sbt_;

  CUdeviceptr raygen_device_ptr_ = 0;
  CUdeviceptr miss_device_ptr_ = 0;
  CUdeviceptr hitgroup_device_ptr_ = 0;

  //    std::unique_ptr<auto> raygen_host_record_ptr_;
  //   std::unique_ptr<auto> miss_host_record_ptr_;
  //    std::unique_ptr<auto> hitgroup_host_record_ptr_;
};

struct CoreCastTraceSBT {
  OptixShaderBindingTable sbt_;

  CUdeviceptr raygen_device_ptr_ = 0;
  CUdeviceptr miss_device_ptr_ = 0;
  CUdeviceptr hitgroup_device_ptr_ = 0;

  //    std::unique_ptr<auto> raygen_host_record_ptr_;
  //   std::unique_ptr<auto> miss_host_record_ptr_;
  //    std::unique_ptr<auto> hitgroup_host_record_ptr_;
};

struct WorkflowOptions {
  OptixDeviceContextOptions context_options_;
  OptixPipelineCompileOptions pipeline_compile_options_;
  OptixModuleCompileOptions module_compile_options_;
  OptixPipelineLinkOptions pipeline_link_options_;
  OptixBuiltinISOptions builtin_is_options_;
  OptixAccelBuildOptions accel_build_options_;
};

struct CoreCastOptixLaunch {
  OptixPipeline pipeline_;
};

}  // namespace corecast::optix