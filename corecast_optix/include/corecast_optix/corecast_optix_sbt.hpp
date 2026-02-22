#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <optix.h>
#include <string>

#include <sutil/Exception.h>

#include "corecast_optix/corecast_optix_program_registry.hpp"

namespace corecast_optix
{

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <typename RecordType, typename DataType>
class CoreCastOptixSBT
{

    public:
    CoreCastOptixSBT(std::string program_name, CoreCastOptixProgramRegistry& program_registry, DataType data){

        host_record_ptr_ = std::make_unique<RecordType>();
        host_record_size_ = sizeof(RecordType);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), host_record_size_));

        host_record_ptr_->data = data;
        OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(program_name), host_record_ptr_.get()));
        CUDA_CHECK(cudaMemcpy(device_ptr_, host_record_ptr_.get(), host_record_size_, cudaMemcpyHostToDevice));
    }


    private:

    OptixShaderBindingTable sbt_;
    
    // Device pointer to the SBT record
    CUdeviceptr device_ptr_;
    size_t device_size_{};
 
    // Host pointer to the SBT record
    std::unique_ptr<RecordType> host_record_ptr_;
    size_t host_record_size_{};


};

}  // namespace corecast_optix
