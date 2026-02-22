#pragma once

#include <optix.h>

#include <cuda_runtime.h>

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
    CoreCastOptixSBT(std::string program_name, CoreCastOptixSBTType type, CoreCastOptixProgramRegistry& program_registry, DataType data){

        host_record_ptr_ = std::make_unique<RecordType>();
        host_record_size_ = sizeof(SbtRecord<RecordType>);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), host_record_size_));

        host_record_ptr_->data = data;
        OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(program_name), host_record_ptr_.get()));
        CUDA_CHECK(cudaMemcpy(device_ptr_, host_record_ptr_.get(), host_record_size_, cudaMemcpyHostToDevice));
    }


    private:

    OptixShaderBindingTable sbt_;
    
    // Device pointer to the SBT record
    CUdeviceptr device_ptr_;
    const size_t device_size_;
 
    // Host pointer to the SBT record
    std::unique_ptr<SbtRecord<RecordType>> host_record_ptr_;
    const size_t host_record_size_;


};

}  // namespace corecast_optix
