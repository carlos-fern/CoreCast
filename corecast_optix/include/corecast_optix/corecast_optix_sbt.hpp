#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <optix.h>
#include <optix_stubs.h>
#include <string>

#include "corecast_optix/corecast_optix_cuda_types.hpp"
#include "corecast_optix/corecast_optix_program_registry.hpp"

namespace corecast::optix
{

template <ValidCUDAType T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <ValidCUDAType RecordType, ValidCUDAType DataType>
class CoreCastOptixSBT
{

    public:
    /**
    * @brief Constructor for the CoreCastOptixSBT class.
    * @param program_name The name of the program to create the SBT for.
    * @param program_registry The program registry to use to get the program group.
    * @param data The data to store in the SBT.
    */
    CoreCastOptixSBT(std::string program_name, CoreCastOptixProgramRegistry& program_registry, DataType data): sbt_({}){

        host_record_ptr_ = std::make_unique<RecordType>();
        host_record_size_ = sizeof(RecordType);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), host_record_size_));

        host_record_ptr_->data = data;
        OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(program_name), host_record_ptr_.get()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(device_ptr_), host_record_ptr_.get(), host_record_size_, cudaMemcpyHostToDevice));
        
        sbt_.raygenRecord = device_ptr_;
        sbt_.missRecordBase = device_ptr_;
        sbt_.missRecordStrideInBytes = static_cast<unsigned int>(host_record_size_);
        sbt_.missRecordCount = 1;
        sbt_.hitgroupRecordBase = device_ptr_;
        sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(host_record_size_);
        sbt_.hitgroupRecordCount = 1;
    }

    /**
    * @brief Destructor for the CoreCastOptixSBT class.
    */
    ~CoreCastOptixSBT()
    {
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
 
    // Host pointer to the SBT record
    std::unique_ptr<RecordType> host_record_ptr_;
    size_t host_record_size_{};


};

template <ValidCUDAType RaygenRecordType, ValidCUDAType RaygenDataType,
          ValidCUDAType MissRecordType, ValidCUDAType MissDataType,
          ValidCUDAType HitgroupRecordType, ValidCUDAType HitgroupDataType>
class CoreCastOptixTraceSBT
{
public:
    CoreCastOptixTraceSBT(
        const std::string& raygen_program_name,
        const std::string& miss_program_name,
        const std::string& hitgroup_program_name,
        CoreCastOptixProgramRegistry& program_registry,
        const RaygenDataType& raygen_data,
        const MissDataType& miss_data,
        const HitgroupDataType& hitgroup_data) : sbt_({}) {

        raygen_host_record_ptr_ = std::make_unique<RaygenRecordType>();
        miss_host_record_ptr_ = std::make_unique<MissRecordType>();
        hitgroup_host_record_ptr_ = std::make_unique<HitgroupRecordType>();

        raygen_host_record_ptr_->data = raygen_data;
        miss_host_record_ptr_->data = miss_data;
        hitgroup_host_record_ptr_->data = hitgroup_data;

        OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(raygen_program_name), raygen_host_record_ptr_.get()));
        OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(miss_program_name), miss_host_record_ptr_.get()));
        OPTIX_CHECK(optixSbtRecordPackHeader(program_registry.get_program_group(hitgroup_program_name), hitgroup_host_record_ptr_.get()));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_device_ptr_), sizeof(RaygenRecordType)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_device_ptr_), sizeof(MissRecordType)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_device_ptr_), sizeof(HitgroupRecordType)));

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_device_ptr_), raygen_host_record_ptr_.get(), sizeof(RaygenRecordType), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_device_ptr_), miss_host_record_ptr_.get(), sizeof(MissRecordType), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_device_ptr_), hitgroup_host_record_ptr_.get(), sizeof(HitgroupRecordType), cudaMemcpyHostToDevice));

        sbt_.raygenRecord = raygen_device_ptr_;
        sbt_.missRecordBase = miss_device_ptr_;
        sbt_.missRecordStrideInBytes = static_cast<unsigned int>(sizeof(MissRecordType));
        sbt_.missRecordCount = 1;
        sbt_.hitgroupRecordBase = hitgroup_device_ptr_;
        sbt_.hitgroupRecordStrideInBytes = static_cast<unsigned int>(sizeof(HitgroupRecordType));
        sbt_.hitgroupRecordCount = 1;
    }

    ~CoreCastOptixTraceSBT() {
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

    const OptixShaderBindingTable& get_sbt() const { return sbt_; }

private:
    OptixShaderBindingTable sbt_;

    CUdeviceptr raygen_device_ptr_ = 0;
    CUdeviceptr miss_device_ptr_ = 0;
    CUdeviceptr hitgroup_device_ptr_ = 0;

    std::unique_ptr<RaygenRecordType> raygen_host_record_ptr_;
    std::unique_ptr<MissRecordType> miss_host_record_ptr_;
    std::unique_ptr<HitgroupRecordType> hitgroup_host_record_ptr_;
};

}  // namespace corecast::optix
