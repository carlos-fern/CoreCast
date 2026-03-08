#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include "corecast_optix/corecast_optix.hpp"

namespace corecast::processing {

template <typename PointCloudType>
class CoreCastDepthMap {


    public:
    CoreCastDepthMap(corecast_optix::CoreCastOptix& optix): optix_(optix) {}

    void setup_depth_map_processing(std::vector<PointCloudType>& point_cloud){


        point_cloud_buffer_ = std::make_unique<corecast_optix::CUDABuffer<PointCloudType, PointCloudType>>(point_cloud.data(), static_cast<int>(point_cloud.size()), true);

        set_point_cloud_build_input();
        set_point_cloud_accel_build_options();
        
        point_cloud_accel_ = std::make_unique<corecast_optix::CoreCastOptixAccel>(optix_->get_device_context(), point_cloud_accel_build_options_, {point_cloud_build_input_});
        point_cloud_accel_->build_acceleration_structure();

        set_pipeline_compile_options();
        set_module_compile_options();
        set_pipeline_link_options();
        set_builtin_is_options();
    }



    }


    void set_pipeline_compile_options(){

    depth_map_pipeline_compile_options_.usesMotionBlur = false;
    depth_map_pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    depth_map_pipeline_compile_options_.numPayloadValues = 1;
    depth_map_pipeline_compile_options_.numAttributeValues = 0;
    depth_map_pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    depth_map_pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
    depth_map_pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
    }

    void set_module_compile_options();

    void set_pipeline_link_options();

    void set_builtin_is_options(){
        depth_map_builtin_is_options_.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    }

    void set_raygen_program(){
        raygen_program_.name = "pointcloud_to_depth_map";
        raygen_program_.options = {};
        raygen_program_.desc = {};
        raygen_program_.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_program_.desc.raygen.entryFunctionName = "__raygen__pointcloud_to_depth_map";
    }

    void set_miss_program(){
        miss_program_.name = "pointcloud_miss";
        miss_program_.options = {};
        miss_program_.desc = {};
        miss_program_.desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_program_.desc.miss.entryFunctionName = "__miss__point_cloud";
    }

    void set_hitgroup_program(){
        hitgroup_program_.name = "pointcloud_hitgroup";
        hitgroup_program_.options = {};
        hitgroup_program_.desc = {};
        hitgroup_program_.desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_program_.desc.hitgroup.moduleIS = sphere_is_module;
        hitgroup_program_.desc.hitgroup.entryFunctionNameIS = nullptr;
        hitgroup_program_.desc.hitgroup.entryFunctionNameCH = "__closesthit__point_cloud";
    }

    void set_point_cloud_build_input(){
        point_cloud_build_input_.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
        point_cloud_build_input_.sphereArray.vertexBuffers = {point_cloud_buffer_->get_device_ptr()};
        point_cloud_build_input_.sphereArray.vertexStrideInBytes = sizeof(PointCloudType);
        point_cloud_build_input_.sphereArray.numVertices = static_cast<unsigned int>(point_cloud_buffer_->get_size());
        point_cloud_build_input_.sphereArray.radiusBuffers = {point_cloud_buffer_->get_device_ptr()};
        point_cloud_build_input_.sphereArray.radiusStrideInBytes = 0;
    }

    void set_point_cloud_accel_build_options(){
        point_cloud_accel_build_options_.buildFlags = OPTIX_BUILD_FLAG_NONE;
        point_cloud_accel_build_options_.operation = OPTIX_BUILD_OPERATION_BUILD;
    }
private:

// Main CoreCastOptix instance
std::shared_ptr<corecast_optix::CoreCastOptix> optix_;

// Optix options
OptixPipelineCompileOptions depth_map_pipeline_compile_options_;
OptixModuleCompileOptions depth_map_module_compile_options_;
OptixPipelineLinkOptions depth_map_pipeline_link_options_;
OptixBuiltinISOptions depth_map_builtin_is_options_;
OptixAccelBuildOptions point_cloud_accel_build_options_;

// Optix Programs
corecast_optix::CoreCastProgram raygen_program_;
corecast_optix::CoreCastProgram miss_program_;
corecast_optix::CoreCastProgram hitgroup_program_;

// Parameters
corecast_optix::PointCloudLaunchParams point_cloud_launch_params_;

//Point cloud items
std::unique_ptr<corecast_optix::CUDABuffer<PointCloudType, PointCloudType>> point_cloud_buffer_;
OptixBuildInput point_cloud_build_input_;

std::unique_ptr<corecast_optix::CoreCastOptixAccel> point_cloud_accel_;
OptixTraversableHandle point_cloud_accel_handle_;

}



}