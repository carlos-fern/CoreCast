#include "corecast_processing/corecast_depth_map.hpp"

#include <optix_host.h>
#include <sutil/Exception.h>

namespace corecast::processing {

template <typename PointCloudType>
CoreCastDepthMap<PointCloudType>::CoreCastDepthMap(corecast_optix::CoreCastOptix& optix) : optix_(optix) {
  static_assert(std::is_same_v<PointCloudType, corecast_optix::PointXYZI>,
                "CoreCastDepthMap currently expects PointCloudType == corecast_optix::PointXYZI");
}

template <typename PointCloudType>
CoreCastDepthMap<PointCloudType>::~CoreCastDepthMap() {
  if (sphere_is_module_ != nullptr) {
    OPTIX_CHECK_NOTHROW(optixModuleDestroy(sphere_is_module_));
  }
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::setup_depth_map_processing(std::vector<PointCloudType>& point_cloud,
                                                                  unsigned int image_width,
                                                                  unsigned int image_height,
                                                                  float point_radius) {
  if (point_cloud.empty()) {
    throw std::runtime_error("Point cloud cannot be empty");
  }

  point_cloud_buffer_ = std::make_unique<corecast_optix::CUDABuffer<PointCloudType, PointCloudType>>(
      point_cloud.data(), static_cast<int>(point_cloud.size()), true);

  point_centers_host_.clear();
  point_centers_host_.reserve(point_cloud.size());
  for (const auto& p : point_cloud) {
    point_centers_host_.push_back(make_float3(p.x, p.y, p.z));
  }

  point_centers_buffer_ = std::make_unique<corecast_optix::CUDABuffer<float3, float3>>(
      point_centers_host_.data(), static_cast<int>(point_centers_host_.size()), true);

  point_radius_buffer_ = std::make_unique<corecast_optix::CUDABuffer<float, float>>(
      &point_radius, 1, true);

  set_point_cloud_build_input();
  set_point_cloud_accel_build_options();

  point_cloud_accel_ = std::make_unique<corecast_optix::CoreCastOptixAccel>(
      optix_.get_device_context(), point_cloud_accel_build_options_,
      std::vector<OptixBuildInput>{point_cloud_build_input_}, /*stream=*/nullptr);
  point_cloud_accel_->build_acceleration_structure();

  set_pipeline_compile_options();
  set_module_compile_options();
  set_pipeline_link_options();
  set_builtin_is_options();
  set_programs();
  set_pipeline_names();
  set_launch_params(image_width, image_height);

  create_module_pipeline_and_sbt();
}

template <typename PointCloudType>
float* CoreCastDepthMap<PointCloudType>::launch_depth_map() {
  optix_.launch_pipeline(pipeline_name_, point_cloud_launch_params_, sbt_name_);
  return optix_.get_result<float, float>(pipeline_name_, *depth_buffer_);
}

template <typename PointCloudType>
const corecast_optix::PointCloudLaunchParams& CoreCastDepthMap<PointCloudType>::get_launch_params() const {
  return point_cloud_launch_params_;
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_pipeline_compile_options() {
  depth_map_pipeline_compile_options_ = {};
  depth_map_pipeline_compile_options_.usesMotionBlur = false;
  depth_map_pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  depth_map_pipeline_compile_options_.numPayloadValues = 1;
  depth_map_pipeline_compile_options_.numAttributeValues = 0;
  depth_map_pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  depth_map_pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
  depth_map_pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_module_compile_options() {
  depth_map_module_compile_options_ = {};
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_pipeline_link_options() {
  depth_map_pipeline_link_options_ = {};
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_builtin_is_options() {
  depth_map_builtin_is_options_ = {};
  depth_map_builtin_is_options_.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_programs() {
  raygen_program_ = {};
  raygen_program_.name = "pointcloud_to_depth_map";
  raygen_program_.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_program_.desc.raygen.entryFunctionName = "__raygen__pointcloud_to_depth_map";

  miss_program_ = {};
  miss_program_.name = "pointcloud_miss";
  miss_program_.desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_program_.desc.miss.entryFunctionName = "__miss__point_cloud";

  hitgroup_program_ = {};
  hitgroup_program_.name = "pointcloud_hitgroup";
  hitgroup_program_.desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_program_.desc.hitgroup.entryFunctionNameIS = nullptr;
  hitgroup_program_.desc.hitgroup.entryFunctionNameCH = "__closesthit__point_cloud";
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_pipeline_names() {
  pipeline_name_ = "pointcloud_to_depth_map_pipeline";
  module_name_ = raygen_program_.name;
  sbt_name_ = "pointcloud_to_depth_map_sbt";
  program_names_ = {raygen_program_.name, miss_program_.name, hitgroup_program_.name};
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_launch_params(unsigned int image_width, unsigned int image_height) {
  point_cloud_launch_params_ = {};
  point_cloud_launch_params_.image_width = image_width;
  point_cloud_launch_params_.image_height = image_height;
  point_cloud_launch_params_.sensor_origin = make_float3(0.0f, 0.0f, 0.0f);
  point_cloud_launch_params_.sensor_x_axis = make_float3(1.0f, 0.0f, 0.0f);
  point_cloud_launch_params_.sensor_y_axis = make_float3(0.0f, 1.0f, 0.0f);
  point_cloud_launch_params_.sensor_z_axis = make_float3(0.0f, 0.0f, 1.0f);
  point_cloud_launch_params_.t_min = 0.001f;
  point_cloud_launch_params_.t_max = 1000.0f;
  point_cloud_launch_params_.handle = point_cloud_accel_->get_output_handle();

  depth_host_buffer_.assign(static_cast<size_t>(image_width) * image_height, 0.0f);
  depth_buffer_ = std::make_unique<corecast_optix::CUDABuffer<float, float>>(
      depth_host_buffer_.data(), static_cast<int>(depth_host_buffer_.size()), false);
  point_cloud_launch_params_.depth_buffer = depth_buffer_->get_device_ptr();
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_point_cloud_build_input() {
  sphere_vertex_buffers_[0] = point_centers_buffer_->device_ptr_;
  sphere_radius_buffers_[0] = point_radius_buffer_->device_ptr_;
  sphere_input_flags_[0] = OPTIX_GEOMETRY_FLAG_NONE;

  point_cloud_build_input_ = {};
  point_cloud_build_input_.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
  point_cloud_build_input_.sphereArray.vertexBuffers = sphere_vertex_buffers_;
  point_cloud_build_input_.sphereArray.vertexStrideInBytes = sizeof(float3);
  point_cloud_build_input_.sphereArray.numVertices = static_cast<unsigned int>(point_centers_host_.size());
  point_cloud_build_input_.sphereArray.radiusBuffers = sphere_radius_buffers_;
  point_cloud_build_input_.sphereArray.radiusStrideInBytes = 0;
  point_cloud_build_input_.sphereArray.singleRadius = 1;
  point_cloud_build_input_.sphereArray.flags = sphere_input_flags_;
  point_cloud_build_input_.sphereArray.numSbtRecords = 1;
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::set_point_cloud_accel_build_options() {
  point_cloud_accel_build_options_ = {};
  point_cloud_accel_build_options_.buildFlags = OPTIX_BUILD_FLAG_NONE;
  point_cloud_accel_build_options_.operation = OPTIX_BUILD_OPERATION_BUILD;
}

template <typename PointCloudType>
void CoreCastDepthMap<PointCloudType>::create_module_pipeline_and_sbt() {
  std::string ptx_path = CORECAST_POINTCLOUD_PTX_PATH;
  optix_.create_module(module_name_, depth_map_pipeline_compile_options_, depth_map_module_compile_options_,
                       ptx_path);

  OPTIX_CHECK(optixBuiltinISModuleGet(optix_.get_device_context(), &depth_map_module_compile_options_,
                                      &depth_map_pipeline_compile_options_, &depth_map_builtin_is_options_,
                                      &sphere_is_module_));
  hitgroup_program_.desc.hitgroup.moduleIS = sphere_is_module_;

  optix_.add_program_to_module(module_name_, raygen_program_);
  optix_.add_program_to_module(module_name_, miss_program_);
  optix_.add_program_to_module(module_name_, hitgroup_program_);
  optix_.build_pipeline(pipeline_name_, program_names_, depth_map_pipeline_link_options_);

  raygen_data_ = {point_cloud_buffer_->get_device_ptr(),
                  static_cast<unsigned int>(point_cloud_buffer_->get_num_elements())};
  miss_data_ = {0.0f};
  hit_data_ = {nullptr, nullptr, 0.0f};

  optix_.create_trace_sbt<
      corecast_optix::SbtRecord<corecast_optix::PointCloudRayGenData>, corecast_optix::PointCloudRayGenData,
      corecast_optix::SbtRecord<corecast_optix::PointCloudMissData>, corecast_optix::PointCloudMissData,
      corecast_optix::SbtRecord<corecast_optix::PointCloudHitData>, corecast_optix::PointCloudHitData>(
      sbt_name_, raygen_program_.name, miss_program_.name, hitgroup_program_.name, raygen_data_, miss_data_,
      hit_data_);
}

template class CoreCastDepthMap<corecast_optix::PointXYZI>;

}  // namespace corecast::processing
