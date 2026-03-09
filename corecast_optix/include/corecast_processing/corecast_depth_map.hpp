#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <optix.h>

#include "corecast_optix/corecast_optix.hpp"

namespace corecast::processing {

template <typename PointCloudType>
class CoreCastDepthMap {
public:
  explicit CoreCastDepthMap(corecast::optix::CoreCastOptix& optix);
  ~CoreCastDepthMap();

  void setup_depth_map_processing(std::vector<PointCloudType>& point_cloud,
                                  unsigned int image_width,
                                  unsigned int image_height,
                                  float point_radius = 0.01f);

  float* launch_depth_map();
  const corecast::optix::PointCloudLaunchParams& get_launch_params() const;
  void update_point_cloud(std::vector<PointCloudType>& point_cloud);

private:
  void set_pipeline_compile_options();
  void set_module_compile_options();
  void set_pipeline_link_options();
  void set_builtin_is_options();
  void set_programs();
  void set_pipeline_names();
  void set_launch_params(unsigned int image_width, unsigned int image_height);
  void set_point_cloud_build_input();
  void set_point_cloud_accel_build_options();
  void create_module_pipeline_and_sbt();

private:
  corecast::optix::CoreCastOptix& optix_;

  OptixPipelineCompileOptions depth_map_pipeline_compile_options_{};
  OptixModuleCompileOptions depth_map_module_compile_options_{};
  OptixPipelineLinkOptions depth_map_pipeline_link_options_{};
  OptixBuiltinISOptions depth_map_builtin_is_options_{};
  OptixAccelBuildOptions point_cloud_accel_build_options_{};
  OptixModule sphere_is_module_ = nullptr;

  corecast::optix::CoreCastProgram raygen_program_{};
  corecast::optix::CoreCastProgram miss_program_{};
  corecast::optix::CoreCastProgram hitgroup_program_{};
  std::vector<std::string> program_names_;

  std::string pipeline_name_;
  std::string module_name_;
  std::string sbt_name_;

  corecast::optix::PointCloudLaunchParams point_cloud_launch_params_{};
  
  //Host buffers
  std::vector<float> depth_host_buffer_;
  std::vector<float3> point_centers_host_;

  //Device buffers
  std::unique_ptr<corecast::optix::CUDABuffer<PointCloudType, PointCloudType>> point_cloud_buffer_;
  std::unique_ptr<corecast::optix::CUDABuffer<float3, float3>> point_centers_buffer_;
  std::unique_ptr<corecast::optix::CUDABuffer<float, float>> depth_buffer_;
  std::unique_ptr<corecast::optix::CoreCastOptixAccel> point_cloud_accel_;
  std::unique_ptr<corecast::optix::CUDABuffer<float, float>> point_radius_buffer_;

  OptixBuildInput point_cloud_build_input_{};
  CUdeviceptr sphere_vertex_buffers_[1]{};
  CUdeviceptr sphere_radius_buffers_[1]{};
  uint32_t sphere_input_flags_[1]{};

  corecast::optix::PointCloudRayGenData raygen_data_{};
  corecast::optix::PointCloudMissData miss_data_{};
  corecast::optix::PointCloudHitData hit_data_{};
};

extern template class CoreCastDepthMap<corecast::optix::PointXYZI>;

}  // namespace corecast::processing