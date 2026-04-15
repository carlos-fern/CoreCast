module;

#include <corecast_optix/corecast_optix_types.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "optix_types.h"

export module coresac_workflow;

import workflow;

export namespace corecast::optix {

struct RayGenRecord {
  int data;
};

struct HitGroupRecord {
  int data;
};

struct MissRecord {
  int data;
};

struct CoreSACParams {
  unsigned int image_width;
  unsigned int image_height;

  // Sensor position and orientation basis vectors
  float3 sensor_origin;
  float3 sensor_x_axis;
  float3 sensor_y_axis;
  float3 sensor_z_axis;

  // Tracing limits
  float t_min;
  float t_max;

  // The 3D scene handle
  OptixTraversableHandle handle;

  OptixAabb* bounding_box;
  float3* hit_points;
  uint32_t* hit_count;
  uint32_t max_hit_points;
};

struct CoreSACGroupParams {
  float3* hit_points;
  uint32_t hit_point_count;

  float voxel_size;
  uint32_t max_num_total_voxels;
  uint32_t max_points_per_voxel;

  uint32_t* voxel_point_count;
  uint32_t* active_voxel_ids;
  float3* voxel_points;
};

struct CoreSACAABBParams {
  uint32_t* voxel_point_count;
  uint32_t* active_voxel_ids;
  float3* voxel_points;
  uint32_t max_num_total_voxels;
  uint32_t max_points_per_voxel;
  uint32_t min_points_per_voxel;

  OptixAabb* bounding_boxes;
};

struct CoreSACScoringParams {
  uint32_t max_num_total_voxels;
  CoreSACAABBParams aabb_params;
};

class CoreSACWorkflow : private BaseWorkflow<CoreSACWorkflow> {
 public:
  CoreSACWorkflow(std::shared_ptr<CoreCastOptix> optix, std::optional<WorkflowOptions> workflow_options)
      : BaseWorkflow(optix, workflow_options) {
    setup_module(module_);
  };

 private:
  CoreCastOptixModule module_;

  void configure_module() {
    // Placehodler values
    // module_.pipeline_compile_options_.usesMotionBlur = 0;
    // module_.pipeline_compile_options_.traversableGraphFlags = 0;
    // module_.pipeline_compile_options_.numPayValues = 1;
    // module_.pipeline_compile_options_.exceptionFlags = 0;
    // module_.pipeline_compile_options_.pipelineLaunchParamsVariableName = "";
    // module_.pipeline_compile_options_.usesPrimitiveTypeFlags = 0;
    // module_.pipeline_compile_options_.allowOpacityMicromaps = 0;
    // module_.pipeline_compile_options_.allowClusteredGeometry = 0;

    // module_.module_compile_options_.maxRegisterCount = 0;
    // module_.module_compile_options_.optLevel = 0;
    // module_.module_compile_options_.debugLevel = 0;
    // module_.module_compile_options_.OptixModuleCompileBoundValueEntry = 0;
    // module_.module_compile_options_.numBoundvalues = 0;
    // module_.module_compile_options_.numPayloadTypes = 0;
    // module_.module_compile_options_.payloadtypes = 0;

    // module_.builtin_is_options_.builtinISModuleType = 0;
    // module_.builtin_is_options_.usesMotionBlur = 0;
    // module_.builtin_is_options_.buildFlags = 0;
    // module_.builtin_is_options_.curveEndcapFlags = 0;

    // module_.ptx_path = ""
  }
};

}  // namespace corecast::optix