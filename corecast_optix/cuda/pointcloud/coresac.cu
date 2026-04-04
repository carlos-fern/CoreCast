#include <optix_device.h>

#include "corecast_optix/corecast_optix_cuda_types.hpp"
#include "pointcloud_ray_utils.cuh"

extern "C" {
__constant__ corecast::optix::CoreSACParams params;
}

extern "C" __global__ void __raygen__coresac() {
  const uint3 pixel_coordinate = optixGetLaunchIndex();
  const uint3 resolution = optixGetLaunchDimensions();
  uint32_t payload = 0;

  const float2 ray_aim_point = corecast::optix::pixel_to_ray_aim_point(pixel_coordinate, resolution);
  const float3 ray_direction = corecast::optix::ray_aim_to_direction(
      ray_aim_point, params.sensor_x_axis, params.sensor_y_axis, params.sensor_z_axis);


  optixTrace(params.handle, params.sensor_origin, ray_direction, params.t_min, params.t_max, 0.0f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, payload);

  const float true_distance = __uint_as_float(payload);
  if (true_distance <= 0.0f) { // No hit
    return;
  }

  const float3 hit_point = make_float3(params.sensor_origin.x + true_distance * ray_direction.x,
                                       params.sensor_origin.y + true_distance * ray_direction.y,
                                       params.sensor_origin.z + true_distance * ray_direction.z);

  const uint32_t hit_index = atomicAdd(params.hit_count, 1U);
  if (hit_index < params.max_hit_points) {
    params.hit_points[hit_index] = hit_point;
  }
}
