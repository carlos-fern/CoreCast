#include <optix_device.h>

#include "corecast_optix/corecast_optix_cuda_types.hpp"

extern "C" {
__constant__ corecast::optix::PointCloudLaunchParams params;
}

extern "C" __global__ void __raygen__pointcloud_to_depth_map() {
  const uint3 pixel_coordinate = optixGetLaunchIndex();
  const uint3 resolution = optixGetLaunchDimensions();
  uint32_t payload = 0;

  // Map pixel coordinates to normalized image-plane coordinates in [-1, 1].
  const float2 ray_aim_point =
      make_float2((2.0f * static_cast<float>(pixel_coordinate.x) + 1.0f) / static_cast<float>(resolution.x) - 1.0f,
                  (2.0f * static_cast<float>(pixel_coordinate.y) + 1.0f) / static_cast<float>(resolution.y) - 1.0f);

  const float3 ray_direction = make_float3(
      ray_aim_point.x * params.sensor_x_axis.x + ray_aim_point.y * params.sensor_y_axis.x + params.sensor_z_axis.x,
      ray_aim_point.x * params.sensor_x_axis.y + ray_aim_point.y * params.sensor_y_axis.y + params.sensor_z_axis.y,
      ray_aim_point.x * params.sensor_x_axis.z + ray_aim_point.y * params.sensor_y_axis.z + params.sensor_z_axis.z);

  optixTrace(params.handle, params.sensor_origin, ray_direction, params.t_min, params.t_max, 0.0f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, payload);

  const float true_distance = __uint_as_float(payload);
  const uint32_t linear_index = pixel_coordinate.y * resolution.x + pixel_coordinate.x;
  params.depth_buffer[linear_index] = true_distance;
}

extern "C" __global__ void __closesthit__point_cloud() {
  const float distance = optixGetRayTmax();
  optixSetPayload_0(__float_as_uint(distance));
}

extern "C" __global__ void __miss__point_cloud() {
  // Return 0.0 when no geometry is hit.
  optixSetPayload_0(__float_as_uint(0.0f));
}