#pragma once

#include <optix_device.h>

namespace corecast::optix {

inline __device__ float2 pixel_to_ray_aim_point(const uint3 pixel_coordinate, const uint3 resolution) {
  return make_float2((2.0f * static_cast<float>(pixel_coordinate.x) + 1.0f) / static_cast<float>(resolution.x) - 1.0f,
                     (2.0f * static_cast<float>(pixel_coordinate.y) + 1.0f) / static_cast<float>(resolution.y) - 1.0f);
}

inline __device__ float3 ray_aim_to_direction(const float2 ray_aim_point, const float3 sensor_x_axis,
                                              const float3 sensor_y_axis, const float3 sensor_z_axis) {
  return make_float3(ray_aim_point.x * sensor_x_axis.x + ray_aim_point.y * sensor_y_axis.x + sensor_z_axis.x,
                     ray_aim_point.x * sensor_x_axis.y + ray_aim_point.y * sensor_y_axis.y + sensor_z_axis.y,
                     ray_aim_point.x * sensor_x_axis.z + ray_aim_point.y * sensor_y_axis.z + sensor_z_axis.z);
}

}  // namespace corecast::optix
