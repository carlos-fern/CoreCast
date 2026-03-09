#include <optix_device.h>

#include "corecast_optix_cuda_types.hpp"

// Declare the global parameters struct so the GPU can access it
extern "C" {
__constant__ PointCloudLaunchParams params;
}

inline __device__ float3 normalize_vec(float3 v) {
  float inv_length = rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
  return make_float3(v.x * inv_length, v.y * inv_length, v.z * inv_length);
}

extern "C" __global__ void __raygen__point_cloud() {
  const uint3 pixel_coordiante = optixGetLaunchIndex();  // Get pixel coordiante i.e x,y
  const uint3 resolution = optixGetLaunchDimensions();   // get total resolution i.e width, height
  uint32_t payload = 0;

  // Range normalized coordiates to [-1.0, 1.0]
  float2 ray_aim_point =
      make_float2((2.0f * static_cast<float>(pixel_coordiante.x) + 1.0f) / static_cast<float>(resolution.x) - 1.0f,
                  (2.0f * static_cast<float>(pixel_coordiante.y) + 1.0f) / static_cast<float>(resolution.y) - 1.0f);

  // Calculate ray direction using the basis vectors
  float3 ray_direction = make_float3(
      ray_aim_point.x * params.sensor_x_axis.x + ray_aim_point.y * params.sensor_y_axis.x + params.sensor_z_axis.x,
      ray_aim_point.x * params.sensor_x_axis.y + ray_aim_point.y * params.sensor_y_axis.y + params.sensor_z_axis.y,
      ray_aim_point.x * params.sensor_x_axis.z + ray_aim_point.y * params.sensor_y_axis.z + params.sensor_z_axis.z);

  optixTrace(params.handle, params.sensor_origin, ray_direction, params.t_min, params.t_max, 0.0f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, payload);

  // 7. Unpack the returned distance and write it to the global output buffer
  float true_distance = __uint_as_float(payload);

  // Calculate 1D index from 2D pixel coordinates
  uint32_t linear_index = pixel_coordiante.y * resolution.x + pixel_coordiante.x;
  params.depth_buffer[linear_index] = true_distance;
}