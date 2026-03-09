#include <optix_device.h>

#include "corecast_optix/corecast_optix_cuda_types.hpp"

extern "C" __global__ void __closesthit__point_cloud() {
  // Requires raygen to normalize the direction vector
  float distance = optixGetRayTmax();

  optixSetPayload_0(__float_as_uint(distance));
}

extern "C" __global__ void __miss__point_cloud() {
  // We hit nothing so return 0
  optixSetPayload_0(__float_as_uint(0.0f));
}