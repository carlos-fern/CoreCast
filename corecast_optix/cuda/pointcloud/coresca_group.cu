#include <cuda_runtime.h>

#include "corecast_optix/corecast_optix_cuda_types.hpp"

extern "C" {
__constant__ corecast::optix::CoreSACGroupParams params;
}

__device__ uint32_t voxel_hash(float3 point, uint32_t max_num_total_voxels) {

  const int ix = static_cast<int>(floorf(point.x / params.voxel_size));
  const int iy = static_cast<int>(floorf(point.y / params.voxel_size));
  const int iz = static_cast<int>(floorf(point.z / params.voxel_size));
  const uint32_t h = (uint32_t)(ix * 73856093) ^ (uint32_t)(iy * 19349663) ^ (uint32_t)(iz * 83492791);
  return h % max_num_total_voxels;
}

extern "C" __global__ void coresac_group(const corecast::optix::CoreSACGroupParams params) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= params.hit_point_count) return;

  const float3 point = params.hit_points[i];

  // Hash integer cell to voxel index in [0, num_voxels)
  const uint32_t voxel_id = voxel_hash(point, params.max_num_total_voxels);

  // Get next free slot in this voxel's point list
  const uint32_t slot = atomicAdd(&params.voxel_point_count[voxel_id], 1u);

  // Write into voxel_points[voxel_id][slot], drop if over capacity
  if (slot < params.max_points_per_voxel) {
    params.voxel_points[voxel_id * params.max_points_per_voxel + slot] = point;
    params.active_voxel_ids[i] = voxel_id;
  }
  else{
    printf("Voxel %d is full, dropping point\n", voxel_id);
  }
}
