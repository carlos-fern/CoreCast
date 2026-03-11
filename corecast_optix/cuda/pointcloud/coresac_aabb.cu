#include <optix_device.h>
#include <limits>

#include "corecast_optix/corecast_optix_cuda_types.hpp"

extern "C" {
__constant__ corecast::optix::CoreSACAABBParams params;
}
 
extern "C" __global__ void __raygen__coresac_aabb() {
    auto id = blockIdx.x * blockDim.x + threadIdx.x;

    const uint32_t voxel_id = params.active_voxel_ids[id];
    if (voxel_id >= params.max_num_total_voxels) {
        return;
    } else if (params.voxel_point_count[voxel_id] < params.min_points_per_voxel){
        return;
    }

    auto voxel_count = params.voxel_point_count[voxel_id];
    const uint32_t point_limit = voxel_count < params.max_points_per_voxel ? voxel_count : params.max_points_per_voxel;

    params.bounding_boxes[id].min.x = std::numeric_limits<float>::max();
    params.bounding_boxes[id].min.y = std::numeric_limits<float>::max();
    params.bounding_boxes[id].min.z = std::numeric_limits<float>::max();
    params.bounding_boxes[id].max.x = -std::numeric_limits<float>::max();
    params.bounding_boxes[id].max.y = -std::numeric_limits<float>::max();
    params.bounding_boxes[id].max.z = -std::numeric_limits<float>::max();

    for (uint32_t i = 0; i < point_limit; i++) {
        params.bounding_boxes[id].min.x = fminf(params.bounding_boxes[id].min.x, params.voxel_points[voxel_id * params.max_points_per_voxel + i].x);
        params.bounding_boxes[id].min.y = fminf(params.bounding_boxes[id].min.y, params.voxel_points[voxel_id * params.max_points_per_voxel + i].y);
        params.bounding_boxes[id].min.z = fminf(params.bounding_boxes[id].min.z, params.voxel_points[voxel_id * params.max_points_per_voxel + i].z);
        params.bounding_boxes[id].max.x = fmaxf(params.bounding_boxes[id].max.x, params.voxel_points[voxel_id * params.max_points_per_voxel + i].x);
        params.bounding_boxes[id].max.y = fmaxf(params.bounding_boxes[id].max.y, params.voxel_points[voxel_id * params.max_points_per_voxel + i].y);
        params.bounding_boxes[id].max.z = fmaxf(params.bounding_boxes[id].max.z, params.voxel_points[voxel_id * params.max_points_per_voxel + i].z);
        
        params.bouding_box_to_voxel_id[id] = voxel_id;
    }
}