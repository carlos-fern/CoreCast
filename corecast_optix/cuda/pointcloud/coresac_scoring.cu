#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "corecast_optix/corecast_optix_cuda_types.hpp"
#include "pointcloud_ray_utils.cuh"

extern "C" {
__constant__ corecast::optix::CoreSACScoringParams params;
}


__device__ void gen_random_int(int* output, int seed) {

    curandState state;

    curand_init(1337, seed, 0, &state);

    float random_val = curand_uniform(&state);

    int value = (int)(random_val * (float)params.aabb_params.max_num_total_voxels);
    if (value >= (int)params.aabb_params.max_num_total_voxels) {
        value = (int)params.aabb_params.max_num_total_voxels - 1;
    }
    *output = value;
}

extern "C" __global__ void __raygen__coresac_scoring() {
    const uint3 pixel_coordinate = optixGetLaunchIndex();
    const uint3 resolution = optixGetLaunchDimensions();
    uint32_t payload = 0;

    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= active_voxel_count) {
        return;
    }

    uint32_t voxel_id = params.aabb_params.active_voxel_ids[id];
    if (voxel_id >= params.aabb_params.max_num_total_voxels) {
        return;
    }

    int random_point;
    gen_random_int(&random_point, id);

    float3 origin = params.aabb_params.voxlel_points[voxel_id * params.aabb_params.max_points_per_voxel + random_point];

    float2 ray_aim_point = corecast::optix::pixel_to_ray_aim_point(pixel_coordinate, resolution); 
    float3 direction = corecast::optix::ray_aim_to_direction(ray_aim_point, params.aabb_params.sensor_x_axis, params.aabb_params.sensor_y_axis, params.aabb_params.sensor_z_axis); 
    
    auto t_min = params.aabb_params.bounding_boxes[params.aabb_params.bouding_box_to_voxel_id[id]].min.x; 
    auto t_max = params.aabb_params.bounding_boxes[params.aabb_params.bouding_box_to_voxel_id[id]].max.x;

    optixTrace(params.aabb_params.handle, origin, direction, t_min, t_max, 0.0f,
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, 0, 1, 0, payload);

}