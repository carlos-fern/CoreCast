

#include <optix.h>
#include "corecast_optix/corecast_optix_cuda_types.hpp"
#include <cuda/helpers.h>
#include <cmath>

using corecast::optix::PointCloudParams;
using corecast::optix::PointCloudRayGenData;
using corecast::optix::PointCloudHitData;
using corecast::optix::PointCloudMissData;
using corecast::optix::PointCloudRayPayload;

extern "C" {
__constant__ PointCloudParams params;
}

extern "C"
__global__ void __raygen__extract_2d_plane()
{
    uint3 launch_index = optixGetLaunchIndex();

    if (launch_index.x >=  params.num_points) { // check if the launch index is out of bounds
        return;
    }

    const corecast::optix::PointXYZI point = params.data[launch_index.x];
    const bool inlier = fabsf(point.z - params.plane_z) < params.distance_threshold;
    params.inlier_mask[launch_index.x] = inlier ? 1 : 0;
}