/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <optix.h>
#include "corecast_optix/corecast_optix_cuda_types.hpp"
#include <cuda/helpers.h>

using corecast_optix::Params;
using corecast_optix::RayGenData;

extern "C" {
__constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float dx = static_cast<float>(launch_index.x) - rtData->circle_center_x;
    const float dy = static_cast<float>(launch_index.y) - rtData->circle_center_y;
    const float radius_sq = rtData->circle_radius * rtData->circle_radius;
    const bool in_circle = (dx * dx + dy * dy) <= radius_sq;
    const float3 color = in_circle ? make_float3(0.0f, 0.0f, 0.0f) : make_float3(rtData->r, rtData->g, rtData->b);
    params.data[launch_index.y * params.image_width + launch_index.x] = make_color(color);
}
