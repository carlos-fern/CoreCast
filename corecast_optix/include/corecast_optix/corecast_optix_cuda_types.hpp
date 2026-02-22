#pragma once

#include <cuda_runtime.h>

namespace corecast_optix
{

struct Params
{
    uchar4* image;
    unsigned int image_width;
    unsigned int image_height;
};

struct RayGenData
{
    float r, g, b;
    float circle_center_x;
    float circle_center_y;
    float circle_radius;
};

}  // namespace corecast_optix
