#pragma once

#include <cuda_runtime.h>
#include <cuda/std/variant>
#include <cstdint>

namespace corecast_optix
{

// Helloworld image types TODO: Remove this in favor of pointcloud types
struct Params
{
    uchar4* data;
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

// Core pointcloud type
struct __attribute__((packed)) PointXYZI {
    float x;         
    float y;         
    float z;         
    float intensity; 
    uint16_t ring;  
    double timestampOffset;
};

struct PointCloudParams
{
    PointXYZI* data;
    uint32_t   num_points;
    OptixTraversableHandle traversable;
};

// Global raygen data (The map)
struct PointCloudRayGenData
{
    PointXYZI* data;
    unsigned int num_points;
};

struct PointCloudHitData{
    float intensity;
    uint16_t ring;
    double timestampOffset;
};

struct PointCloudMissData{
    float empty_color;
};

struct PointCloudRayPayload{
    PointXYZI point;
};








}  // namespace corecast_optix
