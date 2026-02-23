#pragma once

#include <cuda_runtime.h>
#include <cuda/std/variant>

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
    std::vector<PointXYZI>* data; 
    unsigned int num_points;
};

// Global raygen data (The map)
struct PointCloudRayGenData
{
    std::vector<PointXYZI>* data;
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
    float x;
    float y;
    float z;
    float intensity;
    uint16_t ring;
    double timestampOffset;
};








}  // namespace corecast_optix
