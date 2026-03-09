#pragma once

#include <optix.h>

#include <string>
#include <vector>

#include "corecast_optix/corecast_optix_cuda_types.hpp"

namespace corecast::examples {

void render_depth_map_cpu(const std::vector<corecast::optix::PointXYZI>& points, float point_radius,
                          const corecast::optix::PointCloudLaunchParams& params, std::vector<float>* out_depth);

void print_depth_diff_stats(const float* gpu_depth, const std::vector<float>& cpu_depth, float t_max);

void write_depthmap_pgm(const std::string& output_path, const float* depth, unsigned int width, unsigned int height,
                        float miss_value);

void write_depthmap_f32(const std::string& output_path, const float* depth, unsigned int width, unsigned int height);

}  // namespace corecast::examples
