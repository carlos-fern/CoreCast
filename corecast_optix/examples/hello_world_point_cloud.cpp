#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "corecast_optix/corecast_optix.hpp"
#include "corecast_processing/corecast_depth_map.hpp"
#include "hello_world_point_cloud_helpers.hpp"

int main() {
  std::cout << "Starting CoreCastOptix pointcloud example" << std::endl;

  // OptiX setup
  OptixDeviceContextOptions options = {};
  options.logCallbackLevel = 4;
  CUcontext cuCtx = 0;

  corecast::optix::CoreCastOptix optix(cuCtx, options);
  std::cout << "CoreCastOptix created" << std::endl;

  // Build an enormous front-facing cloud (both spatially and in point count).
  std::vector<corecast::optix::PointXYZI> incoming_pointcloud_points;
  incoming_pointcloud_points.reserve(1001 * 1001);
  constexpr float point_spacing = 0.05f;
  for (int yi = -500; yi <= 500; ++yi) {
    for (int xi = -500; xi <= 500; ++xi) {
      corecast::optix::PointXYZI p{};
      p.x = static_cast<float>(xi) * point_spacing;
      p.y = static_cast<float>(yi) * point_spacing;
      // Slight depth staggering improves contrast in the depth image.
      p.z = ((xi + yi) & 1) ? 20.0f : 20.25f;
      p.intensity = 1.0f;
      p.ring = 0;
      p.timestampOffset = 0.0;
      incoming_pointcloud_points.push_back(p);
    }
  }
  std::cout << "Generated point count: " << incoming_pointcloud_points.size() << std::endl;

  constexpr float point_radius = 0.01f;

  corecast::processing::CoreCastDepthMap<corecast::optix::PointXYZI> depth_map_processor(optix);
  depth_map_processor.setup_depth_map_processing(incoming_pointcloud_points,
                                                 /*image_width=*/1920,
                                                 /*image_height=*/1080,
                                                 /*point_radius=*/point_radius);

  const auto gpu_start = std::chrono::high_resolution_clock::now();
  auto* depth = depth_map_processor.launch_depth_map();
  const auto gpu_end = std::chrono::high_resolution_clock::now();
  const double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

  const auto& params = depth_map_processor.get_launch_params();
  std::cout << "Depth sample [0]: " << depth[0] << std::endl;
  const std::string depthmap_path = "pointcloud_depthmap.pgm";
  corecast::examples::write_depthmap_pgm(depthmap_path, depth, params.image_width, params.image_height, params.t_max);
  std::cout << "Saved depth map to " << depthmap_path << std::endl;
  const std::string depthmap_f32_path = "pointcloud_depthmap.f32";
  corecast::examples::write_depthmap_f32(depthmap_f32_path, depth, params.image_width, params.image_height);
  std::cout << "Saved float depth map to " << depthmap_f32_path << std::endl;

  std::vector<float> cpu_depth;
  const auto cpu_start = std::chrono::high_resolution_clock::now();
  corecast::examples::render_depth_map_cpu(incoming_pointcloud_points, point_radius, params, &cpu_depth);
  const auto cpu_end = std::chrono::high_resolution_clock::now();
  const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  const double speedup = gpu_ms > 0.0 ? (cpu_ms / gpu_ms) : 0.0;

  std::cout << "Performance (render only): GPU=" << gpu_ms << " ms, CPU=" << cpu_ms << " ms, speedup=" << speedup << "x"
            << std::endl;
  corecast::examples::print_depth_diff_stats(depth, cpu_depth, params.t_max);

  return 0;
}
