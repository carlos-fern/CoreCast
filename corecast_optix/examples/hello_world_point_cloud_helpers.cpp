#include "hello_world_point_cloud_helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace corecast::examples {

namespace {

inline float dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

}  // namespace

void render_depth_map_cpu(const std::vector<corecast::optix::PointXYZI>& points, float point_radius,
                          const corecast::optix::PointCloudLaunchParams& params, std::vector<float>* out_depth) {
  const unsigned int width = params.image_width;
  const unsigned int height = params.image_height;
  out_depth->assign(static_cast<size_t>(width) * height, 0.0f);

  // Traditional CPU depth-map approach: project points and z-buffer.
  for (const auto& p : points) {
    const float3 world = make_float3(p.x, p.y, p.z);
    const float3 rel = make_float3(world.x - params.sensor_origin.x, world.y - params.sensor_origin.y,
                                   world.z - params.sensor_origin.z);

    const float x_cam = dot3(rel, params.sensor_x_axis);
    const float y_cam = dot3(rel, params.sensor_y_axis);
    const float z_cam = dot3(rel, params.sensor_z_axis);

    if (z_cam <= params.t_min || z_cam >= params.t_max) {
      continue;
    }

    const float aim_x = x_cam / z_cam;
    const float aim_y = y_cam / z_cam;
    const int cx = static_cast<int>((aim_x + 1.0f) * 0.5f * static_cast<float>(width));
    const int cy = static_cast<int>((aim_y + 1.0f) * 0.5f * static_cast<float>(height));
    if (cx < 0 || cx >= static_cast<int>(width) || cy < 0 || cy >= static_cast<int>(height)) {
      continue;
    }

    const float radius_px_f = std::max(1.0f, point_radius / z_cam * 0.5f * static_cast<float>(width));
    const int radius_px = static_cast<int>(std::ceil(radius_px_f));

    for (int oy = -radius_px; oy <= radius_px; ++oy) {
      for (int ox = -radius_px; ox <= radius_px; ++ox) {
        if (ox * ox + oy * oy > radius_px * radius_px) {
          continue;
        }
        const int px = cx + ox;
        const int py = cy + oy;
        if (px < 0 || px >= static_cast<int>(width) || py < 0 || py >= static_cast<int>(height)) {
          continue;
        }
        const size_t idx = static_cast<size_t>(py) * width + static_cast<size_t>(px);
        float& z = (*out_depth)[idx];
        if (z == 0.0f || z_cam < z) {
          z = z_cam;
        }
      }
    }
  }
}

void print_depth_diff_stats(const float* gpu_depth, const std::vector<float>& cpu_depth, float t_max) {
  const size_t count = cpu_depth.size();
  size_t compared = 0;
  double sum_abs = 0.0;
  double max_abs = 0.0;
  size_t gpu_hits = 0;
  size_t cpu_hits = 0;
  for (size_t i = 0; i < count; ++i) {
    const float g = gpu_depth[i];
    const float c = cpu_depth[i];
    if (g > 0.0f && g < t_max) {
      ++gpu_hits;
    }
    if (c > 0.0f && c < t_max) {
      ++cpu_hits;
    }
    if ((g > 0.0f && g < t_max) || (c > 0.0f && c < t_max)) {
      const double abs_err = std::abs(static_cast<double>(g) - static_cast<double>(c));
      sum_abs += abs_err;
      max_abs = std::max(max_abs, abs_err);
      ++compared;
    }
  }

  const double mae = compared > 0 ? (sum_abs / static_cast<double>(compared)) : 0.0;
  std::cout << "CPU/GPU depth comparison: compared=" << compared << ", gpu_hits=" << gpu_hits
            << ", cpu_hits=" << cpu_hits << ", mae=" << mae << ", max_abs_error=" << max_abs << std::endl;
}

void write_depthmap_pgm(const std::string& output_path, const float* depth, unsigned int width, unsigned int height,
                        float miss_value) {
  const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (count == 0 || depth == nullptr) {
    throw std::runtime_error("Depth buffer is empty");
  }

  float min_depth = std::numeric_limits<float>::max();
  float max_depth = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    const float d = depth[i];
    if (d <= 0.0f || d >= miss_value) {
      continue;
    }
    min_depth = std::min(min_depth, d);
    max_depth = std::max(max_depth, d);
  }

  std::vector<uint8_t> pixels(count, 0);
  if (max_depth > min_depth) {
    const float range = max_depth - min_depth;
    for (size_t i = 0; i < count; ++i) {
      const float d = depth[i];
      if (d <= 0.0f || d >= miss_value) {
        pixels[i] = 0;
        continue;
      }
      const float norm = (d - min_depth) / range;
      pixels[i] = static_cast<uint8_t>(255.0f * (1.0f - norm));
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      const float d = depth[i];
      if (d > 0.0f && d < miss_value) {
        pixels[i] = 255;
      }
    }
  }

  std::ofstream out(output_path, std::ios::binary);
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

void write_depthmap_f32(const std::string& output_path, const float* depth, unsigned int width, unsigned int height) {
  const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (count == 0 || depth == nullptr) {
    throw std::runtime_error("Depth buffer is empty");
  }

  std::ofstream out(output_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Failed to open output file: " + output_path);
  }

  out.write(reinterpret_cast<const char*>(&width), sizeof(width));
  out.write(reinterpret_cast<const char*>(&height), sizeof(height));
  out.write(reinterpret_cast<const char*>(depth), static_cast<std::streamsize>(count * sizeof(float)));
}

}  // namespace corecast::examples
