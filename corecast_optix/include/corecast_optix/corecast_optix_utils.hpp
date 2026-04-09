#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <string>
#include <vector>

namespace corecast::optix {
/**
 * @brief Check if the CUDA call was successful.
 * @param result The result of the CUDA call.
 * @param expr The expression that was evaluated.
 */
void check_cuda(cudaError_t result, const char* expr);

/**
 * @brief Check if the OptiX call was successful.
 * @param result The result of the OptiX call.
 * @param expr The expression that was evaluated.
 */
void check_optix(OptixResult result, const char* expr);

/**
 * @brief Read the bytes of a file.
 * @param path The path to the file.
 * @return The bytes of the file.
 */
std::vector<char> read_file_bytes(const std::string& path);

struct float3 {
  float x, y, z;
};

inline OptixBuildInput configure_point_cloud_input(const std::vector<float3>& host_points, float point_radius) {
  // Allocate and copy the point coordinates (centers) to the GPU
  CUdeviceptr d_points = 0;
  size_t points_size = host_points.size() * sizeof(float3);
  cudaMalloc(reinterpret_cast<void**>(&d_points), points_size);
  cudaMemcpy(reinterpret_cast<void*>(d_points), host_points.data(), points_size, cudaMemcpyHostToDevice);

  // Allocate and copy the SINGLE radius to the GPU
  CUdeviceptr d_radius = 0;
  cudaMalloc(reinterpret_cast<void**>(&d_radius), sizeof(float));
  cudaMemcpy(reinterpret_cast<void*>(d_radius), &point_radius, sizeof(float), cudaMemcpyHostToDevice);

  // OptiX requires an array of device pointers for vertices and radii
  // (This is to support motion blur, but we only need 1 array element for static points)
  // NOTE: In a real class, store these arrays as class members so they don't go out of scope!
  static CUdeviceptr d_vertex_buffers[1] = {d_points};
  static CUdeviceptr d_radius_buffers[1] = {d_radius};

  //  Set up geometry flags (tells OptiX if the geometry has transparency, etc.)
  static uint32_t sphere_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  // Configure the Build Input
  OptixBuildInput sphere_input = {};
  sphere_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;

  OptixBuildInputSphereArray& sphere_array = sphere_input.sphereArray;

  // --- Assign Vertices (The Points) ---
  sphere_array.vertexBuffers = d_vertex_buffers;
  sphere_array.vertexStrideInBytes = sizeof(float3);
  sphere_array.numVertices = host_points.size();

  // --- Assign Radii ---
  sphere_array.radiusBuffers = d_radius_buffers;
  sphere_array.radiusStrideInBytes = 0;  // 0 because there is no array to stride through
  sphere_array.singleRadius = 1;         // Crucial: Tells OptiX to use d_radius for ALL points

  // --- Assign Flags and SBT ---
  sphere_array.flags = sphere_flags;
  sphere_array.numSbtRecords = 1;

  return sphere_input;
}

}  // namespace corecast::optix