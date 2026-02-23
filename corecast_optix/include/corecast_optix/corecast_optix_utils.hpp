#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <string>
#include <vector>

namespace corecast_optix
{
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

}  // namespace corecast_optix