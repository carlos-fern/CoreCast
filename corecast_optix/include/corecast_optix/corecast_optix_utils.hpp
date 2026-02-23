#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace corecast_optix {

void check_cuda(cudaError_t result, const char *expr) {
  if (result != cudaSuccess) {
    std::ostringstream oss;
    oss << "CUDA call failed (" << static_cast<int>(result) << "): " << expr
        << " - " << cudaGetErrorString(result);
    throw std::runtime_error(oss.str());
  }
}

void check_optix(OptixResult result, const char *expr) {
  if (result != OPTIX_SUCCESS) {
    std::ostringstream oss;
    oss << "OptiX call failed (" << static_cast<int>(result) << "): " << expr;
    throw std::runtime_error(oss.str());
  }
}

} // namespace corecast_optix