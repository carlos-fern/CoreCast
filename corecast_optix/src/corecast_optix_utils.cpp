#include "corecast_optix/corecast_optix_utils.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace corecast_optix
{

void check_cuda(cudaError_t result, const char* expr)
{
    if (result != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA call failed (" << static_cast<int>(result) << "): " << expr
            << " - " << cudaGetErrorString(result);
        throw std::runtime_error(oss.str());
    }
}

void check_optix(OptixResult result, const char* expr)
{
    if (result != OPTIX_SUCCESS) {
        std::ostringstream oss;
        oss << "OptiX call failed (" << static_cast<int>(result) << "): " << expr;
        throw std::runtime_error(oss.str());
    }
}

std::vector<char> read_file_bytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open CUDA source file: " + path);
    }
    return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

}  // namespace corecast_optix
