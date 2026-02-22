#include "corecast_optix/corecast_optix_module.hpp"

#include <optix_stubs.h>

#include <fstream>
#include <stdexcept>
#include <vector>

namespace corecast_optix
{

namespace
{
std::vector<char> read_file_bytes(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open CUDA source file: " + path);
    }
    return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

}

CoreCastOptixModule::CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context, OptixPipelineCompileOptions& pipeline_compile_options, OptixModuleCompileOptions& module_compile_options) : context_(context), pipeline_compile_options_(pipeline_compile_options), module_compile_options_(module_compile_options)
{
    std::vector<char> ptx = read_file_bytes(CORECAST_PTX_PATH);

    OPTIX_CHECK_LOG( optixModuleCreate(
                context_->get_context(),
                &module_compile_options_,
                &pipeline_compile_options_,
                ptx.data(),
                ptx.size(),
                LOG, &LOG_SIZE,
                &module_
                ) );
}

CoreCastOptixModule::~CoreCastOptixModule() = default;

}  // namespace corecast_optix
