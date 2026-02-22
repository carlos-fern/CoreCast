#include "corecast_optix/corecast_optix_context.hpp"
#include <optix_stubs.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace corecast_optix
{

namespace
{
void check_optix(OptixResult result, const char* expr)
{
  if (result != OPTIX_SUCCESS) {
    std::ostringstream oss;
    oss << "OptiX call failed (" << static_cast<int>(result) << "): " << expr;
    throw std::runtime_error(oss.str());
  }
}
}  // namespace

CoreCastOptixContext::CoreCastOptixContext()
{
  CUcontext cuCtx = 0;  // zero means take the current context
  check_optix(optixInit(), "optixInit()");
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &context_log_cb;
  options.logCallbackLevel = 4;
  check_optix(optixDeviceContextCreate(cuCtx, &options, &this->context_), "optixDeviceContextCreate()");
}

CoreCastOptixContext::~CoreCastOptixContext()
{
  if (context_ != nullptr) {
    optixDeviceContextDestroy(context_);
  }
}

void CoreCastOptixContext::context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/)
{
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

}  // namespace corecast_optix
