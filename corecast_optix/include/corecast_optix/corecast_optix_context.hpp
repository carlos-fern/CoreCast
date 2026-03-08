#pragma once

#include <optix.h>
#include <cuda_runtime.h>

namespace corecast::optix
{

class CoreCastOptixContext
{
public:
  /**
  * @brief Constructor for the CoreCastOptixContext class.
  * @param context_id The CUDA context ID to use.
  * @param options The options for the OptixDeviceContext.
  */
  CoreCastOptixContext(CUcontext context_id, OptixDeviceContextOptions& options);

  /**
  * @brief Destructor for the CoreCastOptixContext class.
  */
  ~CoreCastOptixContext();

  /**
  * @brief Get the OptixDeviceContext.
  * @return The OptixDeviceContext.
  */
  OptixDeviceContext get_context() const { return context_; }

  /**
  * @brief Static callback function for the OptixDeviceContext.
  * @param level The level of the log message.
  * @param tag The tag of the log message.
  * @param message The message of the log message.
  * @param cbdata The callback data.
  */
  static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata*/);

  private:
  OptixDeviceContext context_ = nullptr;
  CUcontext cuCtx_ = 0;
  OptixDeviceContextOptions options_ = {};
};

}  // namespace corecast::optix
