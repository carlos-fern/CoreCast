#include "corecast_optix/corecast_optix_context.hpp"

namespace corecast_optix
{

CoreCastOptixContext::CoreCastOptixContext(){

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &this->context_ ) );
}

CoreCastOptixContext::~CoreCastOptixContext() = default;

void CoreCastOptixContext::context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

}  // namespace corecast_optix
