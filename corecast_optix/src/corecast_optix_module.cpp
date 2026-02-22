#include "corecast_optix/corecast_optix_module.hpp"


namespace corecast_optix
{

CoreCastOptixModule::CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context) : context_(context)
{
    #if OPTIX_DEBUG_DEVICE_CODE
    module_compile_options_.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    pipeline_compile_options_.usesMotionBlur        = false;
    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options_.numPayloadValues      = 2;
    pipeline_compile_options_.numAttributeValues    = 2;
    pipeline_compile_options_.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options_.pipelineLaunchParamsSizeInBytes = sizeof( Params );

    size_t      inputSize = 0;
    const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "draw_solid_color.cu", inputSize );

    OPTIX_CHECK_LOG( optixModuleCreate(
                context_->get_context(),
                &module_compile_options_,
                &pipeline_compile_options_,
                input,
                inputSize,
                LOG, &LOG_SIZE,
                &module_
                ) );
}

CoreCastOptixModule::~CoreCastOptixModule() = default;

}  // namespace corecast_optix
