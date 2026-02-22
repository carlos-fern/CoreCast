#include "corecast_optix/corecast_optix_module.hpp"


namespace corecast_optix
{

CoreCastOptixModule::CoreCastOptixModule(std::shared_ptr<CoreCastOptixContext> context, OptixPipelineCompileOptions& pipeline_compile_options, OptixModuleCompileOptions& module_compile_options) : context_(context), pipeline_compile_options_(pipeline_compile_options), module_compile_options_(module_compile_options)
{
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
