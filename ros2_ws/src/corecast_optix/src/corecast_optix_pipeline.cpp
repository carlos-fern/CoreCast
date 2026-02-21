#include "corecast_optix/corecast_optix_pipeline.hpp"

namespace corecast_optix
{

CoreCastOptixPipeline::CoreCastOptixPipeline(std::shared_ptr<CoreCastOptixContext> context, std::shared_ptr<CoreCastOptixProgramRegistry> program_registry, std::shared_ptr<CoreCastOptixModule> module, OptixPipelineLinkOptions& link_options, std::vector<std::string>& program_names) : context_(context), program_registry_(program_registry), module_(module), link_options_(link_options), program_names_(program_names)
{
    std::vector<OptixProgramGroup> groups = program_registry_->get_program_groups(program_names);

    OPTIX_CHECK_LOG( optixPipelineCreate(
        context_->get_context(),
        &module_->get_pipeline_compile_options(),
        &link_options_,
        groups.data(),
        static_cast<uint32_t>(groups.size()),
        LOG, &LOG_SIZE,
        &pipeline_
    ) );
}

CoreCastOptixPipeline::~CoreCastOptixPipeline() = default;





}  // namespace corecast_optix