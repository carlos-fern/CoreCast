#include "corecast_optix/corecast_optix_program_registry.hpp"

namespace corecast_optix
{

CoreCastOptixProgramRegistry::CoreCastOptixProgramRegistry(std::shared_ptr<CoreCastOptixContext> context) : context_(context)
{
}

CoreCastOptixProgramRegistry::~CoreCastOptixProgramRegistry() = default;

void CoreCastOptixProgramRegistry::register_program(const CoreCastProgram& program, CoreCastOptixModule& module)
{
    program_descriptions_[program.name] = program;

    OptixProgramGroupDesc& description = program_descriptions_[program.name].desc;

    if (description.kind == OPTIX_PROGRAM_GROUP_KIND_RAYGEN) {
        description.raygen.module = module.get_module();
    } 
    else if (description.kind == OPTIX_PROGRAM_GROUP_KIND_MISS) {
        description.miss.module = module.get_module();
    }
    else if (description.kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP) {
        description.hitgroup.moduleCH = module.get_module();
    } else {
        throw std::runtime_error("Invalid program group kind");
    }

    OPTIX_CHECK_LOG( optixProgramGroupCreate(
        context_->get_context(),
        &program_descriptions_[program.name].desc,
        1,
        &program_descriptions_[program.name].options,
        LOG, &LOG_SIZE,
        &program_groups_[program.name]
    ) );

}

OptixProgramGroup CoreCastOptixProgramRegistry::get_program_group(const std::string& name) const
{
    return program_groups_.at(name);
}

std::vector<OptixProgramGroup> CoreCastOptixProgramRegistry::get_program_groups(const std::vector<std::string>& names) const
{
    std::vector<OptixProgramGroup> groups;
    for (const auto& name : names) {
        groups.push_back(program_groups_.at(name));
    }
    return groups;
}
}  // namespace corecast_optix
