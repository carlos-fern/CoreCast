#include "corecast_optix/corecast_optix.hpp"
#include <optixHello/optixHello.h>

int main()
{

    corecast_optix::CoreCastOptix optix;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineLinkOptions link_options = {};
    corecast_optix::Params params = {};
    params.image_width = 1024;
    params.image_height = 1024;
    params.image = new uchar4[params.image_width * params.image_height];

    corecast_optix::CoreCastProgram raygen_program = {};
    raygen_program.name = "raygen";
    raygen_program.options = {};
    raygen_program.desc = {};

    std::string pipeline_name = "raygen_pipeline";
    std::string module_name = "raygen_module";
    std::string sbt_name = "raygen_sbt";
    RayGenData data = {0.462f, 0.725f, 0.f};
    std::vector<std::string> program_names = {raygen_program.name};

    optix.create_module(raygen_program.name, pipeline_compile_options, module_compile_options);
    optix.add_program_to_module(module_name, raygen_program);
    optix.build_pipeline(pipeline_name, program_names, link_options);
    optix.create_sbt<corecast_optix::SbtRecord<RayGenData>, RayGenData>(sbt_name, raygen_program.name, data);
    optix.launch_pipeline(pipeline_name, params, sbt_name);


    return 0;
}