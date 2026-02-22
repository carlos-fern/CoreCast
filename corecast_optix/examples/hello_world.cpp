#include "corecast_optix/corecast_optix.hpp"
#include <optixHello/optixHello.h>

int main()
{

    std::cout << "Starting CoreCastOptix example" << std::endl;
    
    corecast_optix::CoreCastOptix optix;

    std::cout << "CoreCastOptix created" << std::endl;


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
    raygen_program.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program.desc.raygen.entryFunctionName = "__raygen__draw_solid_color";

    std::string pipeline_name = "raygen_pipeline";
    std::string module_name = raygen_program.name;
    std::string sbt_name = "raygen_sbt";
    RayGenData data = {0.462f, 0.725f, 0.f};
    std::vector<std::string> program_names = {raygen_program.name};

    std::cout << "Creating module" << std::endl;
    optix.create_module(module_name, pipeline_compile_options, module_compile_options);
    std::cout << "Adding program to module" << std::endl;
    optix.add_program_to_module(module_name, raygen_program);
    std::cout << "Building pipeline" << std::endl;
    optix.build_pipeline(pipeline_name, program_names, link_options);
    std::cout << "Creating SBT" << std::endl;
    optix.create_sbt<corecast_optix::SbtRecord<RayGenData>, RayGenData>(sbt_name, raygen_program.name, data);
    std::cout << "Launching pipeline" << std::endl;
    optix.launch_pipeline(pipeline_name, params, sbt_name);


    return 0;
}