#include "corecast_optix/corecast_optix.hpp"
#include <optixHello/optixHello.h>
#include <fstream>
#include <vector>

int main()
{

    std::cout << "Starting CoreCastOptix example" << std::endl;

    OptixDeviceContextOptions options = {};
    options.logCallbackLevel = 4;  
    CUcontext cuCtx = 0;
    corecast_optix::CoreCastOptix optix(cuCtx, options);

    std::cout << "CoreCastOptix created" << std::endl;


    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues = 0;
    pipeline_compile_options.numAttributeValues = 0;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    OptixModuleCompileOptions module_compile_options = {};
    OptixPipelineLinkOptions link_options = {};
    corecast_optix::Params params = {};
    params.image_width = 1024;
    params.image_height = 1024;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.image), params.image_width * params.image_height * sizeof(uchar4)));

    corecast_optix::CoreCastProgram raygen_program = {};
    raygen_program.name = "raygen";
    raygen_program.options = {};
    raygen_program.desc = {};
    raygen_program.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program.desc.raygen.entryFunctionName = "__raygen__draw_solid_color";

    std::string pipeline_name = "raygen_pipeline";
    std::string module_name = raygen_program.name;
    std::string sbt_name = "raygen_sbt";
    RayGenData data = {0.982f, 0.725f, 0.f};
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

    std::vector<uchar4> host_pixels(params.image_width * params.image_height);
    CUDA_CHECK(cudaMemcpy(
        host_pixels.data(),
        params.image,
        host_pixels.size() * sizeof(uchar4),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.image)));

    const char* output_path = "corecast_output.ppm";
    std::ofstream out(output_path, std::ios::binary);
    out << "P6\n" << params.image_width << " " << params.image_height << "\n255\n";
    for (const auto& px : host_pixels) {
        out.write(reinterpret_cast<const char*>(&px.x), 1);
        out.write(reinterpret_cast<const char*>(&px.y), 1);
        out.write(reinterpret_cast<const char*>(&px.z), 1);
    }
    out.close();
    std::cout << "Saved output image to " << output_path << std::endl;


    return 0;
}