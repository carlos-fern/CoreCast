#include <fstream>
#include <vector>

#include "corecast_optix/corecast_optix.hpp"

void write_ppm(corecast::optix::Params& params, const uchar4* host_pixels) {
  const char* output_path = "corecast_output.ppm";
  std::ofstream out(output_path, std::ios::binary);
  out << "P6\n" << params.image_width << " " << params.image_height << "\n255\n";
  const size_t num_pixels = static_cast<size_t>(params.image_width) * params.image_height;
  for (size_t i = 0; i < num_pixels; ++i) {
    const auto& px = host_pixels[i];
    out.write(reinterpret_cast<const char*>(&px.x), 1);
    out.write(reinterpret_cast<const char*>(&px.y), 1);
    out.write(reinterpret_cast<const char*>(&px.z), 1);
  }
  out.close();
  std::cout << "Saved output image to " << output_path << std::endl;
}

int main() {
  std::cout << "Starting CoreCastOptix example" << std::endl;

  // Set the OptiX device context options
  OptixDeviceContextOptions options = {};
  options.logCallbackLevel = 4;

  // Choose the CUDA context to use
  CUcontext cuCtx = 0;

  // Create the CoreCastOptix instance
  corecast::optix::CoreCastOptix optix(cuCtx, options);
  std::cout << "CoreCastOptix created" << std::endl;

  // Set the pipeline compile options
  OptixPipelineCompileOptions pipeline_compile_options = {};
  pipeline_compile_options.usesMotionBlur = false;
  pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
  pipeline_compile_options.numPayloadValues = 0;
  pipeline_compile_options.numAttributeValues = 0;
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  // Set the module compile options
  OptixModuleCompileOptions module_compile_options = {};

  // Set the pipeline link options
  OptixPipelineLinkOptions link_options = {};

  // Set the parameters for the pipeline
  corecast::optix::Params params = {};
  params.image_width = 1024;
  params.image_height = 1024;

  std::vector<uchar4> host_pixels(params.image_width * params.image_height);
  std::vector<uchar4> host_pixels_output(params.image_width * params.image_height);

  corecast::optix::CUDABuffer<uchar4, uchar4> host_pixels_buffer(
      host_pixels.data(), static_cast<int>(params.image_width * params.image_height), host_pixels_output.data());
  params.data = host_pixels_buffer.get_device_ptr();

  corecast::optix::CoreCastProgram raygen_program = {};
  raygen_program.name = "raygen";
  raygen_program.options = {};
  raygen_program.desc = {};
  raygen_program.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_program.desc.raygen.entryFunctionName = "__raygen__draw_solid_color";

  std::string pipeline_name = "raygen_pipeline";
  std::string module_name = raygen_program.name;
  std::string sbt_name = "raygen_sbt";
  corecast::optix::RayGenData data = {0.982f,
                                      0.725f,
                                      65.0f,
                                      static_cast<float>(params.image_width) * 0.5f,
                                      static_cast<float>(params.image_height) * 0.5f,
                                      static_cast<float>(params.image_width) / 6.0f};

  std::vector<std::string> program_names = {raygen_program.name};

  std::cout << "Creating module" << std::endl;
  std::string ptx_path = CORECAST_PTX_PATH;
  optix.create_module(module_name, pipeline_compile_options, module_compile_options, ptx_path);

  std::cout << "Adding program to module" << std::endl;
  optix.add_program_to_module(module_name, raygen_program);

  std::cout << "Building pipeline" << std::endl;
  optix.build_pipeline(pipeline_name, program_names, link_options);

  std::cout << "Creating SBT" << std::endl;
  optix.create_sbt<corecast::optix::SbtRecord<corecast::optix::RayGenData>, corecast::optix::RayGenData>(
      sbt_name, raygen_program.name, data);

  std::cout << "Launching pipeline" << std::endl;
  corecast::optix::Params launch_params = params;
  optix.launch_pipeline(pipeline_name, launch_params, sbt_name);

  std::cout << "Getting result" << std::endl;
  write_ppm(params, optix.get_result<uchar4, uchar4>(pipeline_name, host_pixels_buffer));

  return 0;
}