#include "corecast_optix/corecast_optix.hpp"
#include <iostream>
#include <random>
#include <vector>

namespace {

/**
 * Generates a synthetic pointcloud for testing the 2D plane extractor.
 * Creates points near z=0 (inliers) and z=1 (outliers) with random x,y.
 *
 * @param num_inliers Number of points near the plane (z ≈ 0)
 * @param num_outliers Number of points away from the plane (z ≈ 1)
 * @param xy_range Half-extent for x,y in [-xy_range, xy_range]
 * @return Vector of PointXYZI points
 */
std::vector<corecast_optix::PointXYZI> generate_test_pointcloud(
    size_t num_inliers,
    size_t num_outliers,
    float xy_range = 10.0f)
{
    std::vector<corecast_optix::PointXYZI> points;
    points.reserve(num_inliers + num_outliers);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> xy_dist(-xy_range, xy_range);
    std::uniform_real_distribution<float> z_inlier(-0.05f, 0.05f);   // near z=0
    std::uniform_real_distribution<float> z_outlier(0.9f, 1.1f);     // near z=1

    for (size_t i = 0; i < num_inliers; ++i) {
        corecast_optix::PointXYZI p{};
        p.x = xy_dist(rng);
        p.y = xy_dist(rng);
        p.z = z_inlier(rng);
        p.intensity = 1.0f;
        p.ring = static_cast<uint16_t>(i % 16);
        p.timestampOffset = 0.0;
        points.push_back(p);
    }

    for (size_t i = 0; i < num_outliers; ++i) {
        corecast_optix::PointXYZI p{};
        p.x = xy_dist(rng);
        p.y = xy_dist(rng);
        p.z = z_outlier(rng);
        p.intensity = 0.5f;
        p.ring = static_cast<uint16_t>(i % 16);
        p.timestampOffset = 0.0;
        points.push_back(p);
    }

    return points;
}

}  // namespace

int main()
{
    std::cout << "Starting CoreCastOptix pointcloud example" << std::endl;

    // Generate synthetic pointcloud
    const size_t num_inliers = 800;
    const size_t num_outliers = 200;
    auto host_points = generate_test_pointcloud(num_inliers, num_outliers);
    const uint32_t num_points = static_cast<uint32_t>(host_points.size());
    std::cout << "Generated " << num_points << " points (" << num_inliers
              << " inliers, " << num_outliers << " outliers)" << std::endl;

    // OptiX setup
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

    // Allocate device buffers
    corecast_optix::PointXYZI* d_points = nullptr;
    uint8_t* d_inlier_mask = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points), num_points * sizeof(corecast_optix::PointXYZI)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inlier_mask), num_points * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_points, host_points.data(), num_points * sizeof(corecast_optix::PointXYZI), cudaMemcpyHostToDevice));

    // Plane at z=0, threshold 0.1
    corecast_optix::PointCloudParams params = {};
    params.data = d_points;
    params.num_points = num_points;
    params.traversable = 0;
    params.inlier_mask = d_inlier_mask;
    params.plane_z = 0.0f;
    params.distance_threshold = 0.1f;

    corecast_optix::CoreCastProgram raygen_program = {};
    raygen_program.name = "raygen";
    raygen_program.options = {};
    raygen_program.desc = {};
    raygen_program.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_program.desc.raygen.entryFunctionName = "__raygen__extract_2d_plane";

    std::string pipeline_name = "pointcloud_pipeline";
    std::string module_name = raygen_program.name;
    std::string sbt_name = "pointcloud_sbt";
    corecast_optix::PointCloudRayGenData rg_data = { d_points, num_points };
    std::vector<std::string> program_names = { raygen_program.name };

    std::cout << "Creating module from extract_2d_plane.ptx" << std::endl;
    std::string ptx_path = CORECAST_POINTCLOUD_PTX_PATH;
    optix.create_module(module_name, pipeline_compile_options, module_compile_options, ptx_path);

    std::cout << "Adding program to module" << std::endl;
    optix.add_program_to_module(module_name, raygen_program);

    std::cout << "Building pipeline" << std::endl;
    optix.build_pipeline(pipeline_name, program_names, link_options);

    std::cout << "Creating SBT" << std::endl;
    optix.create_sbt<corecast_optix::SbtRecord<corecast_optix::PointCloudRayGenData>, corecast_optix::PointCloudRayGenData>(
        sbt_name, raygen_program.name, rg_data);

    std::cout << "Launching pipeline" << std::endl;
    optix.launch_pipeline(pipeline_name, params, sbt_name);

    // Copy inlier mask back and count
    std::vector<uint8_t> host_inlier_mask(num_points);
    CUDA_CHECK(cudaMemcpy(host_inlier_mask.data(), params.inlier_mask, num_points * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    size_t inlier_count = 0;
    for (uint8_t m : host_inlier_mask)
        if (m) ++inlier_count;

    std::cout << "Inlier count: " << inlier_count << " / " << num_points
              << " (expected ~" << num_inliers << ")" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.inlier_mask)));

    return 0;
}
