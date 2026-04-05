#include <corecast_optix_v2/corecast_workflow.hpp>


namespace corecast::processing {

struct CoreSACParams {
    unsigned int image_width;
    unsigned int image_height;
    
    // Sensor position and orientation basis vectors
    float3 sensor_origin;
    float3 sensor_x_axis;
    float3 sensor_y_axis;
    float3 sensor_z_axis;
    
    // Tracing limits
    float t_min;
    float t_max;
    
    // The 3D scene handle
    OptixTraversableHandle handle;
    
    OptixAabb *bounding_box;
    float3 *hit_points;
    uint32_t *hit_count;
    uint32_t max_hit_points;
    };
    
    struct CoreSACGroupParams {
    float3 *hit_points;
    uint32_t hit_point_count;
    
    float voxel_size;
    uint32_t max_num_total_voxels;
    uint32_t max_points_per_voxel;
    
    uint32_t *voxel_point_count;
    uint32_t *active_voxel_ids;
    float3 *voxel_points;
    };
    
    struct CoreSACAABBParams {
    uint32_t *voxel_point_count;
    uint32_t *active_voxel_ids;
    float3 *voxel_points;
    uint32_t max_num_total_voxels;
    uint32_t max_points_per_voxel;
    uint32_t min_points_per_voxel;
    
    OptixAabb *bounding_boxes;
    };
    
    struct CoreSACScoringParams {
    uint32_t max_num_total_voxels;
    CoreSACAABBParams aabb_params;
    };

class CoreSac : private corecast::optix::CoreCastWorkflow<CoreCastCoreSac> {

public:

    CoreSac(std::shared_ptr<CoreCastOptix> optix){

        
    configure_workflow();

    }

    auto load_data(auto data){

    }

    auto process(){

        

    }

private:
    
    CoreCastOptixModule module_;
    CoreCastOptixGAS gas_;
    CoreCastSBT sbt_;
    CoreCastLaunchParams launch_params_;

    void configure_workflow(){

        // Step 2: Setup module
        setup_module(module_);

        // Step 3: Setup programs and program groups
        setup_programs_and_program_groups();

        // Step 4: Setup pipelines
        setup_pipelines();

        // Step 5: Setup GAS
        setup_gas(gas_);

        // Step 6: Setup SBT
        setup_sbt(sbt_);

        // Step 7: Setup launch
        setup_launch(launch_params_);
    }


}
} // namespace corecast::processing