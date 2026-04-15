module;

#include <corecast_optix/corecast_optix_types.hpp>
#include <memory>
#include <optional>
#include <string>

export module hello_workflow;

import workflow;

// ---------------------------------------------------------------------------
// optixHello data types  (mirrors optixHello.h)
// ---------------------------------------------------------------------------
export namespace corecast::optix {

// GPU launch params  (optixHello: "struct Params")
struct HelloParams {
  unsigned char* image;  // device output RGBA buffer
  unsigned int image_width;
  unsigned int image_height;
};

template <typename T>
struct SbtRecord {
  alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

// Per-raygen SBT payload  (optixHello: "struct RayGenData { float r,g,b; }")
struct HelloRayGenData {
  float r, g, b;
};

// ============================================================================
// HelloWorkflow
//
// A direct translation of optixHello into the BaseWorkflow<T> CRTP pattern.
// The workflow is responsible ONLY for:
//   - declaring its data types
//   - filling in options structs
//   - telling the base *what* programs exist
//
// Everything that touches CUDA or OptiX APIs (cudaMalloc, optixLaunch, etc.)
// must live in BaseWorkflow.  If a step is not yet reachable through the base,
// a TODO comment marks what needs to be fixed there.
// ============================================================================
class HelloWorkflow : public BaseWorkflow<HelloWorkflow> {
 public:
  HelloWorkflow(std::shared_ptr<CoreCastOptix> optix, std::optional<WorkflowOptions> workflow_options)
      : BaseWorkflow(optix, workflow_options) {
    // -----------------------------------------------------------------------
    // Step 3: Describe program groups to the registry.
    //
    // optixHello lines 167-196: one raygen group + one null-miss group.
    // program_registry_ (owned by BaseWorkflow) does the actual
    // optixProgramGroupCreate call inside register_program().
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Step 5: Build the SBT.
    //
    // TODO (workflow.cppm BUG – create_sbt() / sbt_map_ disabled):
    //   BaseWorkflow::create_sbt() and sbt_map_ are commented out because of
    //   a known GCC modules key-function bug (workflow.cppm line 62).
    //   Until that is fixed the derived class cannot call create_sbt() and
    //   must leave SBT setup unimplemented.
    //   Fix needed: uncomment / restructure sbt_map_ in BaseWorkflow.
    //
    // When fixed, this is what HelloWorkflow would call:
    //   HelloRayGenRecord rg_record{};  rg_record.data = {0.462f, 0.725f, 0.f};
    //   HelloMissRecord   ms_record{};
    //   create_sbt("raygen", rg_record, ms_record, ms_record);
    // -----------------------------------------------------------------------
  }

  // -------------------------------------------------------------------------
  // load_data  (CRTP interface required by BaseWorkflow)
  // optixHello has no external input data; no-op here.
  // -------------------------------------------------------------------------
  auto load_data(auto /*data*/) {}

  // -------------------------------------------------------------------------
  // process  (CRTP interface required by BaseWorkflow)
  //
  // optixHello lines 279-300: stream create → cudaMalloc params →
  //   cudaMemcpy → optixLaunch → sync → cudaFree.
  // -------------------------------------------------------------------------
  void process() {
    HelloParams params{};
    // params.image = ... // managed output buffer
    params.image_width = 512;
    params.image_height = 512;

    // Launch pipeline via the base class wrapper
    execute_launch<HelloParams>("pipeline", "pipeline", "pipeline", params, params.image_width, params.image_height);
  }

  // -------------------------------------------------------------------------
  // get_result  (CRTP interface required by BaseWorkflow)
  // -------------------------------------------------------------------------
  auto get_result() { return true; }

  void define_modules() {
    module_.pipeline_compile_options_.usesMotionBlur = false;
    module_.pipeline_compile_options_.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    module_.pipeline_compile_options_.numPayloadValues = 2;
    module_.pipeline_compile_options_.numAttributeValues = 2;
    module_.pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    module_.pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";

    // Point to the compiled PTX; setup_module() reads ptx_path_ to decide
    // whether to call optixModuleCreate (custom) or optixBuiltinISModuleGet.
    module_.ptx_path_ = "draw_solid_color.ptx";  // TODO: wire real path via WorkflowOptions

    add_module("primary", module_);
  }

  void define_programs() {
    // Raygen: __raygen__draw_solid_color
    CoreCastProgram raygen_prog;
    raygen_prog.name = "raygen";
    raygen_prog.desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog.desc.raygen.module = module_.module_;
    raygen_prog.desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
    program_registry_->register_program(raygen_prog, module_);
    pipeline_.program_names_.push_back(raygen_prog.name);

    // Miss: null module + null entry (optixHello leaves these zeroed)
    CoreCastProgram miss_prog;
    miss_prog.name = "miss";
    miss_prog.desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    // module and entryFunctionName intentionally null (zero-init)
    program_registry_->register_program(miss_prog, module_);
    pipeline_.program_names_.push_back(miss_prog.name);
  }
  void define_pipelines() { add_pipeline("pipeline", pipeline_); }

  void define_gases() {};

  void define_sbts() {

    struct RayGenData { float r, g, b; };
    struct MissData { float r, g, b; };
    struct EmptyData {};

    SbtRecord<RayGenData> raygen_record;
    SbtRecord<MissData> miss_record;
    SbtRecord<EmptyData> hitgroup_record;

    // Use our actual records to build the SBT safely
    CoreCastOptixTraceSBT sbt(
        "raygen", "miss", "", *program_registry_, 
        raygen_record, miss_record, hitgroup_record
    );
    
    // Pass it back up to the base class map
    add_sbt("pipeline", sbt);
  };

  void define_launches() {};

 private:
  // module_ is passed by reference into setup_module() and register_program()
  // — those base methods take it as a parameter, so private access is fine.
  CoreCastOptixModule module_;
  CoreCastOptixPipeline pipeline_;

  // pipeline_ and pipeline_link_options_ would move into BaseWorkflow once
  // setup_pipelines() is fixed to not access derived members via CRTP.
  // See Step 4 TODO above.
};

}  // namespace corecast::optix
