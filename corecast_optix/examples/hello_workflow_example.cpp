// hello_workflow_example.cpp
// Exercises HelloWorkflow using the CoreCastOptix facade,
// mirroring how optixHello.cpp drives OptiX from a single main().

#include <corecast_optix/corecast_optix.hpp>
#include <corecast_optix/corecast_optix_program_registry.hpp>  // gcc-14 modules workaround
#include <memory>

import hello_workflow;

int main() {
  auto corecast_optix = std::make_shared<corecast::optix::CoreCastOptix>();

  corecast::optix::WorkflowOptions opts{};

  // Steps 2-5 happen inside the HelloWorkflow constructor.
  // Width/height and ptx_path are embedded inside the workflow via WorkflowOptions
  // once workflow.cppm is extended to carry those fields.
  auto hello = std::make_shared<corecast::optix::HelloWorkflow>(
      corecast_optix, opts);

  // Step 6: Launch
  hello->process();

  return 0;
}
