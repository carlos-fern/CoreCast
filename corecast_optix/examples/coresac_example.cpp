#include <corecast_optix/corecast_optix.hpp>

int main() {
  auto corecast_optix = corecast::optix::CoreCastOptix();

  auto workflow_options = corecast::optix::WorkflowOptions();
  // Psuedo setup options
  // blah blah

  // auto coresac_pipeline = corecast_optix.create_coresac_pipeline(workflow_options);

  // coresac_pipeline.load_data(data);
  // coresac_pipeline.process();
  // auto result = coresac_pipeline.get_result();

  return 0;
}