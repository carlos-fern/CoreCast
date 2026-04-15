#include <optix_function_table_definition.h>

#include <corecast_optix/corecast_optix.hpp>
#include <corecast_optix/corecast_optix_program_registry.hpp>  // needed becusae of gcc 14 bug with modules
#include <memory>                                              // needed because of gcc 14 bug with modules

namespace corecast::optix {

CoreCastOptix::CoreCastOptix() {}

CoreCastOptix::~CoreCastOptix() {}

std::shared_ptr<CoreSACWorkflow> CoreCastOptix::create_coresac_pipeline(std::string& workflow_name,
                                                                        WorkflowOptions& workflow_options) {
  auto workflow = make_shared<CoreSACWorkflow>(shared_from_this(), workflow_options);
  // workflows_[workflow_name] = workflow;
  return workflow;
}

std::shared_ptr<std::any> CoreCastOptix::get_workflow(std::string& workflow_name) {
  // return workflows_[workflow_name];
}

void CoreCastOptix::destroy_workflow(std::string& workflow_name) {
  // workflows_.erase(workflow_name);
}

}  // namespace corecast::optix