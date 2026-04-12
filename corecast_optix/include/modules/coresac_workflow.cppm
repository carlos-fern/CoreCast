module;

#include <corecast_optix/corecast_optix_types.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

export module coresac_workflow;

import workflow;

export namespace corecast::optix {

struct RayGenRecord {
  int data;
};

struct HitGroupRecord {
  int data;
};

struct MissRecord {
  int data;
};

class CoreSACWorkflow : private BaseWorkflow<CoreSACWorkflow> {
 public:
  CoreSACWorkflow(std::shared_ptr<CoreCastOptix> optix, std::optional<WorkflowOptions> workflow_options)
      : BaseWorkflow(optix, workflow_options) {};

 private:
};

}  // namespace corecast::optix