#include <any>
#include <corecast_optix/corecast_optix_types.hpp>
#include <memory>
#include <unordered_map>
import workflow;
import coresac_workflow;
namespace corecast::optix {

class CoreCastOptix : public std::enable_shared_from_this<CoreCastOptix> {
 public:
  CoreCastOptix();

  ~CoreCastOptix();

  std::shared_ptr<CoreSACWorkflow> create_coresac_pipeline(std::string& workflow_name,
                                                           WorkflowOptions& workflow_options);

  std::shared_ptr<std::any> get_workflow(std::string& workflow_name);

  void destroy_workflow(std::string& workflow_name);

 private:
  // std::unordered_map<std::string, ActualWorkflow> workflows_;
  // std::unordered_map<std::string, std::shared_ptr<BaseWorkflow>> workflows_;
};

}  // namespace corecast::optix