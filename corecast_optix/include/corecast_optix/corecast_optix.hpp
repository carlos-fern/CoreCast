#include <memory>
#include <unordered_map>
#include <any>

#include <corecast_optix/corecast_optix_types.hpp>
#include <corecast_optix/corecast_workflow.hpp>

namespace corecast::optix {


class CoreCastOptix : public std::enable_shared_from_this<CoreCastOptix> {
 public:

    CoreCastOptix(auto settings);

    ~CoreCastOptix();

    std::shared_ptr<std::any> create_coresac_pipeline(std::string &workflow_name, WorkflowOptions &workflow_options);

    std::shared_ptr<std::any> get_workflow(std::string &workflow_name);

    void destroy_workflow(std::string &workflow_name);

    private:
    //std::unordered_map<std::string, ActualWorkflow> workflows_;
};

} // namespace corecast::optix