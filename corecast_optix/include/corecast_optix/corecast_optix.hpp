#include <memory>
#include <corecast_optix_types.hpp>

namespace corecast::optix {

class CoreCastOptix : public std::enable_shared_from_this<CoreCastOptix> {
 public:

    CoreCastOptix(auto settings);

    std::shared_ptr<CoreCastWorkflow> create_coresac_pipeline(std::string &workflow_name, workflow_options &workflow_options){

        auto workflow = make_shared<WorkflowType>(shared_from_this(), workflow_options);
        workflows_[workflow_name] = workflow;
        return workflow;
    }

    std::shared_ptr<CoreCastWorkflow> get_workflow(std::string &workflow_name){
        return workflows_[workflow_name];
    }

    void destroy_workflow(std::string &workflow_name){
        workflows_.erase(workflow_name);
    }


    private:
    unordered_map<std::string, WorkflowType> workflows_;


    template <typename WorkflowType>
    auto create_pipeline() -> decltype(WorkflowType){
        return WorkflowType(shared_from_this());
    }
};

} // namespace corecast::optix