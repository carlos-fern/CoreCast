#include <memory>
#include <corecast_optix_types.hpp>

namespace corecast::optix {

class CoreCastOptix : public std::enable_shared_from_this<CoreCastOptix> {
 public:

    CoreCastOptix(auto settings);

    std::shared_ptr<CoreCastWorkflow> create_coresac_pipeline(){

        return std::make_shared<CoreCastWorkflow>(shared_from_this(), workflow_options);
        
    }


    private:
    CoreCastOptixContext context_;


    template <typename WorkflowType>
    auto create_pipeline() -> decltype(WorkflowType){
        return WorkflowType(shared_from_this());
    }
};

} // namespace corecast::optix