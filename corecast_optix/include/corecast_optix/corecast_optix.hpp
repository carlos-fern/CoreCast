#include <memory>
#include <corecast_optix_types.hpp>

namespace corecast::optix {

class CoreCastOptix : public std::enable_shared_from_this<CoreCastOptix> {
 public:

    CoreCastOptix(auto settings);

    template <typename WorkflowType>
    auto create_pipeline() -> decltype(WorkflowType){
        return WorkflowType(shared_from_this());
    }

    private:
    CoreCastOptixContext context_;

};




} // namespace corecast::optix