#include <corecast_optix/corecast_optix.hpp>

namespace corecast::optix {

CoreCastOptix::CoreCastOptix(auto settings){

}

CoreCastOptix::~CoreCastOptix(){

}

std::shared_ptr<std::any> CoreCastOptix::create_coresac_pipeline(std::string &workflow_name, WorkflowOptions &workflow_options){
   // auto workflow = make_shared<CoreCastWorkflow>(shared_from_this(), workflow_options);
    //workflows_[workflow_name] = workflow;
   // return workflow;
}

std::shared_ptr<std::any> CoreCastOptix::get_workflow(std::string &workflow_name){
        //return workflows_[workflow_name];
}

void CoreCastOptix::destroy_workflow(std::string &workflow_name){
    //workflows_.erase(workflow_name);
}

} // namespace corecast::optix