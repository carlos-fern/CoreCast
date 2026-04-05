#pragma once

#include <type_traits>
#include <corecast_optix_v2/corecast_optix_types.hpp>

namespace corecast::optix {

template <typename ActualWorkflow>
class CoreCastWorkflow {

public: 

    CoreCastWorkflow(std::shared_ptr<CoreCastOptix> optix, std::optional<WorkflowOptions> workflow_options);

    auto load_data(auto data) -> decltype(static_cast<ActualWorkflow&>(*this).load_data()){
        return static_cast<ActualWorkflow&>(*this).load_data(data);
    }

    auto process() -> decltype(static_cast<ActualWorkflow&>(*this).process()){
        return static_cast<ActualWorkflow&>(*this).process();
    }

    auto get_result() -> decltype(static_cast<ActualWorkflow&>(*this).get_result()){
        return static_cast<ActualWorkflow&>(*this).get_result();
    }

private:
    std::shared_ptr<CoreCastOptix> optix_;
    CoreCastOptixContext context_;
    CoreCastOptixProgramRegistry program_registry_;

    // Step 1: Initialize the context
    void init_context(CoreCastOptixContext& context);

    // Step 2:Setup module
    void setup_module(CoreCastOptixModule& module);

    // Step 3: Setup programs and program groups
    CoreCastOptixProgramRegistry program_registry_;
    
    // Step 4: Setup pipelines
    void setup_pipelines(CoreCastOptixPipeline& pipeline);

    // Step 5 setup GAS
    OptixTraversableHandle setup_gas(CoreCastOptixGAS& gas);

    //Step 6 : Setup SBT
    void setup_sbt(CoreCastOptixSBT& sbt);

    //Step 7 : setup launch
    void setup_launch(CoreCastOptixLaunch& launch);

    concept ContextLogCallbackType = std::function<void(unsigned int level, const char* tag, const char* message, void*)>;
    // Default log callback for the context
    void context_log_cb(ContextLogCallbackType log_cb);
};
} // namespace corecast::optix
