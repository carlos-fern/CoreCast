#pragma once

#include <type_traits>

namespace corecast::optix {

template <typename ActualWorkflow>
class CoreCastWorkflow {

public: 

    CoreCastWorkflow(std::shared_ptr<CoreCastOptix> optix){
        optix_ = optix;
    }

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

    // Step 1: Initialize the context
    void init_context();

    // Step 2:Setup module
    void setup_module();

    // Step 3: Setup programs and program groups
    CoreCastOptixProgramRegistry program_registry_;
    
    // Step 4: Setup pipelines
    void setup_pipelines()

    // Step 5 setup GAS
    void setup_gas();

    //Step 6 : Setup SBT
    void setup_sbt();

    //Step 7 : setup launch
    void setup_launch();
};
} // namespace corecast::optix
