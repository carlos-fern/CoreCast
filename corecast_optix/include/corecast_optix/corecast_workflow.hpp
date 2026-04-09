#pragma once

#include <type_traits>
#include <corecast_optix/corecast_optix_types.hpp>
#include <functional>
#include <concepts>

namespace corecast::optix {

//Forward declaration
class CoreCastOptix;

template <typename T>
concept ContextLogCallbackType = std::invocable<T, unsigned int, const char*, const char*, void*>;

template <typename ActualWorkflow>
class CoreCastWorkflow {

public: 

    CoreCastWorkflow(std::shared_ptr<CoreCastOptix> optix, std::optional<WorkflowOptions> workflow_options);

    auto load_data(auto data) {
        return static_cast<ActualWorkflow&>(*this).load_data(data);
    }

    void process() {
        static_cast<ActualWorkflow&>(*this).process();
    }

    auto get_result() {
        return static_cast<ActualWorkflow&>(*this).get_result();
    }

protected:
    std::shared_ptr<CoreCastOptix> optix_;
    CoreCastOptixContext context_;
    CoreCastOptixProgramRegistry program_registry_;

    // Step 1: Initialize the context
    void init_context(CoreCastOptixContext& context);

    // Step 2:Setup module
    void setup_module(CoreCastOptixModule& module);
    
    // Step 4: Setup pipelines
    void setup_pipelines(CoreCastOptixPipeline& pipeline);

    // Step 5 setup GAS
    OptixTraversableHandle setup_gas(CoreCastOptixGAS& gas);

    //Step 6 : Setup SBT
    template<typename RecordType, typename DataType>
    void setup_sbt(CoreCastOptixSBT<RecordType, DataType>& sbt);

    //Step 7 : setup launch
    void setup_launch(CoreCastOptixLaunch& launch);

    // Default log callback for the context
    void context_log_cb(ContextLogCallbackType auto log_cb);
};
} // namespace corecast::optix
