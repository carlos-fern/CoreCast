#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include "corecast_optix/corecast_optix_context.hpp"
#include "corecast_optix/corecast_optix_module.hpp"

namespace corecast::optix
{
struct CoreCastProgram {
  std::string name;
  OptixProgramGroupOptions options{};
  OptixProgramGroupDesc desc{};
};

class CoreCastOptixProgramRegistry
{
public:
  /**
  * @brief Constructor for the CoreCastOptixProgramRegistry class.
  * @param context The context for the program registry.
  */
  CoreCastOptixProgramRegistry(std::shared_ptr<CoreCastOptixContext> context);

  /**
  * @brief Destructor for the CoreCastOptixProgramRegistry class.
  */
  ~CoreCastOptixProgramRegistry();

  /**
  * @brief Register a program with the program registry.
  * @param program The program to register.
  * @param module The module to register the program with.
  */
  void register_program(const CoreCastProgram& program, CoreCastOptixModule& module);

  /**
  * @brief Get a program group from the program registry.
  * @param name The name of the program group to get.
  * @return The program group.
  */
  OptixProgramGroup get_program_group(const std::string& name) const;

  /**
  * @brief Get program groups from the program registry.
  * @param names The names of the program groups to get.
  * @return The program groups.
  */
  std::vector<OptixProgramGroup> get_program_groups(const std::vector<std::string>& names) const;
    
private:
  std::shared_ptr<CoreCastOptixContext> context_;
  std::unordered_map<std::string, OptixProgramGroup> program_groups_;
  std::unordered_map<std::string, CoreCastProgram> program_descriptions_;

};

}  // namespace corecast::optix
