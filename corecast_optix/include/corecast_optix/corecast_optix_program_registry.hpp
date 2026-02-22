#pragma once

#include <optix.h>

#include <cuda_runtime.h>

namespace corecast_optix
{
struct CoreCastProgram{
    std::string name;
    OptixProgramGroupOptions options;
    OptixProgramGroupDesc desc;
}

class CoreCastOptixProgramRegistry
{
public:
  CoreCastOptixProgramRegistry(std::shared_ptr<CoreCastOptixContext> context);
  ~CoreCastOptixProgramRegistry();

  void register_program(const CoreCastProgram& program, CoreCastOptixModule& module);
  OptixProgramGroup get_program_group(const std::string& name) const;
  std::vector<OptixProgramGroup> get_program_groups(const std::vector<std::string>& names) const;
    
private:
std::shared_ptr<CoreCastOptixContext> context_;
std::map<std::string, OptixProgramGroup> program_groups_;
std::map<std::string, CoreCastProgram> program_descriptions_;

};

}  // namespace corecast_optix
