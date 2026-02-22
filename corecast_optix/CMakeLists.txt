cmake_minimum_required(VERSION 3.16)
project(corecast_optix LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

set(OPTIX_SDK_PATH "" CACHE PATH "Path to local NVIDIA OptiX SDK")
if(NOT OPTIX_SDK_PATH)
  message(FATAL_ERROR "OPTIX_SDK_PATH is not set. Pass -DOPTIX_SDK_PATH=/path/to/NVIDIA-OptiX-SDK")
endif()

if(NOT EXISTS "${OPTIX_SDK_PATH}/include/optix.h")
  message(FATAL_ERROR "optix.h not found in ${OPTIX_SDK_PATH}/include. Check OPTIX_SDK_PATH.")
endif()

find_package(CUDAToolkit REQUIRED)

set(SAMPLES_DIR "${OPTIX_SDK_PATH}/SDK")
set(SAMPLES_PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/ptx")
set(SAMPLES_CUDA_DIR "${OPTIX_SDK_PATH}/SDK/cuda")
set(SAMPLES_RELATIVE_INCLUDE_DIRS {})
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS {})
set(OPTIX_OPTIXIR_BUILD_CONFIGURATION "Release")
set(CUDA_NVRTC_ENABLED 1)
set(CUDA_NVRTC_OPTIONS "{\"--std=c++17\"}")
set(CUDA_NVRTC_OPTIONS_DEBUG "{\"--std=c++17\"}")
set(SAMPLES_INPUT_GENERATE_OPTIXIR 1)
set(SAMPLES_INPUT_GENERATE_PTX 0)

configure_file(
  "${OPTIX_SDK_PATH}/SDK/sampleConfig.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/sampleConfig.h"
)

add_library(corecast_optix
  src/corecast_optix_context.cpp
  src/corecast_optix_program_registry.cpp
  src/corecast_optix_module.cpp
  src/corecast_optix_pipeline.cpp
)

target_include_directories(corecast_optix
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${OPTIX_SDK_PATH}/include>
    $<BUILD_INTERFACE:${OPTIX_SDK_PATH}/SDK>
    $<BUILD_INTERFACE:${OPTIX_SDK_PATH}/SDK/support>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(corecast_optix
  PUBLIC
    CUDA::cudart
)

install(TARGETS corecast_optix
  DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)
