cmake_minimum_required(VERSION 3.16)
project(corecast_optix VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

set(DEFAULT_OPTIX_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64")
set(OPTIX_SDK_PATH "${DEFAULT_OPTIX_SDK_PATH}" CACHE PATH "Path to local NVIDIA OptiX SDK")
if(NOT OPTIX_SDK_PATH)
  message(FATAL_ERROR "OPTIX_SDK_PATH is not set. Pass -DOPTIX_SDK_PATH=/path/to/NVIDIA-OptiX-SDK")
endif()

if(NOT EXISTS "${OPTIX_SDK_PATH}/include/optix.h")
  message(FATAL_ERROR
    "optix.h not found in ${OPTIX_SDK_PATH}/include.\n"
    "For NVIDIA driver R580, use OptiX 9.0.0 (R570+ compatible).\n"
    "Place it at ${DEFAULT_OPTIX_SDK_PATH} or pass -DOPTIX_SDK_PATH=/path/to/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64"
  )
endif()

find_package(CUDAToolkit REQUIRED)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CORECAST_CUDA_STD_INCLUDE_DIR "" CACHE PATH "Include root that provides the full cuda/std header tree")
if(NOT CORECAST_CUDA_STD_INCLUDE_DIR)
  foreach(cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${cuda_include_dir}/cuda/std/version")
      set(CORECAST_CUDA_STD_INCLUDE_DIR "${cuda_include_dir}")
      break()
    endif()
    if(EXISTS "${cuda_include_dir}/cccl/cuda/std/version")
      set(CORECAST_CUDA_STD_INCLUDE_DIR "${cuda_include_dir}/cccl")
      break()
    endif()
  endforeach()
endif()
if(NOT CORECAST_CUDA_STD_INCLUDE_DIR)
  message(FATAL_ERROR
    "Full cuda/std support requested, but cuda/std headers were not found.\n"
    "Install a CUDA Toolkit that provides libcu++/CCCL or pass -DCORECAST_CUDA_STD_INCLUDE_DIR=/path/to/include-root"
  )
endif()
message(STATUS "Using cuda/std include root: ${CORECAST_CUDA_STD_INCLUDE_DIR}")

set(SAMPLES_INPUT_GENERATE_PTX 0)
set(SAMPLES_INPUT_GENERATE_OPTIXIR 0)
set(SAMPLES_DIR "")
set(SAMPLES_PTX_DIR "")
set(SAMPLES_CUDA_DIR "")
set(SAMPLES_RELATIVE_INCLUDE_DIRS "")
set(SAMPLES_ABSOLUTE_INCLUDE_DIRS "")
set(OPTIX_DEBUG_DEVICE_CODE 0)
set(OPTIX_OPTIXIR_BUILD_CONFIGURATION "")
set(CUDA_NVRTC_ENABLED 0)
set(CUDA_NVRTC_OPTIONS "")
set(CUDA_NVRTC_OPTIONS_DEBUG "")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file(
  "${OPTIX_SDK_PATH}/SDK/sampleConfig.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/sampleConfig.h"
  @ONLY
)

set(CORECAST_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/ptx/draw_solid_color.ptx")
set(CORECAST_POINTCLOUD_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/ptx/pointcloud_to_depth_map.ptx")

add_custom_command(
  OUTPUT "${CORECAST_PTX_FILE}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/ptx"
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    --ptx
    --std=c++20
    -I"${OPTIX_SDK_PATH}/include"
    -I"${OPTIX_SDK_PATH}/SDK"
    -I"${CORECAST_CUDA_STD_INCLUDE_DIR}"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include/corecast_optix"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    -I"${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples/draw_solid_color.cu"
    -o "${CORECAST_PTX_FILE}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples/draw_solid_color.cu"
  COMMENT "Generating OptiX PTX for draw_solid_color.cu"
)

add_custom_command(
  OUTPUT "${CORECAST_POINTCLOUD_PTX_FILE}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/ptx"
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    --ptx
    --std=c++20
    -I"${OPTIX_SDK_PATH}/include"
    -I"${OPTIX_SDK_PATH}/SDK"
    -I"${CORECAST_CUDA_STD_INCLUDE_DIR}"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include/corecast_optix"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/cuda"
    -I"${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda/pointcloud/pointcloud_to_depth_map.cu"
    -o "${CORECAST_POINTCLOUD_PTX_FILE}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/pointcloud/pointcloud_to_depth_map.cu"
  COMMENT "Generating OptiX PTX for pointcloud_to_depth_map.cu"
)

add_custom_target(corecast_optix_ptx DEPENDS "${CORECAST_PTX_FILE}" "${CORECAST_POINTCLOUD_PTX_FILE}")

add_library(corecast_optix
  src/corecast_coresac.cpp
  src/corecast_optix_program_registry.cpp
  src/corecast_workflow.cpp
)
add_library(corecast_optix::corecast_optix ALIAS corecast_optix)
add_dependencies(corecast_optix corecast_optix_ptx)
target_compile_definitions(corecast_optix PUBLIC CORECAST_PTX_PATH="${CORECAST_PTX_FILE}")
target_compile_definitions(corecast_optix PUBLIC CORECAST_POINTCLOUD_PTX_PATH="${CORECAST_POINTCLOUD_PTX_FILE}")

target_include_directories(corecast_optix
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples>
    $<BUILD_INTERFACE:${CORECAST_CUDA_STD_INCLUDE_DIR}>
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

#add_executable(corecast_optix_hello_world
#  examples/hello_world.cpp
#)

#target_link_libraries(corecast_optix_hello_world
#  PRIVATE
#    corecast_optix
#)

#add_executable(corecast_optix_pointcloud_hello_world
#  examples/hello_world_point_cloud.cpp
#  examples/hello_world_point_cloud_helpers.cpp
#)

#target_link_libraries(corecast_optix_pointcloud_hello_world
#  PRIVATE
#    corecast_optix
#)

install(TARGETS corecast_optix
  EXPORT corecast_optixTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(DIRECTORY include/
  DESTINATION include
)

set(CORECAST_OPTIX_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/corecast_optix")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/corecast_optixConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/corecast_optixConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/corecast_optixConfig.cmake"
  INSTALL_DESTINATION "${CORECAST_OPTIX_CONFIG_INSTALL_DIR}"
)

install(EXPORT corecast_optixTargets
  FILE corecast_optixTargets.cmake
  NAMESPACE corecast_optix::
  DESTINATION "${CORECAST_OPTIX_CONFIG_INSTALL_DIR}"
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/corecast_optixConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/corecast_optixConfigVersion.cmake"
  DESTINATION "${CORECAST_OPTIX_CONFIG_INSTALL_DIR}"
)

export(
  EXPORT corecast_optixTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/corecast_optixTargets.cmake"
  NAMESPACE corecast_optix::
)
