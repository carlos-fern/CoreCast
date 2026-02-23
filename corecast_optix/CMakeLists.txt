cmake_minimum_required(VERSION 3.16)
project(corecast_optix LANGUAGES CXX)

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

set(CORECAST_PTX_FILE "${CMAKE_CURRENT_BINARY_DIR}/ptx/draw_solid_color.ptx")

add_custom_command(
  OUTPUT "${CORECAST_PTX_FILE}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/ptx"
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
    --ptx
    --std=c++17
    -I"${OPTIX_SDK_PATH}/include"
    -I"${OPTIX_SDK_PATH}/SDK"
    -I"${CORECAST_CUDA_STD_INCLUDE_DIR}"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
    -I"${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples"
    -I"${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples/draw_solid_color.cu"
    -o "${CORECAST_PTX_FILE}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_examples/draw_solid_color.cu"
  COMMENT "Generating OptiX PTX for draw_solid_color.cu"
)
add_custom_target(corecast_optix_ptx DEPENDS "${CORECAST_PTX_FILE}")

add_library(corecast_optix
  src/corecast_optix_context.cpp
  src/corecast_optix_program_registry.cpp
  src/corecast_optix_module.cpp
  src/corecast_optix_utils.cpp
  src/corecast_optix_pipeline.cpp
  src/corecast_optix.cpp
  src/corecast_optix_launch.cpp
)
add_dependencies(corecast_optix corecast_optix_ptx)
target_compile_definitions(corecast_optix PUBLIC CORECAST_PTX_PATH="${CORECAST_PTX_FILE}")

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

add_executable(corecast_optix_hello_world
  examples/hello_world.cpp
)

target_link_libraries(corecast_optix_hello_world
  PRIVATE
    corecast_optix
)

install(TARGETS corecast_optix
  DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)
