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

add_library(corecast_optix
  src/corecast_optix_context.cpp
)

target_include_directories(corecast_optix
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${OPTIX_SDK_PATH}/include>
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
