cmake_minimum_required(VERSION 3.16)
project(corecast_optix)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(OPTIX_SDK_PATH "" CACHE PATH "Path to local NVIDIA OptiX SDK")
if(NOT OPTIX_SDK_PATH)
  message(FATAL_ERROR "OPTIX_SDK_PATH is not set. Pass -DOPTIX_SDK_PATH=/path/to/NVIDIA-OptiX-SDK")
endif()

if(NOT EXISTS "${OPTIX_SDK_PATH}/include/optix.h")
  message(FATAL_ERROR "optix.h not found in ${OPTIX_SDK_PATH}/include. Check OPTIX_SDK_PATH.")
endif()

add_library(${PROJECT_NAME}
  src/corecast_optix_context.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${OPTIX_SDK_PATH}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
)

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
