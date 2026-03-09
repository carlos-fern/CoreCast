cmake_minimum_required(VERSION 3.8)
project(corecast_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# Prefer local build-tree package config without hardcoding PATHS in find_package.
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../build/corecast_optix")
find_package(corecast_optix CONFIG REQUIRED)

add_executable(corecast_ros2_node src/corecast_ros2_node.cpp)
target_include_directories(corecast_ros2_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(corecast_ros2_node PUBLIC cxx_std_20)
ament_target_dependencies(corecast_ros2_node rclcpp sensor_msgs tf2 tf2_ros)
target_link_libraries(corecast_ros2_node corecast_optix::corecast_optix)

add_executable(corecast_ros2_pointcloud_sim_node src/corecast_ros2_pointcloud_sim_node.cpp)
target_compile_features(corecast_ros2_pointcloud_sim_node PUBLIC cxx_std_20)
ament_target_dependencies(corecast_ros2_pointcloud_sim_node rclcpp sensor_msgs geometry_msgs tf2 tf2_ros)

install(TARGETS corecast_ros2_node corecast_ros2_pointcloud_sim_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
