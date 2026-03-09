#ifndef CORECAST_ROS2__CORECAST_ROS2_NODE_HPP_
#define CORECAST_ROS2__CORECAST_ROS2_NODE_HPP_

#include <rclcpp/rclcpp.hpp>

#include "corecast_ros2/corecast_ros2_depth_map.hpp"

namespace corecast::ros2 {

class CorecastRos2Node : public rclcpp::Node {
 public:
  CorecastRos2Node();

 private:
  CoreCastRos2DepthMap depth_map_processor_;
};

}  // namespace corecast::ros2

#endif  // CORECAST_ROS2__CORECAST_ROS2_NODE_HPP_
